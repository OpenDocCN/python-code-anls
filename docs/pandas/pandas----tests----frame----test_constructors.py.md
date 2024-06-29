# `D:\src\scipysrc\pandas\pandas\tests\frame\test_constructors.py`

```
# 导入必要的模块和库

import array  # 导入 Python 内置的 array 模块
from collections import (  # 导入 collections 模块中的多个类和函数
    OrderedDict,  # 有序字典类
    abc,  # 抽象基类模块
    defaultdict,  # 默认字典类
    namedtuple,  # 命名元组类
)
from collections.abc import Iterator  # 导入 collections.abc 模块中的迭代器抽象基类
from dataclasses import make_dataclass  # 导入 dataclasses 模块中的 make_dataclass 函数
from datetime import (  # 导入 datetime 模块中的多个类和函数
    date,  # 日期类
    datetime,  # 日期时间类
    timedelta,  # 时间间隔类
)
import functools  # 导入 functools 模块
import re  # 导入 re 模块，用于正则表达式操作
import zoneinfo  # 导入 zoneinfo 模块，用于时区信息操作

import numpy as np  # 导入 NumPy 库并重命名为 np
from numpy import ma  # 导入 NumPy 中的 ma 模块
from numpy.ma import mrecords  # 导入 NumPy 中的 mrecords 类
import pytest  # 导入 pytest 测试框架

from pandas._config import using_pyarrow_string_dtype  # 导入 pandas 内部模块中的配置信息
from pandas._libs import lib  # 导入 pandas 内部模块中的 lib 库
from pandas.compat.numpy import np_version_gt2  # 导入 pandas 兼容模块中的 NumPy 版本比较函数
from pandas.errors import IntCastingNaNError  # 导入 pandas 中的特定错误类

from pandas.core.dtypes.common import is_integer_dtype  # 导入 pandas 核心模块中的数据类型检查函数
from pandas.core.dtypes.dtypes import (  # 导入 pandas 核心模块中的数据类型类
    DatetimeTZDtype,  # 时区日期时间数据类型
    IntervalDtype,  # 区间数据类型
    NumpyEADtype,  # NumPy 扩展数组数据类型
    PeriodDtype,  # 时期数据类型
)

import pandas as pd  # 导入 Pandas 库并重命名为 pd
from pandas import (  # 从 Pandas 中导入多个类和函数
    Categorical,  # 分类数据类型
    CategoricalIndex,  # 分类索引类
    DataFrame,  # 数据框类
    DatetimeIndex,  # 日期时间索引类
    Index,  # 索引类
    Interval,  # 区间类
    MultiIndex,  # 多级索引类
    Period,  # 时期类
    RangeIndex,  # 范围索引类
    Series,  # 序列类
    Timedelta,  # 时间增量类
    Timestamp,  # 时间戳类
    cut,  # 切分函数
    date_range,  # 生成日期范围函数
    isna,  # 检查缺失值函数
)
import pandas._testing as tm  # 导入 pandas 内部测试模块并重命名为 tm
from pandas.arrays import (  # 从 Pandas 的 arrays 模块中导入多个数组类
    DatetimeArray,  # 日期时间数组类
    IntervalArray,  # 区间数组类
    PeriodArray,  # 时期数组类
    SparseArray,  # 稀疏数组类
    TimedeltaArray,  # 时间增量数组类
)

MIXED_FLOAT_DTYPES = ["float16", "float32", "float64"]  # 混合浮点数数据类型列表
MIXED_INT_DTYPES = [  # 混合整数数据类型列表
    "uint8",  # 无符号 8 位整数
    "uint16",  # 无符号 16 位整数
    "uint32",  # 无符号 32 位整数
    "uint64",  # 无符号 64 位整数
    "int8",  # 有符号 8 位整数
    "int16",  # 有符号 16 位整数
    "int32",  # 有符号 32 位整数
    "int64",  # 有符号 64 位整数
]


class TestDataFrameConstructors:
    def test_constructor_from_ndarray_with_str_dtype(self):
        # 如果不在 ensure_str_array 周围进行 ravel/reshape，我们得到的将是一个字符串数组，每个元素例如 "[0 1 2]"
        arr = np.arange(12).reshape(4, 3)  # 创建一个 4x3 的 NumPy 数组
        df = DataFrame(arr, dtype=str)  # 使用数组创建一个数据框，并指定数据类型为字符串
        expected = DataFrame(arr.astype(str), dtype=object)  # 创建预期的数据框，将数组元素类型转换为字符串类型
        tm.assert_frame_equal(df, expected)  # 断言两个数据框相等

    def test_constructor_from_2d_datetimearray(self):
        dti = date_range("2016-01-01", periods=6, tz="US/Pacific")  # 创建一个包含时区信息的日期时间索引
        dta = dti._data.reshape(3, 2)  # 将日期时间索引的数据重新排列成 3x2 的数组形式

        df = DataFrame(dta)  # 使用数组创建一个数据框
        expected = DataFrame({0: dta[:, 0], 1: dta[:, 1]})  # 创建预期的数据框，指定列名为 0 和 1
        tm.assert_frame_equal(df, expected)  # 断言两个数据框相等
        # GH#44724 如果解开合并后性能显著下降
        assert len(df._mgr.blocks) == 1  # 断言数据框内部的块数量为 1

    def test_constructor_dict_with_tzaware_scalar(self):
        # GH#42505
        dt = Timestamp("2019-11-03 01:00:00-0700").tz_convert("America/Los_Angeles")  # 创建一个带有时区信息的时间戳并进行时区转换
        dt = dt.as_unit("ns")  # 将时间戳转换为纳秒单位

        df = DataFrame({"dt": dt}, index=[0])  # 使用包含时区信息的标量创建数据框，并指定索引
        expected = DataFrame({"dt": [dt]})  # 创建预期的数据框，列名为 dt，值为包含时区信息的列表
        tm.assert_frame_equal(df, expected)  # 断言两个数据框相等

        # 非同质
        df = DataFrame({"dt": dt, "value": [1]})  # 使用包含时区信息的标量和一个值创建数据框
        expected = DataFrame({"dt": [dt], "value": [1]})  # 创建预期的数据框，包含列 dt 和 value
        tm.assert_frame_equal(df, expected)  # 断言两个数据框相等
    def test_construct_ndarray_with_nas_and_int_dtype(self):
        # GH#26919 match Series by not casting np.nan to meaningless int
        # 创建一个包含 NaN 的 NumPy 数组，保证不将 np.nan 转换为无意义的整数
        arr = np.array([[1, np.nan], [2, 3]])
        # 设置错误消息的模式，用于匹配预期的异常类型和消息内容
        msg = r"Cannot convert non-finite values \(NA or inf\) to integer"
        # 使用 pytest 检查是否引发 IntCastingNaNError 异常，并匹配特定的错误消息
        with pytest.raises(IntCastingNaNError, match=msg):
            # 尝试使用数组构造 DataFrame，期望引发特定的类型转换异常
            DataFrame(arr, dtype="i8")

        # 检查该测试是否与 Series 的行为一致
        with pytest.raises(IntCastingNaNError, match=msg):
            # 以相同的方式检查 Series 的行为
            Series(arr[0], dtype="i8", name=0)

    def test_construct_from_list_of_datetimes(self):
        # 使用当前时间构造一个包含两个日期时间的 DataFrame
        df = DataFrame([datetime.now(), datetime.now()])
        # 断言 DataFrame 的第一列数据类型为 np.datetime64[us]
        assert df[0].dtype == np.dtype("M8[us]")

    def test_constructor_from_tzaware_datetimeindex(self):
        # 不对带有时区信息的 DatetimeIndex 进行强制类型转换，保持其为对象类型
        # GH#6032
        naive = DatetimeIndex(["2013-1-1 13:00", "2013-1-2 14:00"], name="B")
        # 将 naive 的时区设置为 "US/Pacific"
        idx = naive.tz_localize("US/Pacific")

        # 创建一个预期的 Series，使用带有时区的索引，数据类型为对象
        expected = Series(np.array(idx.tolist(), dtype="object"), name="B")
        # 断言预期的数据类型与索引的数据类型相同
        assert expected.dtype == idx.dtype

        # 将索引转换为 Series
        result = Series(idx)
        # 使用 pytest 的 assert_series_equal 方法检查结果与预期的一致性
        tm.assert_series_equal(result, expected)

    def test_columns_with_leading_underscore_work_with_to_dict(self):
        # 定义一个以下划线开头的列名
        col_underscore = "_b"
        # 创建一个 DataFrame，包含两列 "a" 和以下划线开头的列
        df = DataFrame({"a": [1, 2], col_underscore: [3, 4]})
        # 使用 DataFrame 的 to_dict 方法，将数据以 records 方式转换为字典列表
        d = df.to_dict(orient="records")

        # 定义预期的字典列表
        ref_d = [{"a": 1, col_underscore: 3}, {"a": 2, col_underscore: 4}]
        # 断言实际转换结果与预期的一致
        assert ref_d == d

    def test_columns_with_leading_number_and_underscore_work_with_to_dict(self):
        # 定义一个以数字和下划线开头的列名
        col_with_num = "1_b"
        # 创建一个 DataFrame，包含两列 "a" 和以数字和下划线开头的列
        df = DataFrame({"a": [1, 2], col_with_num: [3, 4]})
        # 使用 DataFrame 的 to_dict 方法，将数据以 records 方式转换为字典列表
        d = df.to_dict(orient="records")

        # 定义预期的字典列表
        ref_d = [{"a": 1, col_with_num: 3}, {"a": 2, col_with_num: 4}]
        # 断言实际转换结果与预期的一致
        assert ref_d == d

    def test_array_of_dt64_nat_with_td64dtype_raises(self, frame_or_series):
        # GH#39462
        # 创建一个包含 NaT（Not a Time）的 np.datetime64 数组，数据类型为对象
        nat = np.datetime64("NaT", "ns")
        arr = np.array([nat], dtype=object)
        if frame_or_series is DataFrame:
            # 如果参数是 DataFrame，则将数组重塑为 1x1 的形状
            arr = arr.reshape(1, 1)

        # 设置错误消息，用于匹配预期的异常类型和消息内容
        msg = "Invalid type for timedelta scalar: <class 'numpy.datetime64'>"
        # 使用 pytest 检查是否引发 TypeError 异常，并匹配特定的错误消息
        with pytest.raises(TypeError, match=msg):
            # 使用 frame_or_series 参数构造 DataFrame 或 Series，期望引发特定的类型转换异常
            frame_or_series(arr, dtype="m8[ns]")

    @pytest.mark.parametrize("kind", ["m", "M"])
    # 测试带有对象数据类型的日期时间类值，根据给定的 kind 和 frame_or_series 参数，执行相应的测试
    def test_datetimelike_values_with_object_dtype(self, kind, frame_or_series):
        # 根据 kind 的取值确定 dtype 和 scalar_type 的类型
        # 如果 kind 是 "M"，则 dtype 是 "M8[ns]"，scalar_type 是 Timestamp
        if kind == "M":
            dtype = "M8[ns]"
            scalar_type = Timestamp
        else:
            dtype = "m8[ns]"
            scalar_type = Timedelta

        # 创建一个包含日期时间数据的 numpy 数组 arr
        arr = np.arange(6, dtype="i8").view(dtype).reshape(3, 2)
        # 根据 frame_or_series 的类型选择性地处理数组 arr
        if frame_or_series is Series:
            arr = arr[:, 0]

        # 使用 frame_or_series 函数创建对象 obj，数据类型为 object
        obj = frame_or_series(arr, dtype=object)
        # 断言对象 obj 的第一个数据块的数据类型为 object
        assert obj._mgr.blocks[0].values.dtype == object
        # 断言对象 obj 的第一个数据块的扁平化后的第一个元素的类型为 scalar_type（Timestamp 或 Timedelta）
        assert isinstance(obj._mgr.blocks[0].values.ravel()[0], scalar_type)

        # 使用 frame_or_series 函数创建对象 obj，内部调用 frame_or_series 处理 arr，数据类型为 object
        obj = frame_or_series(frame_or_series(arr), dtype=object)
        # 断言对象 obj 的第一个数据块的数据类型为 object
        assert obj._mgr.blocks[0].values.dtype == object
        # 断言对象 obj 的第一个数据块的扁平化后的第一个元素的类型为 scalar_type
        assert isinstance(obj._mgr.blocks[0].values.ravel()[0], scalar_type)

        # 使用 frame_or_series 函数创建对象 obj，内部调用 frame_or_series 处理 arr，数据类型为 NumpyEADtype(object)
        obj = frame_or_series(frame_or_series(arr), dtype=NumpyEADtype(object))
        # 断言对象 obj 的第一个数据块的数据类型为 object
        assert obj._mgr.blocks[0].values.dtype == object
        # 断言对象 obj 的第一个数据块的扁平化后的第一个元素的类型为 scalar_type
        assert isinstance(obj._mgr.blocks[0].values.ravel()[0], scalar_type)

        # 如果 frame_or_series 是 DataFrame 类型，处理不同的路径，调用 internals.construction
        if frame_or_series is DataFrame:
            # 创建包含多个 Series 对象的列表 sers，每个 Series 对象的数据来源于 arr 的不同行
            sers = [Series(x) for x in arr]
            # 使用 frame_or_series 函数创建对象 obj，数据类型为 object
            obj = frame_or_series(sers, dtype=object)
            # 断言对象 obj 的第一个数据块的数据类型为 object
            assert obj._mgr.blocks[0].values.dtype == object
            # 断言对象 obj 的第一个数据块的扁平化后的第一个元素的类型为 scalar_type
            assert isinstance(obj._mgr.blocks[0].values.ravel()[0], scalar_type)

    # 测试 Series 对象的名称与列名不匹配的情况
    def test_series_with_name_not_matching_column(self):
        # GH#9232
        # 创建两个 Series 对象 x 和 y，分别设置不匹配的名称和列名
        x = Series(range(5), name=1)
        y = Series(range(5), name=0)

        # 创建 DataFrame 对象 result，使用 Series x 作为数据和列名为 [0]
        result = DataFrame(x, columns=[0])
        # 创建预期的空 DataFrame 对象 expected，列名为 [0]
        expected = DataFrame([], columns=[0])
        # 使用测试工具函数 tm.assert_frame_equal 检查 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)

        # 创建 DataFrame 对象 result，使用 Series y 作为数据和列名为 [1]
        result = DataFrame(y, columns=[1])
        # 创建预期的空 DataFrame 对象 expected，列名为 [1]
        expected = DataFrame([], columns=[1])
        # 使用测试工具函数 tm.assert_frame_equal 检查 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)

    # 使用 pytest.mark.parametrize 装饰器，参数化测试空构造函数的多种输入方式
    @pytest.mark.parametrize(
        "constructor",
        [
            lambda: DataFrame(),       # 使用无参数构造函数创建 DataFrame
            lambda: DataFrame(None),   # 使用 None 创建 DataFrame
            lambda: DataFrame(()),     # 使用空元组创建 DataFrame
            lambda: DataFrame([]),     # 使用空列表创建 DataFrame
            lambda: DataFrame(_ for _ in []),  # 使用生成器表达式创建 DataFrame
            lambda: DataFrame(range(0)),       # 使用 range(0) 创建 DataFrame
            lambda: DataFrame(data=None),      # 使用 data=None 创建 DataFrame
            lambda: DataFrame(data=()),        # 使用 data=() 创建 DataFrame
            lambda: DataFrame(data=[]),        # 使用 data=[] 创建 DataFrame
            lambda: DataFrame(data=(_ for _ in [])),  # 使用生成器表达式创建 DataFrame
            lambda: DataFrame(data=range(0)),   # 使用 data=range(0) 创建 DataFrame
        ],
    )
    # 测试空构造函数的各种输入方式，确保生成的 DataFrame 对象是空的，并且与预期结果 expected 相等
    def test_empty_constructor(self, constructor):
        expected = DataFrame()
        # 调用参数化的 constructor 函数生成 DataFrame 对象 result
        result = constructor()
        # 断言生成的 DataFrame 对象 result 的行数为 0
        assert len(result.index) == 0
        # 断言生成的 DataFrame 对象 result 的列数为 0
        assert len(result.columns) == 0
        # 使用测试工具函数 tm.assert_frame_equal 检查 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)

    # 测试带有对象索引的空构造函数
    def test_empty_constructor_object_index(self):
        # 创建预期的空 DataFrame 对象 expected，行索引为 RangeIndex(0)，列索引为 RangeIndex(0)
        expected = DataFrame(index=RangeIndex(0), columns=RangeIndex(0))
        # 使用空字典 {} 创建 DataFrame 对象 result
        result = DataFrame({})
        # 断言生成的 DataFrame 对象 result 的行数为 0
        assert len(result.index) == 0
        # 断言生成的 DataFrame 对象 result 的列数为 0
        assert len(result.columns) == 0
        # 使用测试工具函数 tm.assert_frame_equal 检查 result 和 expected 是否相等，并检查索引类型是否匹配
        tm.assert_frame_equal(result, expected, check_index_type=True)
    # 使用 pytest 的 parametrize 装饰器来定义多个参数化的测试用例
    @pytest.mark.parametrize(
        "emptylike,expected_index,expected_columns",
        [
            # 空列表 [[]] 作为构造函数参数时，期望的索引为 RangeIndex(1)，列索引为 RangeIndex(0)
            ([[]], RangeIndex(1), RangeIndex(0)),
            # 空列表 [[], []] 作为构造函数参数时，期望的索引为 RangeIndex(2)，列索引为 RangeIndex(0)
            ([[], []], RangeIndex(2), RangeIndex(0)),
            # 生成器表达式 (_ for _ in []) 作为构造函数参数时，期望的索引为 RangeIndex(1)，列索引为 RangeIndex(0)
            ([(_ for _ in [])], RangeIndex(1), RangeIndex(0)),
        ],
    )
    # 测试空对象类似的 DataFrame 构造函数
    def test_emptylike_constructor(self, emptylike, expected_index, expected_columns):
        # 根据期望的索引和列索引创建预期的 DataFrame 对象
        expected = DataFrame(index=expected_index, columns=expected_columns)
        # 使用空对象类似的输入参数构造 DataFrame 对象
        result = DataFrame(emptylike)
        # 使用测试框架的 assert_frame_equal 方法来比较 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)

    # 测试混合类型构造函数
    def test_constructor_mixed(self, float_string_frame, using_infer_string):
        # 如果 using_infer_string 为 True，则 dtype 为 "string"，否则为 np.object_
        dtype = "string" if using_infer_string else np.object_
        # 断言 float_string_frame 的列 'foo' 的数据类型是否为预期的 dtype
        assert float_string_frame["foo"].dtype == dtype

    # 测试构造函数中类型转换失败的情况
    def test_constructor_cast_failure(self):
        # 当构造 DataFrame 时，如果无法将字符串转换为浮点数，则应该抛出 ValueError 异常，并且异常信息应包含指定的错误消息
        msg = "could not convert string to float"
        with pytest.raises(ValueError, match=msg):
            DataFrame({"a": ["a", "b", "c"]}, dtype=np.float64)

        # 测试在构造 DataFrame 时使用奇怪的数组
        df = DataFrame(np.ones((4, 2)))

        # 下面的操作是允许的
        df["foo"] = np.ones((4, 2)).tolist()

        # 下面的操作应该引发 ValueError 异常，异常信息应包含指定的错误消息
        msg = "Expected a 1D array, got an array with shape \\(4, 2\\)"
        with pytest.raises(ValueError, match=msg):
            df["test"] = np.ones((4, 2))

        # 下面的操作是允许的
        df["foo2"] = np.ones((4, 2)).tolist()

    # 测试构造函数中 dtype 参数为复制模式时的行为
    def test_constructor_dtype_copy(self):
        # 创建原始的 DataFrame 对象
        orig_df = DataFrame({"col1": [1.0], "col2": [2.0], "col3": [3.0]})

        # 使用 dtype=float 和 copy=True 参数构造新的 DataFrame 对象
        new_df = DataFrame(orig_df, dtype=float, copy=True)

        # 修改新 DataFrame 中 'col1' 列的值，断言原始 DataFrame 中对应位置的值未被改变
        new_df["col1"] = 200.0
        assert orig_df["col1"][0] == 1.0

    # 测试构造函数中 dtype 参数为非转换模式时的 DataFrame 视图行为
    def test_constructor_dtype_nocast_view_dataframe(self):
        # 创建包含单个列表的 DataFrame 对象
        df = DataFrame([[1, 2]])
        # 使用与原始 DataFrame 相同的 dtype 参数构造另一个 DataFrame 对象，应该是原 DataFrame 的视图
        should_be_view = DataFrame(df, dtype=df[0].dtype)
        # 修改 should_be_view 的数据，并断言原始 DataFrame 对应位置的值未被改变
        should_be_view.iloc[0, 0] = 99
        assert df.values[0, 0] == 1

    # 测试构造函数中 dtype 参数为非转换模式时的 2D 数组行为
    def test_constructor_dtype_nocast_view_2d_array(self):
        # 创建具有指定 dtype 的 2D 数组 DataFrame 对象
        df = DataFrame([[1, 2], [3, 4]], dtype="int64")
        # 使用与原始 DataFrame 相同的 dtype 参数构造另一个 DataFrame 对象，应该是原 DataFrame 的视图
        df2 = DataFrame(df.values, dtype=df[0].dtype)
        # 断言 df2 的内部数据块是 C 连续的
        assert df2._mgr.blocks[0].values.flags.c_contiguous

    # 使用 pytest.mark.xfail 标记的测试，测试 1D 对象数组作为构造函数参数时是否复制
    @pytest.mark.xfail(using_pyarrow_string_dtype(), reason="conversion copies")
    def test_1d_object_array_does_not_copy(self):
        # 创建包含对象的 1D 数组
        arr = np.array(["a", "b"], dtype="object")
        # 使用 copy=False 参数构造 DataFrame 对象
        df = DataFrame(arr, copy=False)
        # 断言 df 的值与原始数组共享内存
        assert np.shares_memory(df.values, arr)

    # 使用 pytest.mark.xfail 标记的测试，测试 2D 对象数组作为构造函数参数时是否复制
    @pytest.mark.xfail(using_pyarrow_string_dtype(), reason="conversion copies")
    def test_2d_object_array_does_not_copy(self):
        # 创建包含对象的 2D 数组
        arr = np.array([["a", "b"], ["c", "d"]], dtype="object")
        # 使用 copy=False 参数构造 DataFrame 对象
        df = DataFrame(arr, copy=False)
        # 断言 df 的值与原始数组共享内存
        assert np.shares_memory(df.values, arr)
    def test_constructor_dtype_list_data(self):
        # 创建一个包含对象类型的 DataFrame，数据为二维列表
        df = DataFrame([[1, "2"], [None, "a"]], dtype=object)
        # 断言第二行第一列的值为 None
        assert df.loc[1, 0] is None
        # 断言第一行第二列的值为 "2"
        assert df.loc[0, 1] == "2"

    def test_constructor_list_of_2d_raises(self):
        # 测试当传入不符合要求的数据时是否抛出 ValueError 异常
        # https://github.com/pandas-dev/pandas/issues/32289
        a = DataFrame()
        b = np.empty((0, 0))
        with pytest.raises(ValueError, match=r"shape=\(1, 0, 0\)"):
            DataFrame([a])

        with pytest.raises(ValueError, match=r"shape=\(1, 0, 0\)"):
            DataFrame([b])

        # 创建包含一列的 DataFrame，并尝试传入两个相同结构的 DataFrame，期望抛出异常
        a = DataFrame({"A": [1, 2]})
        with pytest.raises(ValueError, match=r"shape=\(2, 2, 1\)"):
            DataFrame([a, a])

    @pytest.mark.parametrize(
        "typ, ad",
        [
            # 混合浮点数和整数存在于同一帧中
            ["float", {}],
            # 添加多种类型
            ["float", {"A": 1, "B": "foo", "C": "bar"}],
            # GH 622
            ["int", {}],
        ],
    )
    def test_constructor_mixed_dtypes(self, typ, ad):
        if typ == "int":
            # 如果类型是整数，使用预定义的混合整数类型
            dtypes = MIXED_INT_DTYPES
            # 创建包含随机整数的数组列表
            arrays = [
                np.array(np.random.default_rng(2).random(10), dtype=d) for d in dtypes
            ]
        elif typ == "float":
            # 如果类型是浮点数，使用预定义的混合浮点数类型
            dtypes = MIXED_FLOAT_DTYPES
            # 创建包含随机浮点数的数组列表
            arrays = [
                np.array(np.random.default_rng(2).integers(10, size=10), dtype=d)
                for d in dtypes
            ]

        # 断言每个数组的数据类型符合预期
        for d, a in zip(dtypes, arrays):
            assert a.dtype == d
        # 更新 ad 字典，将每种数据类型对应的数组加入其中
        ad.update(dict(zip(dtypes, arrays)))
        # 创建 DataFrame 对象
        df = DataFrame(ad)

        # 定义包含所有混合类型的数据类型列表
        dtypes = MIXED_FLOAT_DTYPES + MIXED_INT_DTYPES
        # 断言 DataFrame 中每种数据类型列的数据类型符合预期
        for d in dtypes:
            if d in df:
                assert df.dtypes[d] == d

    def test_constructor_complex_dtypes(self):
        # GH10952
        # 创建复数类型的随机数组
        a = np.random.default_rng(2).random(10).astype(np.complex64)
        b = np.random.default_rng(2).random(10).astype(np.complex128)

        # 创建包含复数类型列的 DataFrame
        df = DataFrame({"a": a, "b": b})
        # 断言 DataFrame 中列 'a' 的数据类型符合预期
        assert a.dtype == df.a.dtype
        # 断言 DataFrame 中列 'b' 的数据类型符合预期
        assert b.dtype == df.b.dtype

    def test_constructor_dtype_str_na_values(self, string_dtype):
        # 测试在指定字符串类型的情况下处理 NaN 值的正确性
        # https://github.com/pandas-dev/pandas/issues/21083
        # 创建包含字符串类型列的 DataFrame，包含 NaN 值
        df = DataFrame({"A": ["x", None]}, dtype=string_dtype)
        # 检查处理后的 DataFrame 是否与预期结果相等
        result = df.isna()
        expected = DataFrame({"A": [False, True]})
        tm.assert_frame_equal(result, expected)
        # 断言第二行第一列的值为 None
        assert df.iloc[1, 0] is None

        # 创建包含字符串类型列的 DataFrame，包含 NaN 值（使用 np.nan 表示）
        df = DataFrame({"A": ["x", np.nan]}, dtype=string_dtype)
        # 断言第二行第一列的值为 NaN
        assert np.isnan(df.iloc[1, 0])
    # 测试函数，用于测试构造函数处理记录类型的DataFrame
    def test_constructor_rec(self, float_frame):
        # 将浮点数据框转换为记录数组
        rec = float_frame.to_records(index=False)
        # 反转记录数组的字段名顺序
        rec.dtype.names = list(rec.dtype.names)[::-1]

        # 获取浮点数据框的索引
        index = float_frame.index

        # 从记录数组创建DataFrame
        df = DataFrame(rec)
        # 断言DataFrame的列索引与记录数组的字段名索引相等
        tm.assert_index_equal(df.columns, Index(rec.dtype.names))

        # 在指定索引处从记录数组创建DataFrame
        df2 = DataFrame(rec, index=index)
        # 断言DataFrame的列索引与记录数组的字段名索引相等
        tm.assert_index_equal(df2.columns, Index(rec.dtype.names))
        # 断言DataFrame的行索引与原始索引相等
        tm.assert_index_equal(df2.index, index)

        # 处理列与从数据推断的列不匹配的情况
        rng = np.arange(len(rec))[::-1]
        df3 = DataFrame(rec, index=rng, columns=["C", "B"])
        # 从具有指定列的记录数组创建期望的DataFrame，并重新索引列
        expected = DataFrame(rec, index=rng).reindex(columns=["C", "B"])
        # 断言DataFrame与期望的DataFrame相等
        tm.assert_frame_equal(df3, expected)

    # 测试函数，用于测试处理布尔类型数据的构造函数
    def test_constructor_bool(self):
        # 创建具有布尔数据的DataFrame
        df = DataFrame({0: np.ones(10, dtype=bool), 1: np.zeros(10, dtype=bool)})
        # 断言DataFrame的值的数据类型为布尔类型
        assert df.values.dtype == np.bool_

    # 测试函数，用于测试处理溢出到int64的构造函数
    def test_constructor_overflow_int64(self):
        # 创建大整数数组，测试溢出情况
        values = np.array([2**64 - i for i in range(1, 10)], dtype=np.uint64)

        # 从大整数数组创建DataFrame
        result = DataFrame({"a": values})
        # 断言DataFrame列"a"的数据类型为uint64
        assert result["a"].dtype == np.uint64

        # 创建数据并从数组创建DataFrame，测试另一种溢出情况
        data_scores = [
            (6311132704823138710, 273),
            (2685045978526272070, 23),
            (8921811264899370420, 45),
            (17019687244989530680, 270),
            (9930107427299601010, 273),
        ]
        dtype = [("uid", "u8"), ("score", "u8")]
        data = np.zeros((len(data_scores),), dtype=dtype)
        data[:] = data_scores
        df_crawls = DataFrame(data)
        # 断言DataFrame列"uid"的数据类型为uint64
        assert df_crawls["uid"].dtype == np.uint64

    # 使用pytest参数化标记，测试处理整数溢出的构造函数
    @pytest.mark.parametrize(
        "values",
        [
            np.array([2**64], dtype=object),
            np.array([2**65]),
            [2**64 + 1],
            np.array([-(2**63) - 4], dtype=object),
            np.array([-(2**64) - 1]),
            [-(2**65) - 2],
        ],
    )
    def test_constructor_int_overflow(self, values):
        # 创建对象数组，并从中创建DataFrame，测试整数溢出情况
        value = values[0]
        result = DataFrame(values)

        # 断言DataFrame第一列的数据类型为对象类型
        assert result[0].dtype == object
        # 断言DataFrame第一列的第一个元素等于value
        assert result[0][0] == value

    # 使用pytest参数化标记，测试处理numpy无符号整数的构造函数
    @pytest.mark.parametrize(
        "values",
        [
            np.array([1], dtype=np.uint16),
            np.array([1], dtype=np.uint32),
            np.array([1], dtype=np.uint64),
            [np.uint16(1)],
            [np.uint32(1)],
            [np.uint64(1)],
        ],
    )
    def test_constructor_numpy_uints(self, values):
        # 创建无符号整数数组，并从中创建DataFrame
        value = values[0]
        result = DataFrame(values)

        # 断言DataFrame第一列的数据类型与value的数据类型相等
        assert result[0].dtype == value.dtype
        # 断言DataFrame第一列的第一个元素等于value
        assert result[0][0] == value

    # 测试函数，用于测试处理OrderedDict的构造函数
    def test_constructor_ordereddict(self):
        # 创建包含有序字典的DataFrame
        nitems = 100
        nums = list(range(nitems))
        np.random.default_rng(2).shuffle(nums)
        expected = [f"A{i:d}" for i in nums]
        df = DataFrame(OrderedDict(zip(expected, [[0]] * nitems)))
        # 断言DataFrame的列名与预期的列名列表相等
        assert expected == list(df.columns)
    def test_constructor_dict(self):
        # 创建一个包含 30 个浮点数的 Series 对象，以 '2020-01-01' 为起始日期
        datetime_series = Series(
            np.arange(30, dtype=np.float64), index=date_range("2020-01-01", periods=30)
        )
        
        # 从 datetime_series 中获取从索引 5 开始的子序列
        # 预期此操作结果长度为 25
        datetime_series_short = datetime_series[5:]

        # 创建一个 DataFrame 对象，包含两列 'col1' 和 'col2'，分别使用 datetime_series 和 datetime_series_short
        frame = DataFrame({"col1": datetime_series, "col2": datetime_series_short})

        # 断言 datetime_series 长度为 30
        assert len(datetime_series) == 30
        # 断言 datetime_series_short 长度为 25
        assert len(datetime_series_short) == 25

        # 断言 frame 中 'col1' 列与重命名后的 datetime_series 相等
        tm.assert_series_equal(frame["col1"], datetime_series.rename("col1"))

        # 创建一个预期的 Series 对象 exp，包含 'col2' 列的预期值，前面填充 NaN
        exp = Series(
            np.concatenate([[np.nan] * 5, datetime_series_short.values]),
            index=datetime_series.index,
            name="col2",
        )
        # 断言 frame 中 'col2' 列与预期的 exp Series 相等
        tm.assert_series_equal(exp, frame["col2"])

        # 创建一个 DataFrame 对象，包含三列 'col1', 'col2', 'col3'，使用 datetime_series 和 datetime_series_short
        frame = DataFrame(
            {"col1": datetime_series, "col2": datetime_series_short},
            columns=["col2", "col3", "col4"],
        )

        # 断言 frame 的长度与 datetime_series_short 相等
        assert len(frame) == len(datetime_series_short)
        # 断言 'col1' 不在 frame 的列中
        assert "col1" not in frame
        # 断言 frame 中 'col3' 列全部为空值
        assert isna(frame["col3"]).all()

        # 对于边界情况的断言：空 DataFrame 的长度为 0
        assert len(DataFrame()) == 0

        # 测试混合字典和数组的情况，长度不匹配时是否抛出 ValueError 异常
        # 期望捕获 ValueError 异常并且错误消息匹配指定的信息
        msg = "Mixing dicts with non-Series may lead to ambiguous ordering."
        with pytest.raises(ValueError, match=msg):
            DataFrame({"A": {"a": "a", "b": "b"}, "B": ["a", "b", "c"]})

    def test_constructor_dict_length1(self):
        # 长度为 1 的字典的微优化：创建一个 DataFrame 对象，包含一列 'A'，键值对 {"1": 1, "2": 2}
        frame = DataFrame({"A": {"1": 1, "2": 2}})
        # 断言 frame 的索引与预期的 Index(["1", "2"]) 相等
        tm.assert_index_equal(frame.index, Index(["1", "2"]))

    def test_constructor_dict_with_index(self):
        # 空字典加上索引的情况：创建一个 DataFrame 对象，空字典，指定索引 idx
        idx = Index([0, 1, 2])
        frame = DataFrame({}, index=idx)
        # 断言 frame 的索引是 idx
        assert frame.index is idx

    def test_constructor_dict_with_index_and_columns(self):
        # 空字典，带索引和列的情况：创建一个 DataFrame 对象，空字典，指定索引 idx 和列 idx
        idx = Index([0, 1, 2])
        frame = DataFrame({}, index=idx, columns=idx)
        # 断言 frame 的索引是 idx，列是 idx，且 frame 的 _series 长度为 3
        assert frame.index is idx
        assert frame.columns is idx
        assert len(frame._series) == 3

    def test_constructor_dict_of_empty_lists(self):
        # 包含空列表和 Series 的情况：创建一个 DataFrame 对象，包含 'A', 'B' 列，值为空列表
        frame = DataFrame({"A": [], "B": []}, columns=["A", "B"])
        # 断言 frame 的索引是 RangeIndex(0)
        tm.assert_index_equal(frame.index, RangeIndex(0), exact=True)

    def test_constructor_dict_with_none(self):
        # GH 14381：包含 None 值的字典情况
        # 创建一个 DataFrame 对象，包含一列 'a'，值为 None，指定索引为 [0]
        frame_none = DataFrame({"a": None}, index=[0])
        # 创建一个 DataFrame 对象，包含一列 'a'，值为 [None]，指定索引为 [0]
        frame_none_list = DataFrame({"a": [None]}, index=[0])
        # 断言 frame_none 在索引 0 和列 'a' 的值是 None
        assert frame_none._get_value(0, "a") is None
        # 断言 frame_none_list 在索引 0 和列 'a' 的值是 None
        assert frame_none_list._get_value(0, "a") is None
        # 断言 frame_none 和 frame_none_list 相等
        tm.assert_frame_equal(frame_none, frame_none_list)
    def test_constructor_dict_errors(self):
        # 测试用例：检查DataFrame构造函数对于只包含标量值的字典是否会引发错误
        msg = "If using all scalar values, you must pass an index"
        # 使用pytest检查是否引发值错误，并验证错误消息
        with pytest.raises(ValueError, match=msg):
            DataFrame({"a": 0.7})

        with pytest.raises(ValueError, match=msg):
            DataFrame({"a": 0.7}, columns=["a"])

    @pytest.mark.parametrize("scalar", [2, np.nan, None, "D"])
    def test_constructor_invalid_items_unused(self, scalar):
        # 测试用例：当传入的无效（标量）值未被使用时不应引发错误
        result = DataFrame({"a": scalar}, columns=["b"])
        expected = DataFrame(columns=["b"])
        # 使用tm.assert_frame_equal验证预期结果与实际结果是否一致
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("value", [2, np.nan, None, float("nan")])
    def test_constructor_dict_nan_key(self, value):
        # 测试用例：检查DataFrame构造函数对包含NaN键的字典的处理
        # 创建数据字典，每个列都是Series对象，其中键值可能包含NaN
        cols = [1, value, 3]
        idx = ["a", value]
        values = [[0, 3], [1, 4], [2, 5]]
        data = {cols[c]: Series(values[c], index=idx) for c in range(3)}
        # 构造DataFrame并进行排序以验证结果是否与预期一致
        result = DataFrame(data).sort_values(1).sort_values("a", axis=1)
        expected = DataFrame(
            np.arange(6, dtype="int64").reshape(2, 3), index=idx, columns=cols
        )
        tm.assert_frame_equal(result, expected)

        result = DataFrame(data, index=idx).sort_values("a", axis=1)
        tm.assert_frame_equal(result, expected)

        result = DataFrame(data, index=idx, columns=cols)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("value", [np.nan, None, float("nan")])
    def test_constructor_dict_nan_tuple_key(self, value):
        # 测试用例：检查DataFrame构造函数对包含NaN元组键的字典的处理
        cols = Index([(11, 21), (value, 22), (13, value)])
        idx = Index([("a", value), (value, 2)])
        values = [[0, 3], [1, 4], [2, 5]]
        data = {cols[c]: Series(values[c], index=idx) for c in range(3)}
        # 构造DataFrame并进行排序以验证结果是否与预期一致
        result = DataFrame(data).sort_values((11, 21)).sort_values(("a", value), axis=1)
        expected = DataFrame(
            np.arange(6, dtype="int64").reshape(2, 3), index=idx, columns=cols
        )
        tm.assert_frame_equal(result, expected)

        result = DataFrame(data, index=idx).sort_values(("a", value), axis=1)
        tm.assert_frame_equal(result, expected)

        result = DataFrame(data, index=idx, columns=cols)
        tm.assert_frame_equal(result, expected)

    def test_constructor_dict_order_insertion(self):
        # 测试用例：检查DataFrame构造函数对插入顺序的处理
        datetime_series = Series(
            np.arange(10, dtype=np.float64), index=date_range("2020-01-01", periods=10)
        )
        datetime_series_short = datetime_series[:5]

        # 创建数据字典，并按照插入顺序构造DataFrame
        d = {"b": datetime_series_short, "a": datetime_series}
        frame = DataFrame(data=d)
        expected = DataFrame(data=d, columns=list("ba"))
        # 使用tm.assert_frame_equal验证预期结果与实际结果是否一致
        tm.assert_frame_equal(frame, expected)
    # 测试构造函数处理 NaN 键和列名的情况
    def test_constructor_dict_nan_key_and_columns(self):
        # GH 16894：GitHub 上的 issue 编号
        # 创建一个 DataFrame，包含 NaN 键和整数列 2，数据为 [[1, 2], [2, 3]]
        result = DataFrame({np.nan: [1, 2], 2: [2, 3]}, columns=[np.nan, 2])
        # 期望的 DataFrame，数据为 [[1, 2], [2, 3]]，列名为 [NaN, 2]
        expected = DataFrame([[1, 2], [2, 3]], columns=[np.nan, 2])
        # 使用测试工具比较两个 DataFrame 是否相等
        tm.assert_frame_equal(result, expected)

    # 测试构造函数处理多级索引的情况
    def test_constructor_multi_index(self):
        # GH 4078：GitHub 上的 issue 编号
        # 创建一个 MultiIndex 对象，由元组 (2, 3), (3, 3), (3, 3) 组成
        tuples = [(2, 3), (3, 3), (3, 3)]
        mi = MultiIndex.from_tuples(tuples)
        # 创建一个索引和列都是 mi 的 DataFrame
        df = DataFrame(index=mi, columns=mi)
        # 断言 DataFrame 中所有值是否为 NaN
        assert isna(df).values.ravel().all()

        # 创建一个不同顺序的 MultiIndex 对象，由元组 (3, 3), (2, 3), (3, 3) 组成
        tuples = [(3, 3), (2, 3), (3, 3)]
        mi = MultiIndex.from_tuples(tuples)
        # 创建一个索引和列都是 mi 的 DataFrame
        df = DataFrame(index=mi, columns=mi)
        # 断言 DataFrame 中所有值是否为 NaN
        assert isna(df).values.ravel().all()

    # 测试构造函数处理二维索引的情况
    def test_constructor_2d_index(self):
        # GH 25416：GitHub 上的 issue 编号
        # 创建一个 DataFrame，数据为 [[1]]，列名为 [[1]]，索引为 [1, 2]
        df = DataFrame([[1]], columns=[[1]], index=[1, 2])
        # 期望的 DataFrame，数据为 [1, 1]，索引为 [1, 2]，列为 MultiIndex([[1]])
        expected = DataFrame(
            [1, 1],
            index=Index([1, 2], dtype="int64"),
            columns=MultiIndex(levels=[[1]], codes=[[0]]),
        )
        # 使用测试工具比较两个 DataFrame 是否相等
        tm.assert_frame_equal(df, expected)

        # 创建一个 DataFrame，数据为 [[1]]，列名为 [[1]]，索引为 [[1, 2]]
        df = DataFrame([[1]], columns=[[1]], index=[[1, 2]])
        # 期望的 DataFrame，数据为 [1, 1]，索引为 MultiIndex([[1, 2]]），列为 MultiIndex([[1]])
        expected = DataFrame(
            [1, 1],
            index=MultiIndex(levels=[[1, 2]], codes=[[0, 1]]),
            columns=MultiIndex(levels=[[1]], codes=[[0]]),
        )
        # 使用测试工具比较两个 DataFrame 是否相等
        tm.assert_frame_equal(df, expected)
    def test_constructor_error_msgs(self):
        msg = "Empty data passed with indices specified."
        # 当传入空数组时且指定了索引，抛出 ValueError 异常，匹配错误信息"Empty data passed with indices specified."
        with pytest.raises(ValueError, match=msg):
            DataFrame(np.empty(0), index=[1])

        msg = "Mixing dicts with non-Series may lead to ambiguous ordering."
        # 混合使用字典和数组时，尺寸错误，抛出 ValueError 异常，匹配错误信息"Mixing dicts with non-Series may lead to ambiguous ordering."
        with pytest.raises(ValueError, match=msg):
            DataFrame({"A": {"a": "a", "b": "b"}, "B": ["a", "b", "c"]})

        # 尺寸错误的 ndarray，GH 3105
        msg = r"Shape of passed values is \(4, 3\), indices imply \(3, 3\)"
        with pytest.raises(ValueError, match=msg):
            DataFrame(
                np.arange(12).reshape((4, 3)),
                columns=["foo", "bar", "baz"],
                index=date_range("2000-01-01", periods=3),
            )

        arr = np.array([[4, 5, 6]])
        msg = r"Shape of passed values is \(1, 3\), indices imply \(1, 4\)"
        # 当传入的数据尺寸与索引尺寸不匹配时，抛出 ValueError 异常，匹配错误信息"Shape of passed values is (1, 3), indices imply (1, 4)"
        with pytest.raises(ValueError, match=msg):
            DataFrame(index=[0], columns=range(4), data=arr)

        arr = np.array([4, 5, 6])
        msg = r"Shape of passed values is \(3, 1\), indices imply \(1, 4\)"
        # 当传入的数据尺寸与索引尺寸不匹配时，抛出 ValueError 异常，匹配错误信息"Shape of passed values is (3, 1), indices imply (1, 4)"
        with pytest.raises(ValueError, match=msg):
            DataFrame(index=[0], columns=range(4), data=arr)

        # 高维数据引发异常
        with pytest.raises(ValueError, match="Must pass 2-d input"):
            DataFrame(np.zeros((3, 3, 3)), columns=["A", "B", "C"], index=[1])

        # 尺寸错误的轴标签
        msg = r"Shape of passed values is \(2, 3\), indices imply \(1, 3\)"
        with pytest.raises(ValueError, match=msg):
            DataFrame(
                np.random.default_rng(2).random((2, 3)),
                columns=["A", "B", "C"],
                index=[1],
            )

        msg = r"Shape of passed values is \(2, 3\), indices imply \(2, 2\)"
        # 当传入的数据尺寸与索引尺寸不匹配时，抛出 ValueError 异常，匹配错误信息"Shape of passed values is (2, 3), indices imply (2, 2)"
        with pytest.raises(ValueError, match=msg):
            DataFrame(
                np.random.default_rng(2).random((2, 3)),
                columns=["A", "B"],
                index=[1, 2],
            )

        # gh-26429
        msg = "2 columns passed, passed data had 10 columns"
        # 当传入的数据列数与指定的列数不匹配时，抛出 ValueError 异常，匹配错误信息"2 columns passed, passed data had 10 columns"
        with pytest.raises(ValueError, match=msg):
            DataFrame((range(10), range(10, 20)), columns=("ones", "twos"))

        msg = "If using all scalar values, you must pass an index"
        # 如果使用全部标量值，必须传递索引，否则抛出 ValueError 异常，匹配错误信息"If using all scalar values, you must pass an index"
        with pytest.raises(ValueError, match=msg):
            DataFrame({"a": False, "b": True})
    def test_constructor_subclass_dict(self, dict_subclass):
        # Test for passing dict subclass to constructor
        # 创建一个测试数据字典，包含两个列，每列是一个自定义字典子类的实例，每个实例包含十个键值对
        data = {
            "col1": dict_subclass((x, 10.0 * x) for x in range(10)),
            "col2": dict_subclass((x, 20.0 * x) for x in range(10)),
        }
        # 使用测试数据字典创建一个 DataFrame 对象
        df = DataFrame(data)
        # 根据测试数据字典创建一个参考 DataFrame 对象
        refdf = DataFrame({col: dict(val.items()) for col, val in data.items()})
        # 使用测试工具比较两个 DataFrame 对象是否相等
        tm.assert_frame_equal(refdf, df)

        # 将测试数据字典转换成字典子类的实例，并创建一个新的 DataFrame 对象
        data = dict_subclass(data.items())
        df = DataFrame(data)
        # 使用测试工具比较转换后的 DataFrame 对象是否与参考 DataFrame 对象相等
        tm.assert_frame_equal(refdf, df)

    def test_constructor_defaultdict(self, float_frame):
        # try with defaultdict
        # 创建一个空数据字典
        data = {}
        # 将 float_frame 的前十一行第 B 列设为 NaN
        float_frame.loc[: float_frame.index[10], "B"] = np.nan

        # 遍历 float_frame 的每一列和其对应的 Series 对象
        for k, v in float_frame.items():
            # 创建一个 defaultdict 对象 dct，并用 v 的字典表示更新 dct
            dct = defaultdict(dict)
            dct.update(v.to_dict())
            data[k] = dct
        # 使用数据字典创建一个 DataFrame 对象
        frame = DataFrame(data)
        # 根据 float_frame 的索引重新索引 frame，并将结果赋给 expected
        expected = frame.reindex(index=float_frame.index)
        # 使用测试工具比较 float_frame 和 expected 是否相等
        tm.assert_frame_equal(float_frame, expected)

    def test_constructor_dict_block(self):
        # GH #1490
        # 创建一个预期的 numpy 数组
        expected = np.array([[4.0, 3.0, 2.0, 1.0]])
        # 使用字典数据创建一个 DataFrame 对象，指定列顺序
        df = DataFrame(
            {"d": [4.0], "c": [3.0], "b": [2.0], "a": [1.0]},
            columns=["d", "c", "b", "a"],
        )
        # 使用测试工具比较 DataFrame 对象的值数组是否与预期的 numpy 数组相等
        tm.assert_numpy_array_equal(df.values, expected)

    def test_constructor_dict_cast(self, using_infer_string):
        # cast float tests
        # 创建一个测试数据字典，包含两列，其中一列的值是整数，另一列的值是字符串
        test_data = {"A": {"1": 1, "2": 2}, "B": {"1": "1", "2": "2", "3": "3"}}
        # 使用测试数据字典创建一个 DataFrame 对象，指定数据类型为 float
        frame = DataFrame(test_data, dtype=float)
        # 断言 DataFrame 的长度为 3
        assert len(frame) == 3
        # 断言 DataFrame 的列 'B' 的数据类型为 np.float64
        assert frame["B"].dtype == np.float64
        # 断言 DataFrame 的列 'A' 的数据类型为 np.float64
        assert frame["A"].dtype == np.float64

        # 使用测试数据字典创建一个 DataFrame 对象，自动推断数据类型
        frame = DataFrame(test_data)
        # 断言 DataFrame 的长度为 3
        assert len(frame) == 3
        # 如果不使用推断字符串类型，则断言 DataFrame 的列 'B' 的数据类型为 np.object_，否则为 "string"
        assert frame["B"].dtype == np.object_ if not using_infer_string else "string"
        # 断言 DataFrame 的列 'A' 的数据类型为 np.float64
        assert frame["A"].dtype == np.float64

    def test_constructor_dict_cast2(self):
        # can't cast to float
        # 创建一个测试数据字典，其中一列包含字符串，无法转换为 float
        test_data = {
            "A": dict(zip(range(20), [f"word_{i}" for i in range(20)])),
            "B": dict(zip(range(15), np.random.default_rng(2).standard_normal(15))),
        }
        # 使用测试工具断言创建 DataFrame 对象时会引发 ValueError 异常
        with pytest.raises(ValueError, match="could not convert string"):
            DataFrame(test_data, dtype=float)

    def test_constructor_dict_dont_upcast(self):
        # 创建一个测试数据字典，其中一列包含字符串和 NaN
        d = {"Col1": {"Row1": "A String", "Row2": np.nan}}
        # 使用测试数据字典创建一个 DataFrame 对象
        df = DataFrame(d)
        # 断言 DataFrame 中 'Col1' 列的 'Row2' 元素的类型为 float
        assert isinstance(df["Col1"]["Row2"], float)

    def test_constructor_dict_dont_upcast2(self):
        # 创建一个测试数据矩阵，包含整数和字符串
        dm = DataFrame([[1, 2], ["a", "b"]], index=[1, 2], columns=[1, 2])
        # 断言数据矩阵中位置 [1, 1] 的元素类型为 int
        assert isinstance(dm[1][1], int)

    def test_constructor_dict_of_tuples(self):
        # GH #1491
        # 创建一个测试数据字典，其中每个值是一个元组
        data = {"a": (1, 2, 3), "b": (4, 5, 6)}

        # 使用测试数据字典创建一个 DataFrame 对象
        result = DataFrame(data)
        # 根据测试数据字典创建一个期望的 DataFrame 对象
        expected = DataFrame({k: list(v) for k, v in data.items()})
        # 使用测试工具比较两个 DataFrame 对象是否相等，不检查数据类型
        tm.assert_frame_equal(result, expected, check_dtype=False)
    # 测试用例：使用字典中的范围构造DataFrame对象
    def test_constructor_dict_of_ranges(self):
        # GH 26356
        # 创建包含键值对的字典，其中值是范围对象
        data = {"a": range(3), "b": range(3, 6)}

        # 使用字典数据创建DataFrame对象
        result = DataFrame(data)
        
        # 预期的DataFrame对象，将范围对象转换为列表作为DataFrame的列
        expected = DataFrame({"a": [0, 1, 2], "b": [3, 4, 5]})
        
        # 使用测试工具比较结果和预期DataFrame对象
        tm.assert_frame_equal(result, expected)

    # 测试用例：使用字典中的迭代器构造DataFrame对象
    def test_constructor_dict_of_iterators(self):
        # GH 26349
        # 创建包含键值对的字典，其中值是迭代器对象
        data = {"a": iter(range(3)), "b": reversed(range(3))}

        # 使用字典数据创建DataFrame对象
        result = DataFrame(data)
        
        # 预期的DataFrame对象，将迭代器对象转换为列表作为DataFrame的列
        expected = DataFrame({"a": [0, 1, 2], "b": [2, 1, 0]})
        
        # 使用测试工具比较结果和预期DataFrame对象
        tm.assert_frame_equal(result, expected)

    # 测试用例：使用字典中的生成器构造DataFrame对象
    def test_constructor_dict_of_generators(self):
        # GH 26349
        # 创建包含键值对的字典，其中值是生成器对象
        data = {"a": (i for i in (range(3))), "b": (i for i in reversed(range(3)))}
        
        # 使用字典数据创建DataFrame对象
        result = DataFrame(data)
        
        # 预期的DataFrame对象，将生成器对象转换为列表作为DataFrame的列
        expected = DataFrame({"a": [0, 1, 2], "b": [2, 1, 0]})
        
        # 使用测试工具比较结果和预期DataFrame对象
        tm.assert_frame_equal(result, expected)

    # 测试用例：使用具有多级索引的字典构造DataFrame对象
    def test_constructor_dict_multiindex(self):
        # 创建具有多级键值对的字典
        d = {
            ("a", "a"): {("i", "i"): 0, ("i", "j"): 1, ("j", "i"): 2},
            ("b", "a"): {("i", "i"): 6, ("i", "j"): 5, ("j", "i"): 4},
            ("b", "c"): {("i", "i"): 7, ("i", "j"): 8, ("j", "i"): 9},
        }
        
        # 对字典的键值对进行排序
        _d = sorted(d.items())
        
        # 使用字典数据创建DataFrame对象
        df = DataFrame(d)
        
        # 从排序后的键值对中创建预期的DataFrame对象
        expected = DataFrame(
            [x[1] for x in _d], index=MultiIndex.from_tuples([x[0] for x in _d])
        ).T
        expected.index = MultiIndex.from_tuples(expected.index)
        
        # 使用测试工具比较结果和预期DataFrame对象
        tm.assert_frame_equal(df, expected)

        # 向字典中添加新的键值对
        d["z"] = {"y": 123.0, ("i", "i"): 111, ("i", "j"): 111, ("j", "i"): 111}
        
        # 将新的键值对插入到排序后的键值对列表中
        _d.insert(0, ("z", d["z"]))
        
        # 重新创建预期的DataFrame对象，包括新添加的键值对
        expected = DataFrame(
            [x[1] for x in _d], index=Index([x[0] for x in _d], tupleize_cols=False)
        ).T
        expected.index = Index(expected.index, tupleize_cols=False)
        
        # 使用字典数据创建DataFrame对象
        df = DataFrame(d)
        
        # 使用预期的索引和列重新索引DataFrame对象
        df = df.reindex(columns=expected.columns, index=expected.index)
        
        # 使用测试工具比较结果和预期DataFrame对象
        tm.assert_frame_equal(df, expected)
    # 测试用例，验证构造函数对于具有 datetime64 索引的字典的行为
    def test_constructor_dict_datetime64_index(self):
        # GH 10160，GitHub 上的 issue 编号
        dates_as_str = ["1984-02-19", "1988-11-06", "1989-12-03", "1990-03-15"]

        # 定义一个内部函数用于创建数据字典，根据给定的构造函数 constructor
        def create_data(constructor):
            return {i: {constructor(s): 2 * i} for i, s in enumerate(dates_as_str)}

        # 使用 np.datetime64 构造日期时间数据字典
        data_datetime64 = create_data(np.datetime64)
        # 使用 lambda 表达式和 datetime.strptime 方法构造日期时间数据字典
        data_datetime = create_data(lambda x: datetime.strptime(x, "%Y-%m-%d"))
        # 使用 Timestamp 类构造日期时间数据字典
        data_Timestamp = create_data(Timestamp)

        # 期望的 DataFrame 结果，使用 Timestamp 对象作为索引
        expected = DataFrame(
            [
                {0: 0, 1: None, 2: None, 3: None},
                {0: None, 1: 2, 2: None, 3: None},
                {0: None, 1: None, 2: 4, 3: None},
                {0: None, 1: None, 2: None, 3: 6},
            ],
            index=[Timestamp(dt) for dt in dates_as_str],
        )

        # 分别创建三个 DataFrame 对象
        result_datetime64 = DataFrame(data_datetime64)
        result_datetime = DataFrame(data_datetime)
        result_Timestamp = DataFrame(data_Timestamp)

        # 使用测试工具库中的方法比较 DataFrame 对象是否相等
        tm.assert_frame_equal(result_datetime64, expected)
        tm.assert_frame_equal(result_datetime, expected)
        tm.assert_frame_equal(result_Timestamp, expected)

    # 使用 pytest 的参数化装饰器进行多参数测试
    @pytest.mark.parametrize(
        "klass,name",
        [
            (lambda x: np.timedelta64(x, "D"), "timedelta64"),
            (lambda x: timedelta(days=x), "pytimedelta"),
            (lambda x: Timedelta(x, "D"), "Timedelta[ns]"),
            (lambda x: Timedelta(x, "D").as_unit("s"), "Timedelta[s]"),
        ],
    )
    # 测试构造函数对于具有 timedelta64 索引的字典的行为
    def test_constructor_dict_timedelta64_index(self, klass, name):
        # GH 10160，GitHub 上的 issue 编号
        td_as_int = [1, 2, 3, 4]

        # 创建数据字典，使用给定的 klass 构造函数
        data = {i: {klass(s): 2 * i} for i, s in enumerate(td_as_int)}

        # 期望的 DataFrame 结果，使用 Timedelta 对象作为索引
        expected = DataFrame(
            [
                {0: 0, 1: None, 2: None, 3: None},
                {0: None, 1: 2, 2: None, 3: None},
                {0: None, 1: None, 2: 4, 3: None},
                {0: None, 1: None, 2: None, 3: 6},
            ],
            index=[Timedelta(td, "D") for td in td_as_int],
        )

        # 创建测试结果的 DataFrame 对象
        result = DataFrame(data)

        # 使用测试工具库中的方法比较 DataFrame 对象是否相等
        tm.assert_frame_equal(result, expected)

    # 测试构造 PeriodIndex 的行为
    def test_constructor_period_dict(self):
        # 创建一个 PeriodIndex，频率为月份（M）
        a = pd.PeriodIndex(["2012-01", "NaT", "2012-04"], freq="M")
        # 创建另一个 PeriodIndex，频率为天（D）
        b = pd.PeriodIndex(["2012-02-01", "2012-03-01", "NaT"], freq="D")
        
        # 创建一个 DataFrame，包含两个 PeriodIndex 对象
        df = DataFrame({"a": a, "b": b})
        # 断言列 'a' 和 'b' 的数据类型与期望相同
        assert df["a"].dtype == a.dtype
        assert df["b"].dtype == b.dtype

        # 将 PeriodIndex 转换为对象列表，并创建 DataFrame
        df = DataFrame({"a": a.astype(object).tolist(), "b": b.astype(object).tolist()})
        # 断言列 'a' 和 'b' 的数据类型与期望相同
        assert df["a"].dtype == a.dtype
        assert df["b"].dtype == b.dtype
    # 测试用例：构造函数测试，使用标量作为扩展数据
    def test_constructor_dict_extension_scalar(self, ea_scalar_and_dtype):
        # 解包测试参数
        ea_scalar, ea_dtype = ea_scalar_and_dtype
        # 创建包含单个列"a"的DataFrame，数据为标量ea_scalar，索引为[0]
        df = DataFrame({"a": ea_scalar}, index=[0])
        # 断言列"a"的数据类型等于期望的ea_dtype
        assert df["a"].dtype == ea_dtype

        # 期望的DataFrame，只包含单个列"a"，数据与df相同
        expected = DataFrame(index=[0], columns=["a"], data=ea_scalar)

        # 使用测试工具函数assert_frame_equal比较df和expected，确认它们相等
        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize(
        "data,dtype",
        [
            (Period("2020-01"), PeriodDtype("M")),  # 参数化测试数据为Period类型
            (Interval(left=0, right=5), IntervalDtype("int64", "right")),  # 参数化测试数据为Interval类型
            (
                Timestamp("2011-01-01", tz="US/Eastern"),  # 参数化测试数据为带时区的Timestamp类型
                DatetimeTZDtype(unit="s", tz="US/Eastern"),  # 期望的数据类型为带时区的DatetimeTZDtype
            ),
        ],
    )
    # 测试用例：构造函数测试，使用扩展标量作为数据
    def test_constructor_extension_scalar_data(self, data, dtype):
        # GH 34832

        # 创建DataFrame，包含两行，列名为"a"和"b"，数据为传入的data
        df = DataFrame(index=[0, 1], columns=["a", "b"], data=data)

        # 断言列"a"和"b"的数据类型等于期望的dtype
        assert df["a"].dtype == dtype
        assert df["b"].dtype == dtype

        # 使用pd.array创建数组arr，包含两行，数据类型为dtype
        arr = pd.array([data] * 2, dtype=dtype)
        # 期望的DataFrame，包含列"a"和"b"，数据为arr
        expected = DataFrame({"a": arr, "b": arr})

        # 使用测试工具函数assert_frame_equal比较df和expected，确认它们相等
        tm.assert_frame_equal(df, expected)

    # 测试用例：嵌套字典构造DataFrame
    def test_nested_dict_frame_constructor(self):
        # 创建时间范围rng，从"1/1/2000"开始，连续5个周期
        rng = pd.period_range("1/1/2000", periods=5)
        # 创建10行5列的DataFrame，数据为标准正态分布随机数
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 5)), columns=rng)

        # 初始化空字典data
        data = {}
        # 遍历df的每一列
        for col in df.columns:
            # 遍历df的每一行
            for row in df.index:
                # 将df中(row, col)位置的值存入data[col][row]
                data.setdefault(col, {})[row] = df._get_value(row, col)

        # 根据data创建新的DataFrame，列为rng
        result = DataFrame(data, columns=rng)
        # 使用测试工具函数assert_frame_equal比较result和df，确认它们相等
        tm.assert_frame_equal(result, df)

        # 初始化空字典data
        data = {}
        # 遍历df的每一列
        for col in df.columns:
            # 遍历df的每一行
            for row in df.index:
                # 将df中(row, col)位置的值存入data[row][col]
                data.setdefault(row, {})[col] = df._get_value(row, col)

        # 根据data创建新的DataFrame，索引为rng的转置
        result = DataFrame(data, index=rng).T
        # 使用测试工具函数assert_frame_equal比较result和df，确认它们相等
        tm.assert_frame_equal(result, df)
    def _check_basic_constructor(self, empty):
        # 使用 empty 函数创建一个形状为 (2, 3)，数据类型为 float 的二维矩阵 mat
        mat = empty((2, 3), dtype=float)
        # 使用 DataFrame 类构造一个二维数据帧 frame，传入 mat 作为数据，列名为 ["A", "B", "C"]，行索引为 [1, 2]
        frame = DataFrame(mat, columns=["A", "B", "C"], index=[1, 2])

        # 断言数据帧 frame 的行索引长度为 2
        assert len(frame.index) == 2
        # 断言数据帧 frame 的列名长度为 3
        assert len(frame.columns) == 3

        # 使用 empty 函数创建一个形状为 (3,) 的一维数组 mat
        frame = DataFrame(empty((3,)), columns=["A"], index=[1, 2, 3])
        # 断言数据帧 frame 的行索引长度为 3
        assert len(frame.index) == 3
        # 断言数据帧 frame 的列名长度为 1
        assert len(frame.columns) == 1

        # 如果 empty 不是 np.ones 函数，则抛出 IntCastingNaNError 异常，匹配异常信息 msg
        if empty is not np.ones:
            msg = r"Cannot convert non-finite values \(NA or inf\) to integer"
            with pytest.raises(IntCastingNaNError, match=msg):
                # 使用 DataFrame 类构造一个数据帧 frame，传入 mat 作为数据，列名为 ["A", "B", "C"]，行索引为 [1, 2]，数据类型为 np.int64
                DataFrame(mat, columns=["A", "B", "C"], index=[1, 2], dtype=np.int64)
            return
        else:
            # 使用 DataFrame 类构造一个数据帧 frame，传入 mat 作为数据，列名为 ["A", "B", "C"]，行索引为 [1, 2]，数据类型为 np.int64
            frame = DataFrame(
                mat, columns=["A", "B", "C"], index=[1, 2], dtype=np.int64
            )
            # 断言数据帧 frame 的值的数据类型为 np.int64
            assert frame.values.dtype == np.int64

        # 错误的行索引尺寸标签
        msg = r"Shape of passed values is \(2, 3\), indices imply \(1, 3\)"
        with pytest.raises(ValueError, match=msg):
            # 使用 DataFrame 类构造一个数据帧 frame，传入 mat 作为数据，列名为 ["A", "B", "C"]，行索引为 [1]
            DataFrame(mat, columns=["A", "B", "C"], index=[1])
        msg = r"Shape of passed values is \(2, 3\), indices imply \(2, 2\)"
        with pytest.raises(ValueError, match=msg):
            # 使用 DataFrame 类构造一个数据帧 frame，传入 mat 作为数据，列名为 ["A", "B"]，行索引为 [1, 2]
            DataFrame(mat, columns=["A", "B"], index=[1, 2])

        # 高维度数组引发异常
        with pytest.raises(ValueError, match="Must pass 2-d input"):
            # 使用 DataFrame 类构造一个数据帧 frame，传入形状为 (3, 3, 3) 的数组作为数据，列名为 ["A", "B", "C"]，行索引为 [1]
            DataFrame(empty((3, 3, 3)), columns=["A", "B", "C"], index=[1])

        # 自动标签
        # 使用 DataFrame 类构造一个数据帧 frame，传入 mat 作为数据
        frame = DataFrame(mat)
        # 断言数据帧 frame 的行索引与 Index(range(2)) 完全相等
        tm.assert_index_equal(frame.index, Index(range(2)), exact=True)
        # 断言数据帧 frame 的列名与 Index(range(3)) 完全相等
        tm.assert_index_equal(frame.columns, Index(range(3)), exact=True)

        # 使用 DataFrame 类构造一个数据帧 frame，传入 mat 作为数据，行索引为 [1, 2]
        frame = DataFrame(mat, index=[1, 2])
        # 断言数据帧 frame 的列名与 Index(range(3)) 完全相等
        tm.assert_index_equal(frame.columns, Index(range(3)), exact=True)

        # 使用 DataFrame 类构造一个数据帧 frame，传入 mat 作为数据，列名为 ["A", "B", "C"]
        frame = DataFrame(mat, columns=["A", "B", "C"])
        # 断言数据帧 frame 的行索引与 Index(range(2)) 完全相等
        tm.assert_index_equal(frame.index, Index(range(2)), exact=True)

        # 长度为 0 的轴
        # 使用 DataFrame 类构造一个数据帧 frame，传入形状为 (0, 3) 的空数组作为数据
        frame = DataFrame(empty((0, 3)))
        # 断言数据帧 frame 的行索引长度为 0
        assert len(frame.index) == 0

        # 使用 DataFrame 类构造一个数据帧 frame，传入形状为 (3, 0) 的空数组作为数据
        frame = DataFrame(empty((3, 0)))
        # 断言数据帧 frame 的列名长度为 0
        assert len(frame.columns) == 0
    @pytest.mark.filterwarnings(
        "ignore:elementwise comparison failed:DeprecationWarning"
    )
    # 定义测试函数：测试在处理非浮点数时的构造函数行为
    def test_constructor_maskedarray_nonfloat(self):
        # 创建一个所有元素为 masked 值的 masked array，指定数据类型为整数
        mat = ma.masked_all((2, 3), dtype=int)
        # 创建一个 DataFrame 对象，使用上述 masked array 作为数据，指定列名和索引
        frame = DataFrame(mat, columns=["A", "B", "C"], index=[1, 2])

        # 断言 DataFrame 的行数为 2
        assert len(frame.index) == 2
        # 断言 DataFrame 的列数为 3
        assert len(frame.columns) == 3
        # 断言 DataFrame 中所有元素与自身比较的结果都为 False
        assert np.all(~np.asarray(frame == frame))

        # 将 masked array 强制转换为指定的数据类型 np.float64
        frame = DataFrame(mat, columns=["A", "B", "C"], index=[1, 2], dtype=np.float64)
        # 断言 DataFrame 的值的数据类型为 np.float64
        assert frame.values.dtype == np.float64

        # 检查非 masked 的数值
        mat2 = ma.copy(mat)
        mat2[0, 0] = 1
        mat2[1, 2] = 2
        # 使用修改后的 masked array 创建 DataFrame 对象
        frame = DataFrame(mat2, columns=["A", "B", "C"], index=[1, 2])
        # 断言 DataFrame 中指定位置的值符合预期
        assert 1 == frame["A"][1]
        assert 2 == frame["C"][2]

        # 创建一个所有元素为 masked 值的 masked array，指定数据类型为 np.datetime64[ns]
        mat = ma.masked_all((2, 3), dtype="M8[ns]")
        # 创建一个 DataFrame 对象，使用上述 masked array 作为数据，指定列名和索引
        frame = DataFrame(mat, columns=["A", "B", "C"], index=[1, 2])

        # 断言 DataFrame 的行数为 2
        assert len(frame.index) == 2
        # 断言 DataFrame 的列数为 3
        assert len(frame.columns) == 3
        # 断言 DataFrame 中所有元素都为 NaN
        assert isna(frame).values.all()

        # 尝试将 masked array 强制转换为指定的数据类型 np.int64，预期引发 TypeError
        msg = r"datetime64\[ns\] values and dtype=int64 is not supported"
        with pytest.raises(TypeError, match=msg):
            DataFrame(mat, columns=["A", "B", "C"], index=[1, 2], dtype=np.int64)

        # 检查非 masked 的数值
        mat2 = ma.copy(mat)
        mat2[0, 0] = 1
        mat2[1, 2] = 2
        # 使用修改后的 masked array 创建 DataFrame 对象，并将特定列转换为整数后进行断言
        frame = DataFrame(mat2, columns=["A", "B", "C"], index=[1, 2])
        assert 1 == frame["A"].astype("i8")[1]
        assert 2 == frame["C"].astype("i8")[2]

        # 创建一个所有元素为 masked 值的 masked array，指定数据类型为布尔型
        mat = ma.masked_all((2, 3), dtype=bool)
        # 创建一个 DataFrame 对象，使用上述 masked array 作为数据，指定列名和索引
        frame = DataFrame(mat, columns=["A", "B", "C"], index=[1, 2])

        # 断言 DataFrame 的行数为 2
        assert len(frame.index) == 2
        # 断言 DataFrame 的列数为 3
        assert len(frame.columns) == 3
        # 断言 DataFrame 中所有元素与自身比较的结果都为 False
        assert np.all(~np.asarray(frame == frame))

        # 将 masked array 强制转换为指定的数据类型 object
        frame = DataFrame(mat, columns=["A", "B", "C"], index=[1, 2], dtype=object)
        # 断言 DataFrame 的值的数据类型为 object
        assert frame.values.dtype == object

        # 检查非 masked 的数值
        mat2 = ma.copy(mat)
        mat2[0, 0] = True
        mat2[1, 2] = False
        # 使用修改后的 masked array 创建 DataFrame 对象，并进行布尔值的断言
        frame = DataFrame(mat2, columns=["A", "B", "C"], index=[1, 2])
        assert frame["A"][1] is True
        assert frame["C"][2] is False
    # 定义测试函数，用于测试构造带有硬屏蔽掩码的 numpy 掩码数组
    def test_constructor_maskedarray_hardened(self):
        # 检查使用硬屏蔽的 numpy 掩码数组，来源于 GH24574
        mat_hard = ma.masked_all((2, 2), dtype=float).harden_mask()
        # 创建 DataFrame 对象，使用 mat_hard 数据，列名为 ["A", "B"]，索引为 [1, 2]
        result = DataFrame(mat_hard, columns=["A", "B"], index=[1, 2])
        # 期望的 DataFrame 对象，包含列 "A" 和 "B"，索引为 [1, 2]，数据类型为 float
        expected = DataFrame(
            {"A": [np.nan, np.nan], "B": [np.nan, np.nan]},
            columns=["A", "B"],
            index=[1, 2],
            dtype=float,
        )
        # 断言 result 和 expected 的 DataFrame 是否相等
        tm.assert_frame_equal(result, expected)
        
        # 检查硬屏蔽但未屏蔽任何数据的情况
        mat_hard = ma.ones((2, 2), dtype=float).harden_mask()
        # 创建 DataFrame 对象，使用 mat_hard 数据，列名为 ["A", "B"]，索引为 [1, 2]
        result = DataFrame(mat_hard, columns=["A", "B"], index=[1, 2])
        # 期望的 DataFrame 对象，包含列 "A" 和 "B"，索引为 [1, 2]，数据类型为 float
        expected = DataFrame(
            {"A": [1.0, 1.0], "B": [1.0, 1.0]},
            columns=["A", "B"],
            index=[1, 2],
            dtype=float,
        )
        # 断言 result 和 expected 的 DataFrame 是否相等
        tm.assert_frame_equal(result, expected)

    # 测试构造函数，验证对 maskedrecarray 的 dtype 是否遵守
    def test_constructor_maskedrecarray_dtype(self):
        # 创建一个带有结构化 dtype 的 numpy 掩码数组，元素都是零，并指定所有元素未屏蔽
        data = np.ma.array(
            np.ma.zeros(5, dtype=[("date", "<f8"), ("price", "<f8")]), mask=[False] * 5
        )
        # 将 data 视图转换为 MaskedRecords 类型
        data = data.view(mrecords.mrecarray)
        # 使用 pytest 断言，检查构造 DataFrame 时是否会抛出 TypeError 异常，异常消息应包含特定文本
        with pytest.raises(TypeError, match=r"Pass \{name: data\[name\]"):
            # 支持 MaskedRecords 已弃用 GH#40363
            DataFrame(data, dtype=int)

    # 测试构造函数的边界情况，验证 DataFrame 对象的形状是否为 (0, 0)
    def test_constructor_corner_shape(self):
        df = DataFrame(index=[])
        assert df.values.shape == (0, 0)

    # 使用 pytest 的参数化标记，测试构造函数的 dtype 参数
    @pytest.mark.parametrize(
        "data, index, columns, dtype, expected",
        [
            # 参数化测试用例：data 为 None，index 为 [0, 1, ..., 9]，columns 为 ["a", "b"]，dtype 为 object，期望的 dtype 为 np.object_
            (None, list(range(10)), ["a", "b"], object, np.object_),
            # 参数化测试用例：data 为 None，index 为 None，columns 为 ["a", "b"]，dtype 为 "int64"，期望的 dtype 为 np.dtype("int64")
            (None, None, ["a", "b"], "int64", np.dtype("int64")),
            # 参数化测试用例：data 为 None，index 为 [0, 1, ..., 9]，columns 为 ["a", "b"]，dtype 为 int，期望的 dtype 为 np.dtype("float64")
            (None, list(range(10)), ["a", "b"], int, np.dtype("float64")),
            # 参数化测试用例：data 为 {}，index 为 None，columns 为 ["foo", "bar"]，dtype 为 None，期望的 dtype 为 np.object_
            ({}, None, ["foo", "bar"], None, np.object_),
            # 参数化测试用例：data 为 {"b": 1}，index 为 [0, 1, ..., 9]，columns 为 ["a", "b"]，dtype 为 int，期望的 dtype 为 np.dtype("float64")
            ({"b": 1}, list(range(10)), list("abc"), int, np.dtype("float64")),
        ],
    )
    # 参数化测试函数，验证 DataFrame 构造函数的 dtype 参数
    def test_constructor_dtype(self, data, index, columns, dtype, expected):
        # 调用 DataFrame 构造函数，使用给定的 data, index, columns, dtype 参数
        df = DataFrame(data, index, columns, dtype)
        # 使用断言验证 DataFrame 的值的 dtype 是否等于预期的 expected
        assert df.values.dtype == expected

    # 使用 pytest 的参数化标记，测试构造函数对可空扩展数组的 dtype 参数
    @pytest.mark.parametrize(
        "data,input_dtype,expected_dtype",
        (
            # 参数化测试用例：data 包含 [True, False, None]，input_dtype 为 "boolean"，期望的 dtype 为 pd.BooleanDtype
            ([True, False, None], "boolean", pd.BooleanDtype),
            # 参数化测试用例：data 包含 [1.0, 2.0, None]，input_dtype 为 "Float64"，期望的 dtype 为 pd.Float64Dtype
            ([1.0, 2.0, None], "Float64", pd.Float64Dtype),
            # 参数化测试用例：data 包含 [1, 2, None]，input_dtype 为 "Int64"，期望的 dtype 为 pd.Int64Dtype
            ([1, 2, None], "Int64", pd.Int64Dtype),
            # 参数化测试用例：data 包含 ["a", "b", "c"]，input_dtype 为 "string"，期望的 dtype 为 pd.StringDtype
            (["a", "b", "c"], "string", pd.StringDtype),
        ),
    )
    # 参数化测试函数，验证 DataFrame 构造函数对可空扩展数组的 dtype 参数
    def test_constructor_dtype_nullable_extension_arrays(
        self, data, input_dtype, expected_dtype
    ):
        # 调用 DataFrame 构造函数，使用 data 构造一个 DataFrame 对象，列名为 "a"，dtype 为 input_dtype
        df = DataFrame({"a": data}, dtype=input_dtype)
        # 使用断言验证 DataFrame 列 "a" 的 dtype 是否等于预期的 expected_dtype
        assert df["a"].dtype == expected_dtype()
    # 测试DataFrame的构造函数，使用标量类型推断数据类型
    def test_constructor_scalar_inference(self, using_infer_string):
        # 创建包含不同类型标量数据的字典
        data = {"int": 1, "bool": True, "float": 3.0, "complex": 4j, "object": "foo"}
        # 使用DataFrame构造函数，传入数据和索引
        df = DataFrame(data, index=np.arange(10))

        # 断言不同列的数据类型是否符合预期
        assert df["int"].dtype == np.int64
        assert df["bool"].dtype == np.bool_
        assert df["float"].dtype == np.float64
        assert df["complex"].dtype == np.complex128
        # 根据是否使用推断字符串，检查"object"列的数据类型
        assert df["object"].dtype == np.object_ if not using_infer_string else "string"

    # 测试DataFrame的构造函数，使用数组和标量的组合
    def test_constructor_arrays_and_scalars(self):
        # 创建包含随机数组和布尔值的字典
        df = DataFrame({"a": np.random.default_rng(2).standard_normal(10), "b": True})
        # 创建期望的DataFrame，确保数据正确初始化
        exp = DataFrame({"a": df["a"].values, "b": [True] * 10})

        # 使用工具函数检查实际的DataFrame和期望的DataFrame是否相等
        tm.assert_frame_equal(df, exp)

        # 使用断言检查构造DataFrame时缺少索引是否会引发异常
        with pytest.raises(ValueError, match="must pass an index"):
            DataFrame({"a": False, "b": True})

    # 测试DataFrame的构造函数，使用另一个DataFrame作为数据源
    def test_constructor_DataFrame(self, float_frame):
        # 使用另一个DataFrame作为数据源创建新的DataFrame
        df = DataFrame(float_frame)
        # 使用工具函数检查两个DataFrame是否相等
        tm.assert_frame_equal(df, float_frame)

        # 使用指定的数据类型创建新的DataFrame，并检查其数据类型是否符合预期
        df_casted = DataFrame(float_frame, dtype=np.int64)
        assert df_casted.values.dtype == np.int64

    # 测试DataFrame的构造函数，使用空的DataFrame创建另一个DataFrame
    def test_constructor_empty_dataframe(self):
        # GH 20624
        # 使用空的DataFrame作为数据源创建新的DataFrame，并指定数据类型为对象
        actual = DataFrame(DataFrame(), dtype="object")
        expected = DataFrame([], dtype="object")
        # 使用工具函数检查两个DataFrame是否相等
        tm.assert_frame_equal(actual, expected)

    # 测试DataFrame的构造函数，包含更多复杂的构造方式
    def test_constructor_more(self, float_frame):
        # 创建包含随机数组的DataFrame，指定列名和索引
        arr = np.random.default_rng(2).standard_normal(10)
        dm = DataFrame(arr, columns=["A"], index=np.arange(10))
        # 使用断言检查DataFrame的数据维度是否为2
        assert dm.values.ndim == 2

        # 创建包含零个元素的随机数组的DataFrame
        arr = np.random.default_rng(2).standard_normal(0)
        dm = DataFrame(arr)
        assert dm.values.ndim == 2
        assert dm.values.ndim == 2

        # 创建没有具体数据但指定列名和索引的DataFrame
        dm = DataFrame(columns=["A", "B"], index=np.arange(10))
        assert dm.values.shape == (10, 2)

        # 创建没有具体数据但指定列名的DataFrame
        dm = DataFrame(columns=["A", "B"])
        assert dm.values.shape == (0, 2)

        # 创建没有具体数据但指定索引的DataFrame
        dm = DataFrame(index=np.arange(10))
        assert dm.values.shape == (10, 0)

        # 尝试将字符串数组转换为浮点数数组时应该引发异常
        mat = np.array(["foo", "bar"], dtype=object).reshape(2, 1)
        msg = "could not convert string to float: 'foo'"
        with pytest.raises(ValueError, match=msg):
            DataFrame(mat, index=[0, 1], columns=[0], dtype=float)

        # 使用另一个DataFrame的Series列表创建新的DataFrame
        dm = DataFrame(DataFrame(float_frame._series))
        tm.assert_frame_equal(dm, float_frame)

        # 创建包含不同数据类型的列的DataFrame，并检查其列数和数据类型
        dm = DataFrame(
            {"A": np.ones(10, dtype=int), "B": np.ones(10, dtype=np.float64)},
            index=np.arange(10),
        )
        assert len(dm.columns) == 2
        assert dm.values.dtype == np.float64
    # 测试空列表作为数据创建 DataFrame，验证预期结果为空的 DataFrame
    def test_constructor_empty_list(self):
        # 创建空 DataFrame，指定空索引
        df = DataFrame([], index=[])
        # 预期的结果是一个空的 DataFrame，同样指定空索引
        expected = DataFrame(index=[])
        # 使用测试工具比较两个 DataFrame 是否相等
        tm.assert_frame_equal(df, expected)

        # GH 9939
        # 创建一个空的 DataFrame，但指定列名为 ["A", "B"]
        df = DataFrame([], columns=["A", "B"])
        # 预期的结果是一个空的 DataFrame，但列名为 ["A", "B"]
        expected = DataFrame({}, columns=["A", "B"])
        # 使用测试工具比较两个 DataFrame 是否相等
        tm.assert_frame_equal(df, expected)

        # 空的生成器：list(empty_gen()) == []
        def empty_gen():
            yield from ()

        # 使用空生成器创建 DataFrame，指定列名为 ["A", "B"]
        df = DataFrame(empty_gen(), columns=["A", "B"])
        # 使用测试工具比较创建的 DataFrame 是否与预期结果相等
        tm.assert_frame_equal(df, expected)

    # 测试使用嵌套的列表作为数据创建 DataFrame
    def test_constructor_list_of_lists(self, using_infer_string):
        # GH #484
        # 使用包含列表的列表创建 DataFrame，指定列名为 ["num", "str"]
        df = DataFrame(data=[[1, "a"], [2, "b"]], columns=["num", "str"])
        # 验证 DataFrame 中 "num" 列的数据类型是否为整数
        assert is_integer_dtype(df["num"])
        # 验证 DataFrame 中 "str" 列的数据类型是否为对象类型（np.object）或字符串类型（根据 using_infer_string 参数）
        assert df["str"].dtype == np.object_ if not using_infer_string else "string"

        # GH 4851
        # 创建一个预期的 DataFrame，其中包含一个从 0 到 9 的一维 ndarray
        expected = DataFrame({0: np.arange(10)})
        # 创建包含 0 维 ndarray 的列表
        data = [np.array(x) for x in range(10)]
        # 使用这些数据创建 DataFrame
        result = DataFrame(data)
        # 使用测试工具比较创建的 DataFrame 是否与预期结果相等
        tm.assert_frame_equal(result, expected)

    # 测试嵌套 PandasArray 与嵌套 ndarray 匹配的情况
    def test_nested_pandasarray_matches_nested_ndarray(self):
        # GH#43986
        # 创建一个 Series
        ser = Series([1, 2])

        # 创建一个包含对象类型的 ndarray
        arr = np.array([None, None], dtype=object)
        arr[0] = ser
        arr[1] = ser * 2

        # 使用这个 ndarray 创建 DataFrame
        df = DataFrame(arr)
        # 创建一个预期的 DataFrame，使用 pd.array 包装传入的数组
        expected = DataFrame(pd.array(arr))
        # 使用测试工具比较创建的 DataFrame 是否与预期结果相等
        tm.assert_frame_equal(df, expected)
        # 验证 DataFrame 的形状是否为 (2, 1)
        assert df.shape == (2, 1)
        # 使用测试工具比较 DataFrame 中第一列的值是否与传入的数组 arr 相等
        tm.assert_numpy_array_equal(df[0].values, arr)

    # 测试使用列表形式数据创建 DataFrame，其中包含嵌套列表作为列名
    def test_constructor_list_like_data_nested_list_column(self):
        # GH 32173
        # 创建一个包含嵌套列表的列表
        arrays = [list("abcd"), list("cdef")]
        # 使用这些数据创建 DataFrame
        result = DataFrame([[1, 2, 3, 4], [4, 5, 6, 7]], columns=arrays)

        # 使用 MultiIndex.from_arrays 创建预期的 DataFrame，用嵌套列表作为列名
        mi = MultiIndex.from_arrays(arrays)
        expected = DataFrame([[1, 2, 3, 4], [4, 5, 6, 7]], columns=mi)

        # 使用测试工具比较创建的 DataFrame 是否与预期结果相等
        tm.assert_frame_equal(result, expected)

    # 测试使用不正确长度的嵌套列表作为列名创建 DataFrame
    def test_constructor_wrong_length_nested_list_column(self):
        # GH 32173
        # 创建一个包含不正确长度嵌套列表的列表
        arrays = [list("abc"), list("cde")]

        # 预期会引发 ValueError 异常，因为传入的列数与数据不匹配
        msg = "3 columns passed, passed data had 4"
        with pytest.raises(ValueError, match=msg):
            # 使用这些数据创建 DataFrame
            DataFrame([[1, 2, 3, 4], [4, 5, 6, 7]], columns=arrays)

    # 测试使用长度不同的嵌套列表作为列名创建 DataFrame
    def test_constructor_unequal_length_nested_list_column(self):
        # GH 32173
        # 创建一个包含长度不同的嵌套列表的列表
        arrays = [list("abcd"), list("cde")]

        # 预期会引发 ValueError 异常，因为所有的数组必须具有相同的长度
        msg = "all arrays must be same length"
        with pytest.raises(ValueError, match=msg):
            # 使用这些数据创建 DataFrame
            DataFrame([[1, 2, 3, 4], [4, 5, 6, 7]], columns=arrays)

    # 使用参数化测试数据进行单元测试，测试使用包含单个元素的数据列表创建 DataFrame
    @pytest.mark.parametrize(
        "data",
        [
            [[Timestamp("2021-01-01")]],
            [{"x": Timestamp("2021-01-01")}],
            {"x": [Timestamp("2021-01-01")]},
            {"x": Timestamp("2021-01-01")},
        ],
    )
    def test_constructor_one_element_data_list(self, data):
        # GH#42810
        # 使用指定的数据和索引创建 DataFrame，列名为 ["x"]
        result = DataFrame(data, index=[0, 1, 2], columns=["x"])
        # 创建一个预期的 DataFrame，其中 "x" 列包含多个相同的时间戳
        expected = DataFrame({"x": [Timestamp("2021-01-01")] * 3})
        # 使用测试工具比较创建的 DataFrame 是否与预期结果相等
        tm.assert_frame_equal(result, expected)
    def test_constructor_sequence_like(self):
        # GH 3783
        # collections.Sequence like
        
        # 定义一个继承自 abc.Sequence 的虚拟容器类 DummyContainer
        class DummyContainer(abc.Sequence):
            def __init__(self, lst) -> None:
                self._lst = lst

            # 实现 __getitem__ 方法，使得该容器能够像序列一样获取元素
            def __getitem__(self, n):
                return self._lst.__getitem__(n)

            # 实现 __len__ 方法，返回容器的长度
            def __len__(self) -> int:
                return self._lst.__len__()

        # 创建包含 DummyContainer 实例的列表 lst_containers
        lst_containers = [DummyContainer([1, "a"]), DummyContainer([2, "b"])]
        # 定义 DataFrame 的列名
        columns = ["num", "str"]
        # 使用 lst_containers 和 columns 创建 DataFrame 对象 result
        result = DataFrame(lst_containers, columns=columns)
        # 创建期望的 DataFrame 对象 expected
        expected = DataFrame([[1, "a"], [2, "b"]], columns=columns)
        # 使用 tm.assert_frame_equal 进行结果比较，忽略数据类型的检查
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_constructor_stdlib_array(self):
        # GH 4297
        # support Array
        
        # 创建一个包含 array.array 实例的 DataFrame 对象 result
        result = DataFrame({"A": array.array("i", range(10))})
        # 创建一个期望的 DataFrame 对象 expected，使用列表来初始化
        expected = DataFrame({"A": list(range(10))})
        # 使用 tm.assert_frame_equal 进行结果比较，忽略数据类型的检查
        tm.assert_frame_equal(result, expected, check_dtype=False)

        # 创建一个包含多个 array.array 实例的 DataFrame 对象 result
        result = DataFrame([array.array("i", range(10)), array.array("i", range(10))])
        # 创建一个期望的 DataFrame 对象 expected，使用列表来初始化
        expected = DataFrame([list(range(10)), list(range(10))])
        # 使用 tm.assert_frame_equal 进行结果比较，忽略数据类型的检查
        tm.assert_frame_equal(result, expected, check_dtype=False)

    def test_constructor_range(self):
        # GH26342
        
        # 使用 range(10) 初始化 DataFrame 对象 result
        result = DataFrame(range(10))
        # 使用列表来初始化期望的 DataFrame 对象 expected
        expected = DataFrame(list(range(10)))
        # 使用 tm.assert_frame_equal 进行结果比较
        tm.assert_frame_equal(result, expected)

    def test_constructor_list_of_ranges(self):
        
        # 创建一个包含多个 range(10) 实例的 DataFrame 对象 result
        result = DataFrame([range(10), range(10)])
        # 创建一个期望的 DataFrame 对象 expected，使用列表来初始化
        expected = DataFrame([list(range(10)), list(range(10))])
        # 使用 tm.assert_frame_equal 进行结果比较
        tm.assert_frame_equal(result, expected)

    def test_constructor_iterable(self):
        # GH 21987
        
        # 定义一个迭代器类 Iter
        class Iter:
            def __iter__(self) -> Iterator:
                for i in range(10):
                    yield [1, 2, 3]

        # 创建一个期望的 DataFrame 对象 expected，包含多个相同的 [1, 2, 3] 列表
        expected = DataFrame([[1, 2, 3]] * 10)
        # 使用 Iter 实例创建 DataFrame 对象 result
        result = DataFrame(Iter())
        # 使用 tm.assert_frame_equal 进行结果比较
        tm.assert_frame_equal(result, expected)

    def test_constructor_iterator(self):
        
        # 使用 iter(range(10)) 创建 DataFrame 对象 result
        result = DataFrame(iter(range(10)))
        # 使用列表来初始化期望的 DataFrame 对象 expected
        expected = DataFrame(list(range(10)))
        # 使用 tm.assert_frame_equal 进行结果比较
        tm.assert_frame_equal(result, expected)

    def test_constructor_list_of_iterators(self):
        
        # 创建一个包含多个 iter(range(10)) 实例的 DataFrame 对象 result
        result = DataFrame([iter(range(10)), iter(range(10))])
        # 创建一个期望的 DataFrame 对象 expected，使用列表来初始化
        expected = DataFrame([list(range(10)), list(range(10))])
        # 使用 tm.assert_frame_equal 进行结果比较
        tm.assert_frame_equal(result, expected)

    def test_constructor_generator(self):
        # related #2305
        
        # 创建生成器 gen1 和 gen2
        gen1 = (i for i in range(10))
        gen2 = (i for i in range(10))

        # 创建一个期望的 DataFrame 对象 expected，包含两个相同的 [0, 1, ..., 9] 列表
        expected = DataFrame([list(range(10)), list(range(10))])
        # 使用生成器 gen1 和 gen2 创建 DataFrame 对象 result
        result = DataFrame([gen1, gen2])
        # 使用 tm.assert_frame_equal 进行结果比较
        tm.assert_frame_equal(result, expected)

        # 创建一个生成器 gen，每个元素为一个列表 [i, "a"]
        gen = ([i, "a"] for i in range(10))
        # 使用生成器 gen 创建 DataFrame 对象 result
        result = DataFrame(gen)
        # 创建一个期望的 DataFrame 对象 expected，包含两列：第一列为 range(10)，第二列为 "a"
        expected = DataFrame({0: range(10), 1: "a"})
        # 使用 tm.assert_frame_equal 进行结果比较，忽略数据类型的检查
        tm.assert_frame_equal(result, expected, check_dtype=False)
    # 测试以空字典为元素的列表作为数据构建 DataFrame，期望结果是一个具有单行索引和空列的 DataFrame
    def test_constructor_list_of_dicts(self):
        result = DataFrame([{}])
        expected = DataFrame(index=RangeIndex(1), columns=[])
        tm.assert_frame_equal(result, expected)

    # 测试使用有序字典构建嵌套结构的 DataFrame，并保持列的顺序不变
    # 详见 issue gh-18166
    def test_constructor_ordered_dict_nested_preserve_order(self):
        nested1 = OrderedDict([("b", 1), ("a", 2)])
        nested2 = OrderedDict([("b", 2), ("a", 5)])
        data = OrderedDict([("col2", nested1), ("col1", nested2)])
        result = DataFrame(data)
        data = {"col2": [1, 2], "col1": [2, 5]}
        expected = DataFrame(data=data, index=["b", "a"])
        tm.assert_frame_equal(result, expected)

    # 使用参数化测试（dict_type 分别为 dict 和 OrderedDict）测试使用有序字典构建 DataFrame 时保持键的顺序
    # 详见 issue gh-13304
    @pytest.mark.parametrize("dict_type", [dict, OrderedDict])
    def test_constructor_ordered_dict_preserve_order(self, dict_type):
        expected = DataFrame([[2, 1]], columns=["b", "a"])

        data = dict_type()
        data["b"] = [2]
        data["a"] = [1]

        result = DataFrame(data)
        tm.assert_frame_equal(result, expected)

        data = dict_type()
        data["b"] = 2
        data["a"] = 1

        result = DataFrame([data])
        tm.assert_frame_equal(result, expected)

    # 使用参数化测试（dict_type 分别为 dict 和 OrderedDict）测试在有多个字典时，DataFrame 构造函数使用第一个字典的顺序
    def test_constructor_ordered_dict_conflicting_orders(self, dict_type):
        # 第一个字典元素设置了 DataFrame 的顺序，即使后续字典的顺序有冲突
        row_one = dict_type()
        row_one["b"] = 2
        row_one["a"] = 1

        row_two = dict_type()
        row_two["a"] = 1
        row_two["b"] = 2

        row_three = {"b": 2, "a": 1}

        expected = DataFrame([[2, 1], [2, 1]], columns=["b", "a"])
        result = DataFrame([row_one, row_two])
        tm.assert_frame_equal(result, expected)

        expected = DataFrame([[2, 1], [2, 1], [2, 1]], columns=["b", "a"])
        result = DataFrame([row_one, row_two, row_three])
        tm.assert_frame_equal(result, expected)

    # 测试以序列列表作为数据构建 DataFrame，并确保索引对齐
    def test_constructor_list_of_series_aligned_index(self):
        series = [Series(i, index=["b", "a", "c"], name=str(i)) for i in range(3)]
        result = DataFrame(series)
        expected = DataFrame(
            {"b": [0, 1, 2], "a": [0, 1, 2], "c": [0, 1, 2]},
            columns=["b", "a", "c"],
            index=["0", "1", "2"],
        )
        tm.assert_frame_equal(result, expected)

    # 测试以派生自定义字典类的字典列表作为数据构建 DataFrame
    def test_constructor_list_of_derived_dicts(self):
        class CustomDict(dict):
            pass

        d = {"a": 1.5, "b": 3}

        data_custom = [CustomDict(d)]
        data = [d]

        result_custom = DataFrame(data_custom)
        result = DataFrame(data)
        tm.assert_frame_equal(result, result_custom)
    def test_constructor_ragged(self):
        # 准备测试数据，包含两个不同长度的数组
        data = {
            "A": np.random.default_rng(2).standard_normal(10),
            "B": np.random.default_rng(2).standard_normal(8),
        }
        # 断言在创建 DataFrame 时会抛出 ValueError 异常，异常信息匹配指定字符串
        with pytest.raises(ValueError, match="All arrays must be of the same length"):
            DataFrame(data)

    def test_constructor_scalar(self):
        # 创建一个 Index 对象
        idx = Index(range(3))
        # 使用单个标量值创建 DataFrame
        df = DataFrame({"a": 0}, index=idx)
        # 创建预期的 DataFrame
        expected = DataFrame({"a": [0, 0, 0]}, index=idx)
        # 断言两个 DataFrame 相等，忽略数据类型的检查
        tm.assert_frame_equal(df, expected, check_dtype=False)

    def test_constructor_Series_copy_bug(self, float_frame):
        # 从另一个 DataFrame 中选择一列创建新的 DataFrame，然后进行复制操作
        df = DataFrame(float_frame["A"], index=float_frame.index, columns=["A"])
        df.copy()

    def test_constructor_mixed_dict_and_Series(self):
        # 创建一个包含字典和 Series 的数据结构
        data = {}
        data["A"] = {"foo": 1, "bar": 2, "baz": 3}
        data["B"] = Series([4, 3, 2, 1], index=["bar", "qux", "baz", "foo"])

        # 通过混合数据创建 DataFrame
        result = DataFrame(data)
        # 断言结果的索引是单调递增的
        assert result.index.is_monotonic_increasing

        # 使用字典包含不同数据类型时，预期会抛出 ValueError 异常
        with pytest.raises(ValueError, match="ambiguous ordering"):
            DataFrame({"A": ["a", "b"], "B": {"a": "a", "b": "b"}})

        # 使用 Series 和列表可以正常创建 DataFrame
        result = DataFrame({"A": ["a", "b"], "B": Series(["a", "b"], index=["a", "b"])})
        expected = DataFrame({"A": ["a", "b"], "B": ["a", "b"]}, index=["a", "b"])
        tm.assert_frame_equal(result, expected)

    def test_constructor_mixed_type_rows(self):
        # 处理包含不同类型行的数据创建 DataFrame 的问题
        # Issue 25075
        data = [[1, 2], (3, 4)]
        result = DataFrame(data)
        expected = DataFrame([[1, 2], [3, 4]])
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "tuples,lists",
        [
            # 使用不同的元组和列表参数化测试数据
            ((), []),
            (((),), [[]]),
            (((), ()), [(), ()]),
            (((), ()), [[], []]),
            (([], []), [[], []]),
            (([1], [2]), [[1], [2]]),  # GH 32776
            (([1, 2, 3], [4, 5, 6]), [[1, 2, 3], [4, 5, 6]]),
        ],
    )
    def test_constructor_tuple(self, tuples, lists):
        # 处理元组数据创建 DataFrame 的问题
        # GH 25691
        result = DataFrame(tuples)
        expected = DataFrame(lists)
        tm.assert_frame_equal(result, expected)

    def test_constructor_list_of_tuples(self):
        # 使用元组列表创建 DataFrame
        result = DataFrame({"A": [(1, 2), (3, 4)]})
        expected = DataFrame({"A": Series([(1, 2), (3, 4)])})
        tm.assert_frame_equal(result, expected)

    def test_constructor_list_of_namedtuples(self):
        # 处理命名元组列表创建 DataFrame 的问题
        # GH11181
        named_tuple = namedtuple("Pandas", list("ab"))
        tuples = [named_tuple(1, 3), named_tuple(2, 4)]
        expected = DataFrame({"a": [1, 2], "b": [3, 4]})
        result = DataFrame(tuples)
        tm.assert_frame_equal(result, expected)

        # 使用指定列名创建 DataFrame
        expected = DataFrame({"y": [1, 2], "z": [3, 4]})
        result = DataFrame(tuples, columns=["y", "z"])
        tm.assert_frame_equal(result, expected)
    # 定义测试函数，用于测试将数据类实例列表作为输入的构造函数
    def test_constructor_list_of_dataclasses(self):
        # GH21910
        # 创建一个名为Point的数据类，包含x和y两个字段，类型为整数
        Point = make_dataclass("Point", [("x", int), ("y", int)])
        
        # 创建一个包含Point实例的数据列表
        data = [Point(0, 3), Point(1, 3)]
        
        # 期望的DataFrame对象，包含"x"和"y"两列，数据分别为[0, 1]和[3, 3]
        expected = DataFrame({"x": [0, 1], "y": [3, 3]})
        
        # 使用DataFrame构造函数创建result对象，传入data列表
        result = DataFrame(data)
        
        # 使用测试框架中的assert_frame_equal函数比较result和expected是否相等
        tm.assert_frame_equal(result, expected)

    # 定义测试函数，测试包含不同数据类实例的列表作为输入的构造函数
    def test_constructor_list_of_dataclasses_with_varying_types(self):
        # GH21910
        # 创建名为Point的数据类，包含"x"和"y"两个整数字段
        Point = make_dataclass("Point", [("x", int), ("y", int)])
        
        # 创建名为HLine的数据类，包含"x0"、"x1"和"y"三个整数字段
        HLine = make_dataclass("HLine", [("x0", int), ("x1", int), ("y", int)])
        
        # 创建包含Point和HLine实例的数据列表
        data = [Point(0, 3), HLine(1, 3, 3)]
        
        # 期望的DataFrame对象，包含"x"、"y"、"x0"和"x1"四列，数据分别为[0, NaN]、[3, 3]、[NaN, 1]和[NaN, 3]
        expected = DataFrame(
            {"x": [0, np.nan], "y": [3, 3], "x0": [np.nan, 1], "x1": [np.nan, 3]}
        )
        
        # 使用DataFrame构造函数创建result对象，传入data列表
        result = DataFrame(data)
        
        # 使用测试框架中的assert_frame_equal函数比较result和expected是否相等
        tm.assert_frame_equal(result, expected)

    # 定义测试函数，测试在输入列表中包含非数据类实例时是否抛出错误
    def test_constructor_list_of_dataclasses_error_thrown(self):
        # GH21910
        # 创建名为Point的数据类，包含"x"和"y"两个整数字段
        Point = make_dataclass("Point", [("x", int), ("y", int)])
        
        # 期望引发TypeError的错误消息
        msg = "asdict() should be called on dataclass instances"
        
        # 使用pytest中的raises装饰器来检查是否抛出指定的TypeError，并匹配预期的错误消息
        with pytest.raises(TypeError, match=re.escape(msg)):
            # 尝试使用DataFrame构造函数创建result对象，传入Point实例和一个字典，期望引发TypeError
            DataFrame([Point(0, 0), {"x": 1, "y": 0}])

    # 定义测试函数，测试输入列表中包含字典的顺序对DataFrame构造的影响
    def test_constructor_list_of_dict_order(self):
        # GH10056
        # 创建包含三个字典的数据列表，每个字典包含不同顺序和不同键的键值对
        data = [
            {"First": 1, "Second": 4, "Third": 7, "Fourth": 10},
            {"Second": 5, "First": 2, "Fourth": 11, "Third": 8},
            {"Second": 6, "First": 3, "Fourth": 12, "Third": 9, "YYY": 14, "XXX": 13},
        ]
        
        # 期望的DataFrame对象，包含"First"、"Second"、"Third"、"Fourth"、"YYY"和"XXX"六列
        # 数据按照data列表中字典的键顺序进行排列，缺失的键值用NaN填充
        expected = DataFrame(
            {
                "First": [1, 2, 3],
                "Second": [4, 5, 6],
                "Third": [7, 8, 9],
                "Fourth": [10, 11, 12],
                "YYY": [None, None, 14],
                "XXX": [None, None, 13],
            }
        )
        
        # 使用DataFrame构造函数创建result对象，传入data列表
        result = DataFrame(data)
        
        # 使用测试框架中的assert_frame_equal函数比较result和expected是否相等
        tm.assert_frame_equal(result, expected)
    def test_constructor_Series_named(self):
        # 创建一个具有命名和索引的 Series 对象
        a = Series([1, 2, 3], index=["a", "b", "c"], name="x")
        # 使用该 Series 对象创建一个 DataFrame 对象
        df = DataFrame(a)
        # 断言 DataFrame 的第一列的名称为 "x"
        assert df.columns[0] == "x"
        # 断言 DataFrame 的索引与 Series 对象的索引相等
        tm.assert_index_equal(df.index, a.index)

        # ndarray 类型数据
        arr = np.random.default_rng(2).standard_normal(10)
        # 创建一个没有索引名称的 Series 对象
        s = Series(arr, name="x")
        # 使用该 Series 对象创建一个 DataFrame 对象
        df = DataFrame(s)
        # 创建一个预期的 DataFrame 对象，包含一个以 "x" 命名的列
        expected = DataFrame({"x": s})
        # 断言两个 DataFrame 对象相等
        tm.assert_frame_equal(df, expected)

        # 创建一个具有指定索引的 Series 对象
        s = Series(arr, index=range(3, 13))
        # 使用该 Series 对象创建一个 DataFrame 对象
        df = DataFrame(s)
        # 创建一个预期的 DataFrame 对象，列名为 0
        expected = DataFrame({0: s})
        # 断言两个 DataFrame 对象相等
        tm.assert_frame_equal(df, expected)

        # 预期引发 ValueError 异常，消息为指定内容
        msg = r"Shape of passed values is \(10, 1\), indices imply \(10, 2\)"
        with pytest.raises(ValueError, match=msg):
            # 创建 DataFrame 对象时指定列名
            DataFrame(s, columns=[1, 2])

        # #2234
        # 创建一个空的 Series 对象
        a = Series([], name="x", dtype=object)
        # 使用该 Series 对象创建一个 DataFrame 对象
        df = DataFrame(a)
        # 断言 DataFrame 的第一列的名称为 "x"
        assert df.columns[0] == "x"

        # 创建一个具有名称和未命名的 Series 对象
        s1 = Series(arr, name="x")
        # 使用列表创建一个转置的 DataFrame 对象
        df = DataFrame([s1, arr]).T
        # 创建一个预期的 DataFrame 对象，包含 "x" 和未命名列
        expected = DataFrame({"x": s1, "Unnamed 0": arr}, columns=["x", "Unnamed 0"])
        # 断言两个 DataFrame 对象相等
        tm.assert_frame_equal(df, expected)

        # 创建一个转置的 DataFrame 对象
        df = DataFrame([arr, s1]).T
        # 创建一个预期的 DataFrame 对象，列名为 0 和 1
        expected = DataFrame({1: s1, 0: arr}, columns=[0, 1])
        # 断言两个 DataFrame 对象相等
        tm.assert_frame_equal(df, expected)
    def test_constructor_index_names(self, name_in1, name_in2, name_in3, name_out):
        # GH13475
        # 创建三个不同的 Index 对象，每个对象包含不同的名称列表
        indices = [
            Index(["a", "b", "c"], name=name_in1),
            Index(["b", "c", "d"], name=name_in2),
            Index(["c", "d", "e"], name=name_in3),
        ]
        # 创建一个包含 Series 对象的字典，每个 Series 对象使用对应的 Index 对象和固定的数据
        series = {
            c: Series([0, 1, 2], index=i) for i, c in zip(indices, ["x", "y", "z"])
        }
        # 使用上述 series 字典创建 DataFrame 对象
        result = DataFrame(series)

        # 创建一个预期的 Index 对象，包含合并后的名称列表和指定的名称
        exp_ind = Index(["a", "b", "c", "d", "e"], name=name_out)
        # 使用 exp_ind 和指定的数据创建预期的 DataFrame 对象
        expected = DataFrame(
            {
                "x": [0, 1, 2, np.nan, np.nan],
                "y": [np.nan, 0, 1, 2, np.nan],
                "z": [np.nan, np.nan, 0, 1, 2],
            },
            index=exp_ind,
        )

        # 使用 assert_frame_equal 函数比较 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)

    def test_constructor_manager_resize(self, float_frame):
        # 获取 float_frame 的前五个索引和前三个列，并转换为列表
        index = list(float_frame.index[:5])
        columns = list(float_frame.columns[:3])

        # 设置警告消息内容
        msg = "Passing a BlockManager to DataFrame"
        # 使用 assert_produces_warning 上下文管理器检查是否产生 DeprecationWarning 类型的警告，并匹配指定消息
        with tm.assert_produces_warning(
            DeprecationWarning, match=msg, check_stacklevel=False
        ):
            # 使用 float_frame 的 BlockManager 创建一个新的 DataFrame 对象，指定索引和列
            result = DataFrame(float_frame._mgr, index=index, columns=columns)
        
        # 使用 assert_index_equal 函数比较 result 的索引和 columns 是否符合预期
        tm.assert_index_equal(result.index, Index(index))
        tm.assert_index_equal(result.columns, Index(columns))

    def test_constructor_mix_series_nonseries(self, float_frame):
        # 创建一个包含 "A" 和 "B" 两列的 DataFrame 对象，其中 "A" 列来自 float_frame，"B" 列来自 float_frame 的列表形式
        df = DataFrame(
            {"A": float_frame["A"], "B": list(float_frame["B"])}, columns=["A", "B"]
        )
        # 使用 assert_frame_equal 函数比较 df 和 float_frame 的部分列是否相等
        tm.assert_frame_equal(df, float_frame.loc[:, ["A", "B"]])

        # 设置错误消息内容
        msg = "does not match index length"
        # 使用 pytest.raises 上下文管理器检查是否产生 ValueError 类型的异常，并匹配指定消息
        with pytest.raises(ValueError, match=msg):
            # 尝试创建一个 DataFrame 对象，其中 "A" 列来自 float_frame，"B" 列来自 float_frame 的部分列（长度比原列少两个元素）
            DataFrame({"A": float_frame["A"], "B": list(float_frame["B"])[:-2]})

    def test_constructor_miscast_na_int_dtype(self):
        # 设置错误消息内容，匹配指定的异常信息
        msg = r"Cannot convert non-finite values \(NA or inf\) to integer"

        # 使用 pytest.raises 上下文管理器检查是否产生 IntCastingNaNError 类型的异常，并匹配指定消息
        with pytest.raises(IntCastingNaNError, match=msg):
            # 尝试创建一个包含 NaN 的 np.int64 类型的 DataFrame 对象
            DataFrame([[np.nan, 1], [1, 0]], dtype=np.int64)

    def test_constructor_column_duplicates(self):
        # 创建包含重复列名 "a" 的 DataFrame 对象 df 和 edf
        df = DataFrame([[8, 5]], columns=["a", "a"])
        edf = DataFrame([[8, 5]])
        # 将 edf 的列名设置为重复的 "a"
        edf.columns = ["a", "a"]

        # 使用 assert_frame_equal 函数比较 df 和 edf 是否相等
        tm.assert_frame_equal(df, edf)

        # 使用 from_records 方法创建包含重复列名 "a" 的 DataFrame 对象 idf
        idf = DataFrame.from_records([(8, 5)], columns=["a", "a"])

        # 使用 assert_frame_equal 函数比较 idf 和 edf 是否相等
        tm.assert_frame_equal(idf, edf)

    def test_constructor_empty_with_string_dtype(self):
        # 创建一个预期的空 DataFrame 对象，指定索引和列的名称以及数据类型为 object
        expected = DataFrame(index=[0, 1], columns=[0, 1], dtype=object)

        # 创建三个不同的 DataFrame 对象，指定索引和列的名称及数据类型分别为 str、np.str_、"U5"，并与预期的 DataFrame 对象比较
        df = DataFrame(index=[0, 1], columns=[0, 1], dtype=str)
        tm.assert_frame_equal(df, expected)
        df = DataFrame(index=[0, 1], columns=[0, 1], dtype=np.str_)
        tm.assert_frame_equal(df, expected)
        df = DataFrame(index=[0, 1], columns=[0, 1], dtype="U5")
        tm.assert_frame_equal(df, expected)
    # 定义一个测试方法，测试在使用空数据框和字符串类型的数据类型时的构造函数行为
    def test_constructor_empty_with_string_extension(self, nullable_string_dtype):
        # 期望创建一个指定列和数据类型的数据框
        expected = DataFrame(columns=["c1"], dtype=nullable_string_dtype)
        # 创建一个空的数据框，指定列和数据类型
        df = DataFrame(columns=["c1"], dtype=nullable_string_dtype)
        # 使用测试工具断言两个数据框是否相等
        tm.assert_frame_equal(df, expected)

    # 定义一个测试方法，测试在使用单个值构造数据框时的行为
    def test_constructor_single_value(self):
        # 在此处期望进行单个值的类型提升
        # 创建一个浮点型数据框，指定索引和列
        df = DataFrame(0.0, index=[1, 2, 3], columns=["a", "b", "c"])
        # 使用测试工具断言两个数据框是否相等，通过创建一个全零浮点型数组来验证
        tm.assert_frame_equal(
            df, DataFrame(np.zeros(df.shape).astype("float64"), df.index, df.columns)
        )

        # 创建一个整型数据框，指定索引和列
        df = DataFrame(0, index=[1, 2, 3], columns=["a", "b", "c"])
        # 使用测试工具断言两个数据框是否相等，通过创建一个全零整型数组来验证
        tm.assert_frame_equal(
            df, DataFrame(np.zeros(df.shape).astype("int64"), df.index, df.columns)
        )

        # 创建一个字符串类型数据框，指定索引和列
        df = DataFrame("a", index=[1, 2], columns=["a", "c"])
        # 使用测试工具断言两个数据框是否相等，通过创建一个指定字符串的二维数组来验证
        tm.assert_frame_equal(
            df,
            DataFrame(
                np.array([["a", "a"], ["a", "a"]], dtype=object),
                index=[1, 2],
                columns=["a", "c"],
            ),
        )

        # 验证在不正确调用数据框构造函数时会引发特定错误消息的异常
        msg = "DataFrame constructor not properly called!"
        with pytest.raises(ValueError, match=msg):
            DataFrame("a", [1, 2])
        with pytest.raises(ValueError, match=msg):
            DataFrame("a", columns=["a", "c"])

        # 验证在数据和数据类型不兼容时会引发特定错误消息的异常
        msg = "incompatible data and dtype"
        with pytest.raises(TypeError, match=msg):
            DataFrame("a", [1, 2], ["a", "c"], float)
    # 使用特定的推断字符串类型，获取整数的数据类型名称
    intname = np.dtype(int).name
    # 获取 np.float64 数据类型的名称
    floatname = np.dtype(np.float64).name
    # 获取 np.object_ 数据类型的名称
    objectname = np.dtype(np.object_).name

    # 创建 DataFrame 对象，包含五列数据：整数、字符串、字符串、时间戳和日期时间
    df = DataFrame(
        {
            "A": 1,
            "B": "foo",
            "C": "bar",
            "D": Timestamp("20010101"),
            "E": datetime(2001, 1, 2, 0, 0),
        },
        index=np.arange(10),  # 指定索引为 0 到 9 的整数序列
    )
    # 获取 DataFrame 各列的数据类型
    result = df.dtypes
    # 期望的数据类型 Series 对象，包含五种数据类型
    expected = Series(
        [np.dtype("int64")]
        + [np.dtype(objectname) if not using_infer_string else "string"] * 2
        + [np.dtype("M8[s]"), np.dtype("M8[us]")],
        index=list("ABCDE"),  # 指定索引为 'A' 到 'E' 的字符序列
    )
    # 断言 DataFrame 各列的数据类型是否与期望一致
    tm.assert_series_equal(result, expected)

    # 使用 ndarray 构造 DataFrame 对象，其中包含四列数据：浮点数、整数、字符串和两个特定数据类型的 ndarray
    df = DataFrame(
        {
            "a": 1.0,
            "b": 2,
            "c": "foo",
            floatname: np.array(1.0, dtype=floatname),
            intname: np.array(1, dtype=intname),
        },
        index=np.arange(10),  # 指定索引为 0 到 9 的整数序列
    )
    # 获取 DataFrame 各列的数据类型
    result = df.dtypes
    # 期望的数据类型 Series 对象，包含五种数据类型
    expected = Series(
        [np.dtype("float64")]
        + [np.dtype("int64")]
        + [np.dtype("object") if not using_infer_string else "string"]
        + [np.dtype("float64")]
        + [np.dtype(intname)],
        index=["a", "b", "c", floatname, intname],  # 指定索引为 'a', 'b', 'c', floatname, intname
    )
    # 断言 DataFrame 各列的数据类型是否与期望一致
    tm.assert_series_equal(result, expected)

    # 使用 ndarray 构造 DataFrame 对象，其中包含四列数据：浮点数、整数、字符串和两个特定数据类型的 ndarray，ndarray 的维度大于 0
    df = DataFrame(
        {
            "a": 1.0,
            "b": 2,
            "c": "foo",
            floatname: np.array([1.0] * 10, dtype=floatname),
            intname: np.array([1] * 10, dtype=intname),
        },
        index=np.arange(10),  # 指定索引为 0 到 9 的整数序列
    )
    # 获取 DataFrame 各列的数据类型
    result = df.dtypes
    # 期望的数据类型 Series 对象，包含五种数据类型
    expected = Series(
        [np.dtype("float64")]
        + [np.dtype("int64")]
        + [np.dtype("object") if not using_infer_string else "string"]
        + [np.dtype("float64")]
        + [np.dtype(intname)],
        index=["a", "b", "c", floatname, intname],  # 指定索引为 'a', 'b', 'c', floatname, intname
    )
    # 断言 DataFrame 各列的数据类型是否与期望一致
    tm.assert_series_equal(result, expected)

    # 创建时间范围为 "2000-01-01" 开始，频率为每日，共 10 个时间点的日期时间索引
    ind = date_range(start="2000-01-01", freq="D", periods=10)
    # 将时间戳转换为 Python datetime 对象，存储在列表中
    datetimes = [ts.to_pydatetime() for ts in ind]
    # 创建日期时间的 Series 对象
    datetime_s = Series(datetimes)
    # 断言 Series 对象的数据类型是否为 "M8[us]"
    assert datetime_s.dtype == "M8[us]"
    def test_constructor_with_datetimes2(self):
        # GH 2810
        # 创建一个日期范围索引，从"2000-01-01"开始，每天一个频率，共10个周期
        ind = date_range(start="2000-01-01", freq="D", periods=10)
        # 将日期时间对象转换为 Python 的 datetime 对象
        datetimes = [ts.to_pydatetime() for ts in ind]
        # 提取日期部分
        dates = [ts.date() for ts in ind]
        # 创建一个包含 "datetimes" 和 "dates" 列的 DataFrame
        df = DataFrame(datetimes, columns=["datetimes"])
        df["dates"] = dates
        # 返回 DataFrame 中各列的数据类型
        result = df.dtypes
        # 创建预期的数据类型 Series
        expected = Series(
            [np.dtype("datetime64[us]"), np.dtype("object")],
            index=["datetimes", "dates"],
        )
        # 检查结果是否与预期相等
        tm.assert_series_equal(result, expected)

    def test_constructor_with_datetimes3(self):
        # GH 7594
        # 创建一个带时区信息的 datetime 对象
        dt = datetime(2012, 1, 1, tzinfo=zoneinfo.ZoneInfo("US/Eastern"))

        # 创建包含带时区信息的 datetime 列的 DataFrame
        df = DataFrame({"End Date": dt}, index=[0])
        # 断言 DataFrame 中指定位置的值与预期的 datetime 对象相等
        assert df.iat[0, 0] == dt
        # 检查 DataFrame 列的数据类型是否符合预期
        tm.assert_series_equal(
            df.dtypes, Series({"End Date": "datetime64[us, US/Eastern]"}, dtype=object)
        )

        # 创建包含带时区信息的 datetime 对象的 DataFrame
        df = DataFrame([{"End Date": dt}])
        # 断言 DataFrame 中指定位置的值与预期的 datetime 对象相等
        assert df.iat[0, 0] == dt
        # 检查 DataFrame 列的数据类型是否符合预期
        tm.assert_series_equal(
            df.dtypes, Series({"End Date": "datetime64[us, US/Eastern]"}, dtype=object)
        )

    def test_constructor_with_datetimes4(self):
        # tz-aware (UTC and other tz's)
        # GH 8411
        # 创建不带时区信息的日期范围
        dr = date_range("20130101", periods=3)
        # 创建包含日期范围的 DataFrame
        df = DataFrame({"value": dr})
        # 断言 DataFrame 中指定位置的值的时区为空
        assert df.iat[0, 0].tz is None
        # 创建带 UTC 时区信息的日期范围
        dr = date_range("20130101", periods=3, tz="UTC")
        # 创建包含日期范围的 DataFrame
        df = DataFrame({"value": dr})
        # 断言 DataFrame 中指定位置的值的时区为 UTC
        assert str(df.iat[0, 0].tz) == "UTC"
        # 创建带 US/Eastern 时区信息的日期范围
        dr = date_range("20130101", periods=3, tz="US/Eastern")
        # 创建包含日期范围的 DataFrame
        df = DataFrame({"value": dr})
        # 断言 DataFrame 中指定位置的值的时区为 US/Eastern
        assert str(df.iat[0, 0].tz) == "US/Eastern"

    def test_constructor_with_datetimes5(self):
        # GH 7822
        # 创建带时区信息的日期范围索引
        i = date_range("1/1/2011", periods=5, freq="10s", tz="US/Eastern")

        # 创建预期的 DataFrame，包含 "a" 列并保持索引的时区信息
        expected = DataFrame({"a": i.to_series().reset_index(drop=True)})
        # 创建一个空的 DataFrame
        df = DataFrame()
        # 向 DataFrame 添加 "a" 列
        df["a"] = i
        # 检查 DataFrame 是否与预期相等
        tm.assert_frame_equal(df, expected)

        # 创建包含带时区信息的日期范围索引的 DataFrame
        df = DataFrame({"a": i})
        # 检查 DataFrame 是否与预期相等
        tm.assert_frame_equal(df, expected)

    def test_constructor_with_datetimes6(self):
        # multiples
        # 创建带时区信息和不带时区信息的日期范围索引
        i = date_range("1/1/2011", periods=5, freq="10s", tz="US/Eastern")
        i_no_tz = date_range("1/1/2011", periods=5, freq="10s")
        # 创建包含 "a" 和 "b" 两列的 DataFrame
        df = DataFrame({"a": i, "b": i_no_tz})
        # 创建预期的 DataFrame，包含 "a" 列并保持索引的时区信息
        expected = DataFrame({"a": i.to_series().reset_index(drop=True), "b": i_no_tz})
        # 检查 DataFrame 是否与预期相等
        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize(
        "arr",
        [
            # 不同的 NaN 值组合，包含 datetime 对象
            np.array([None, None, None, None, datetime.now(), None]),
            np.array([None, None, datetime.now(), None]),
            [[np.datetime64("NaT")], [None]],
            [[np.datetime64("NaT")], [pd.NaT]],
            [[None], [np.datetime64("NaT")]],
            [[None], [pd.NaT]],
            [[pd.NaT], [np.datetime64("NaT")]],
            [[pd.NaT], [None]],
        ],
    )
    # 测试函数：test_constructor_datetimes_with_nulls，接受一个名为arr的参数
    def test_constructor_datetimes_with_nulls(self, arr):
        # 注释：测试用例标识，关联GitHub问题号GH-15869和GH-11220
        result = DataFrame(arr).dtypes
        # 单位初始化为纳秒
        unit = "ns"
        if isinstance(arr, np.ndarray):
            # 如果arr是numpy数组，推断单位为微秒
            unit = "us"
        elif not any(isinstance(x, np.datetime64) for y in arr for x in y):
            # 如果arr中没有任何元素是np.datetime64类型，单位设置为秒
            # TODO: 此条件不太清楚为什么会有不同的行为
            unit = "s"
        # 期望结果是一个包含np.dtype对象的Series
        expected = Series([np.dtype(f"datetime64[{unit}]")])
        # 断言测试结果与期望结果相等
        tm.assert_series_equal(result, expected)

    # 参数化测试函数，使用order和unit作为参数
    @pytest.mark.parametrize("order", ["K", "A", "C", "F"])
    @pytest.mark.parametrize(
        "unit",
        ["M", "D", "h", "m", "s", "ms", "us", "ns"],
    )
    def test_constructor_datetimes_non_ns(self, order, unit):
        # 构建dtype字符串
        dtype = f"datetime64[{unit}]"
        # 创建一个numpy数组na，包含日期时间字符串，使用指定的dtype和order
        na = np.array(
            [
                ["2015-01-01", "2015-01-02", "2015-01-03"],
                ["2017-01-01", "2017-01-02", "2017-02-03"],
            ],
            dtype=dtype,
            order=order,
        )
        # 使用na创建DataFrame对象df
        df = DataFrame(na)
        # 期望结果是一个包含M8[ns]类型的DataFrame
        expected = DataFrame(na.astype("M8[ns]"))
        if unit in ["M", "D", "h", "m"]:
            # 如果单位是月、天、小时或分钟，预期会抛出TypeError异常，匹配异常信息"Cannot cast"
            with pytest.raises(TypeError, match="Cannot cast"):
                expected.astype(dtype)
            # 替代方案是将DataFrame强制转换为最接近的支持单位，即"datetime64[s]"
            expected = expected.astype("datetime64[s]")
        else:
            # 否则，按照指定的dtype进行强制类型转换
            expected = expected.astype(dtype=dtype)
        # 断言DataFrame df与期望的DataFrame expected相等
        tm.assert_frame_equal(df, expected)

    # 参数化测试函数，使用order和unit作为参数
    @pytest.mark.parametrize("order", ["K", "A", "C", "F"])
    @pytest.mark.parametrize(
        "unit",
        [
            "D",
            "h",
            "m",
            "s",
            "ms",
            "us",
            "ns",
        ],
    )
    def test_constructor_timedelta_non_ns(self, order, unit):
        # 构建dtype字符串
        dtype = f"timedelta64[{unit}]"
        # 创建一个numpy数组na，包含时间间隔对象，使用指定的dtype和order
        na = np.array(
            [
                [np.timedelta64(1, "D"), np.timedelta64(2, "D")],
                [np.timedelta64(4, "D"), np.timedelta64(5, "D")],
            ],
            dtype=dtype,
            order=order,
        )
        # 使用na创建DataFrame对象df
        df = DataFrame(na)
        if unit in ["D", "h", "m"]:
            # 如果单位是天、小时或分钟，期望的单位是"秒"
            exp_unit = "s"
        else:
            # 否则，期望的单位与传入的单位一致
            exp_unit = unit
        # 构建期望的dtype对象
        exp_dtype = np.dtype(f"m8[{exp_unit}]")
        # 创建期望的DataFrame对象，指定dtype
        expected = DataFrame(
            [
                [Timedelta(1, "D"), Timedelta(2, "D")],
                [Timedelta(4, "D"), Timedelta(5, "D")],
            ],
            dtype=exp_dtype,
        )
        # TODO(2.0): 理想情况下，我们应该在不传递dtype=exp_dtype的情况下获得相同的'expected'。
        # 断言DataFrame df与期望的DataFrame expected相等
        tm.assert_frame_equal(df, expected)
    def test_constructor_for_list_with_dtypes(self, using_infer_string):
        # 测试包含数据类型的列表/ndarrays的构造函数

        # 创建一个 DataFrame，其中每一行是一个由 0 到 4 组成的 ndarray，共创建 5 行
        df = DataFrame([np.arange(5) for x in range(5)])
        # 获取 DataFrame 的列数据类型
        result = df.dtypes
        # 期望每列数据类型为整数类型，共 5 列
        expected = Series([np.dtype("int")] * 5)
        # 断言 DataFrame 的列数据类型与期望值相等
        tm.assert_series_equal(result, expected)

        # 创建一个 DataFrame，其中每一行是一个由 0 到 4 组成的 int32 类型的 ndarray，共创建 5 行
        df = DataFrame([np.array(np.arange(5), dtype="int32") for x in range(5)])
        # 获取 DataFrame 的列数据类型
        result = df.dtypes
        # 期望每列数据类型为 int32 类型，共 5 列
        expected = Series([np.dtype("int32")] * 5)
        # 断言 DataFrame 的列数据类型与期望值相等
        tm.assert_series_equal(result, expected)

        # 创建一个 DataFrame，其中包含一个列 'a'，其值为 [2**31, 2**31 + 1]
        df = DataFrame({"a": [2**31, 2**31 + 1]})
        # 断言 DataFrame 第一列的数据类型为 int64
        assert df.dtypes.iloc[0] == np.dtype("int64")

        # 创建一个 DataFrame，其中包含一列整数值 [1, 2]，未指定索引
        df = DataFrame([1, 2])
        # 断言 DataFrame 第一列的数据类型为 int64
        assert df.dtypes.iloc[0] == np.dtype("int64")

        # 创建一个 DataFrame，其中包含一列浮点数值 [1.0, 2.0]
        df = DataFrame([1.0, 2.0])
        # 断言 DataFrame 第一列的数据类型为 float64
        assert df.dtypes.iloc[0] == np.dtype("float64")

        # 创建一个 DataFrame，其中包含一列 'a'，其值为 [1, 2]
        df = DataFrame({"a": [1, 2]})
        # 断言 DataFrame 第一列的数据类型为 int64
        assert df.dtypes.iloc[0] == np.dtype("int64")

        # 创建一个 DataFrame，其中包含一列 'a'，其值为 [1.0, 2.0]
        df = DataFrame({"a": [1.0, 2.0]})
        # 断言 DataFrame 第一列的数据类型为 float64
        assert df.dtypes.iloc[0] == np.dtype("float64")

        # 创建一个 DataFrame，其中包含一列 'a'，值为整数 1，索引为 [0, 1, 2]
        df = DataFrame({"a": 1}, index=range(3))
        # 断言 DataFrame 第一列的数据类型为 int64
        assert df.dtypes.iloc[0] == np.dtype("int64")

        # 创建一个 DataFrame，其中包含一列 'a'，值为浮点数 1.0，索引为 [0, 1, 2]
        df = DataFrame({"a": 1.0}, index=range(3))
        # 断言 DataFrame 第一列的数据类型为 float64
        assert df.dtypes.iloc[0] == np.dtype("float64")

        # 创建一个 DataFrame，包含多列数据，其中包括整数、浮点数、字符串、日期时间以及浮点数类型的混合
        df = DataFrame(
            {
                "a": [1, 2, 4, 7],
                "b": [1.2, 2.3, 5.1, 6.3],
                "c": list("abcd"),
                "d": [datetime(2000, 1, 1) for i in range(4)],
                "e": [1.0, 2, 4.0, 7],
            }
        )
        # 获取 DataFrame 的列数据类型
        result = df.dtypes
        # 期望每列数据类型与其名称对应，包括 int64、float64、object（或 string，取决于 using_infer_string 的值）、datetime64[us]、float64
        expected = Series(
            [
                np.dtype("int64"),
                np.dtype("float64"),
                np.dtype("object") if not using_infer_string else "string",
                np.dtype("datetime64[us]"),
                np.dtype("float64"),
            ],
            index=list("abcde"),
        )
        # 断言 DataFrame 的列数据类型与期望值相等
        tm.assert_series_equal(result, expected)

    def test_constructor_frame_copy(self, float_frame):
        # 测试 DataFrame 的构造函数是否正确执行深拷贝

        # 从 float_frame 深拷贝创建 DataFrame cop
        cop = DataFrame(float_frame, copy=True)
        # 修改 cop 的列 'A' 的所有值为 5
        cop["A"] = 5
        # 断言 cop 的列 'A' 的所有值为 5
        assert (cop["A"] == 5).all()
        # 断言 float_frame 的列 'A' 的所有值不全为 5
        assert not (float_frame["A"] == 5).all()

    def test_constructor_frame_shallow_copy(self, float_frame):
        # 测试 DataFrame 的构造函数是否正确执行浅拷贝

        # 从 float_frame 创建一个 DataFrame cop，copy=False 仍应生成一个“浅”拷贝（共享数据，不共享属性）
        orig = float_frame.copy()
        cop = DataFrame(float_frame)
        # 断言 cop 的数据管理器 _mgr 不等于 float_frame 的数据管理器 _mgr
        assert cop._mgr is not float_frame._mgr
        # 修改 cop 的索引，不会改变原始 float_frame 的索引
        cop.index = np.arange(len(cop))
        # 断言 float_frame 和 orig 相等
        tm.assert_frame_equal(float_frame, orig)
    # 测试函数，用于测试通过复制构造DataFrame对象时的行为
    def test_constructor_ndarray_copy(self, float_frame):
        # 从 float_frame 的值复制一个数组
        arr = float_frame.values.copy()
        # 用复制的数组创建一个DataFrame对象
        df = DataFrame(arr)

        # 修改原始数组中的第5个元素
        arr[5] = 5
        # 断言：检查 DataFrame 中的第5个值是否不全等于5
        assert not (df.values[5] == 5).all()

        # 用复制数组再次创建一个DataFrame对象
        df = DataFrame(arr, copy=True)
        # 修改原始数组中的第6个元素
        arr[6] = 6
        # 断言：检查 DataFrame 中的第6个值是否不全等于6
        assert not (df.values[6] == 6).all()

    # 测试函数，用于测试通过复制构造Series对象时的行为
    def test_constructor_series_copy(self, float_frame):
        # 从 float_frame 中获取 Series 对象
        series = float_frame._series

        # 用 Series 对象的一部分创建一个DataFrame对象，通过复制方式
        df = DataFrame({"A": series["A"]}, copy=True)
        # 使用.loc方式修改DataFrame中的值，将第一行的"A"列值设置为5
        # TODO 在不违反关于就地变异的弃用警告后，可以替换为 `df.loc[:, "A"] = 5`
        df.loc[df.index[0] : df.index[-1], "A"] = 5

        # 断言：检查 Series 中的"A"列是否不全等于5
        assert not (series["A"] == 5).all()

    # 使用参数化测试装饰器标记的测试函数，用于测试具有NaN值的DataFrame对象的构造行为
    @pytest.mark.parametrize(
        "df",
        [
            DataFrame([[1, 2, 3], [4, 5, 6]], index=[1, np.nan]),
            DataFrame([[1, 2, 3], [4, 5, 6]], columns=[1.1, 2.2, np.nan]),
            DataFrame([[0, 1, 2, 3], [4, 5, 6, 7]], columns=[np.nan, 1.1, 2.2, np.nan]),
            DataFrame(
                [[0.0, 1, 2, 3.0], [4, 5, 6, 7]], columns=[np.nan, 1.1, 2.2, np.nan]
            ),
            DataFrame([[0.0, 1, 2, 3.0], [4, 5, 6, 7]], columns=[np.nan, 1, 2, 2]),
        ],
    )
    def test_constructor_with_nas(self, df):
        # GH 5016
        # 索引中的NaN值
        # GH 21428（非唯一列）

        # 遍历DataFrame对象的每一列
        for i in range(len(df.columns)):
            df.iloc[:, i]

        # 使用isna函数获取列索引中NaN的位置
        indexer = np.arange(len(df.columns))[isna(df.columns)]

        # 如果没有找到NaN值，抛出KeyError错误
        if len(indexer) == 0:
            with pytest.raises(KeyError, match="^nan$"):
                df.loc[:, np.nan]
        # 如果只有一个NaN值，结果应该是Series对象
        elif len(indexer) == 1:
            tm.assert_series_equal(df.iloc[:, indexer[0]], df.loc[:, np.nan])
        # 如果有多个NaN值，结果应该是DataFrame对象
        else:
            tm.assert_frame_equal(df.iloc[:, indexer], df.loc[:, np.nan])

    # 测试函数，用于测试列表转换为对象数据类型（np.object_）的行为
    def test_constructor_lists_to_object_dtype(self):
        # 来自 Issue #1074
        # 创建一个包含NaN和False的DataFrame对象
        d = DataFrame({"a": [np.nan, False]})
        # 断言：检查列"a"的数据类型是否为np.object_
        assert d["a"].dtype == np.object_
        # 断言：检查列"a"的第二个元素是否为False
        assert not d["a"][1]

    # 测试函数，用于测试从分类数据类型构造DataFrame对象的行为
    def test_constructor_ndarray_categorical_dtype(self):
        # 创建一个分类变量
        cat = Categorical(["A", "B", "C"])
        # 创建一个数组，包含分类变量的广播版本
        arr = np.array(cat).reshape(-1, 1)
        arr = np.broadcast_to(arr, (3, 4))

        # 使用分类变量的数据类型创建DataFrame对象
        result = DataFrame(arr, dtype=cat.dtype)

        # 创建一个预期的DataFrame对象，每列都是相同的分类变量
        expected = DataFrame({0: cat, 1: cat, 2: cat, 3: cat})
        # 断言：检查结果DataFrame对象与预期DataFrame对象是否相等
        tm.assert_frame_equal(result, expected)
    def test_constructor_categorical(self):
        # GH8626
        # 创建一个包含单列'A'的DataFrame，数据类型为category
        df = DataFrame({"A": list("abc")}, dtype="category")
        # 创建预期结果，包含单列'A'的Series，数据类型为category
        expected = Series(list("abc"), dtype="category", name="A")
        # 断言确认df['A']与预期结果相等
        tm.assert_series_equal(df["A"], expected)

        # 将Series转换为DataFrame
        s = Series(list("abc"), dtype="category")
        result = s.to_frame()
        # 创建预期结果，包含单列'0'的Series，数据类型为category
        expected = Series(list("abc"), dtype="category", name=0)
        # 断言确认result[0]与预期结果相等
        tm.assert_series_equal(result[0], expected)

        # 将Series转换为DataFrame，并指定列名为'foo'
        result = s.to_frame(name="foo")
        # 创建预期结果，包含单列'foo'的Series，数据类型为category
        expected = Series(list("abc"), dtype="category", name="foo")
        # 断言确认result['foo']与预期结果相等
        tm.assert_series_equal(result["foo"], expected)

        # 使用list创建DataFrame，数据类型为category
        df = DataFrame(list("abc"), dtype="category")
        # 创建预期结果，包含单列'0'的Series，数据类型为category
        expected = Series(list("abc"), dtype="category", name=0)
        # 断言确认df[0]与预期结果相等
        tm.assert_series_equal(df[0], expected)

    def test_construct_from_1item_list_of_categorical(self):
        # 在2.0之前，此操作会生成DataFrame({0: cat})，在2.0中移除了Categorical特殊情况
        # ndim != 1
        # 创建一个Categorical对象
        cat = Categorical(list("abc"))
        # 使用Categorical对象创建DataFrame
        df = DataFrame([cat])
        # 创建预期结果，包含一个转换为object类型的Categorical对象的DataFrame
        expected = DataFrame([cat.astype(object)])
        # 断言确认df与预期结果相等
        tm.assert_frame_equal(df, expected)

    def test_construct_from_list_of_categoricals(self):
        # 在2.0之前，此操作会生成DataFrame({0: cat})，在2.0中移除了Categorical特殊情况
        # 使用两个Categorical对象创建DataFrame
        df = DataFrame([Categorical(list("abc")), Categorical(list("abd"))])
        # 创建预期结果，包含两个列表["a", "b", "c"]和["a", "b", "d"]的DataFrame
        expected = DataFrame([["a", "b", "c"], ["a", "b", "d"]])
        # 断言确认df与预期结果相等
        tm.assert_frame_equal(df, expected)

    def test_from_nested_listlike_mixed_types(self):
        # 在2.0之前，此操作会生成DataFrame({0: cat})，在2.0中移除了Categorical特殊情况
        # mixed
        # 使用一个Categorical对象和一个列表创建DataFrame
        df = DataFrame([Categorical(list("abc")), list("def")])
        # 创建预期结果，包含两个列表["a", "b", "c"]和["d", "e", "f"]的DataFrame
        expected = DataFrame([["a", "b", "c"], ["d", "e", "f"]])
        # 断言确认df与预期结果相等
        tm.assert_frame_equal(df, expected)

    def test_construct_from_listlikes_mismatched_lengths(self):
        # 使用两个长度不同的Categorical对象创建DataFrame
        df = DataFrame([Categorical(list("abc")), Categorical(list("abdefg"))])
        # 创建预期结果，包含两个列表["a", "b", "c"]和["a", "b", "d", "e", "f", "g"]的DataFrame
        expected = DataFrame([list("abc"), list("abdefg")])
        # 断言确认df与预期结果相等
        tm.assert_frame_equal(df, expected)

    def test_constructor_categorical_series(self):
        # 创建一个包含整数和字符的Series，数据类型转换为category
        items = [1, 2, 3, 1]
        exp = Series(items).astype("category")
        res = Series(items, dtype="category")
        # 断言确认res与exp相等
        tm.assert_series_equal(res, exp)

        # 创建一个包含字符的Series，数据类型转换为category
        items = ["a", "b", "c", "a"]
        exp = Series(items).astype("category")
        res = Series(items, dtype="category")
        # 断言确认res与exp相等
        tm.assert_series_equal(res, exp)

        # 将Series插入到具有不同索引的DataFrame中
        # GH 8076
        # 创建一个日期范围为'20000101'到'20000103'的索引
        index = date_range("20000101", periods=3)
        # 创建一个包含NaN值和类别为["a", "b", "c"]的Series
        expected = Series(
            Categorical(values=[np.nan, np.nan, np.nan], categories=["a", "b", "c"])
        )
        expected.index = index
        # 创建预期结果，包含单列'x'的DataFrame
        expected = DataFrame({"x": expected})
        # 创建一个包含类别为["a", "b", "c"]的Series，并将其插入到具有日期索引的DataFrame中
        df = DataFrame({"x": Series(["a", "b", "c"], dtype="category")}, index=index)
        # 断言确认df与预期结果相等
        tm.assert_frame_equal(df, expected)
    @pytest.mark.parametrize(
        "dtype",
        tm.ALL_NUMERIC_DTYPES
        + tm.DATETIME64_DTYPES
        + tm.TIMEDELTA64_DTYPES
        + tm.BOOL_DTYPES,
    )
    # 使用 pytest 的参数化标记，为单元测试 test_check_dtype_empty_numeric_column 提供多个数据类型的参数化输入
    def test_check_dtype_empty_numeric_column(self, dtype):
        # GH24386: 确保对空DataFrame设置正确的数据类型。
        # 通过非重叠列的字典数据生成空DataFrame。
        data = DataFrame({"a": [1, 2]}, columns=["b"], dtype=dtype)

        # 断言空DataFrame的指定列b的数据类型是否为预期的dtype
        assert data.b.dtype == dtype

    @pytest.mark.parametrize(
        "dtype", tm.STRING_DTYPES + tm.BYTES_DTYPES + tm.OBJECT_DTYPES
    )
    # 使用 pytest 的参数化标记，为单元测试 test_check_dtype_empty_string_column 提供多个数据类型的参数化输入
    def test_check_dtype_empty_string_column(self, dtype):
        # GH24386: 确保对空DataFrame设置正确的数据类型。
        # 通过非重叠列的字典数据生成空DataFrame。
        data = DataFrame({"a": [1, 2]}, columns=["b"], dtype=dtype)
        
        # 断言空DataFrame的指定列b的数据类型名称是否为'object'
        assert data.b.dtype.name == "object"

    def test_to_frame_with_falsey_names(self):
        # GH 16114
        # 创建一个Series，名称为0，数据类型为object，然后将其转换为DataFrame，查看其数据类型
        result = Series(name=0, dtype=object).to_frame().dtypes
        expected = Series({0: object})
        
        # 断言转换后的DataFrame的数据类型与预期是否一致
        tm.assert_series_equal(result, expected)

        # 创建一个DataFrame，以包含具有名称0和数据类型为object的Series，然后查看其数据类型
        result = DataFrame(Series(name=0, dtype=object)).dtypes
        
        # 断言转换后的DataFrame的数据类型与预期是否一致
        tm.assert_series_equal(result, expected)

    @pytest.mark.arm_slow
    @pytest.mark.parametrize("dtype", [None, "uint8", "category"])
    # 使用 pytest 的参数化标记，为单元测试 test_constructor_range_dtype 提供多个数据类型的参数化输入
    def test_constructor_range_dtype(self, dtype):
        # 创建一个预期的DataFrame，包含列'A'，值为[0, 1, 2, 3, 4]，数据类型为dtype或默认为'int64'
        expected = DataFrame({"A": [0, 1, 2, 3, 4]}, dtype=dtype or "int64")

        # GH 26342
        # 使用range(5)创建一个DataFrame，列名为'A'，数据类型为dtype
        result = DataFrame(range(5), columns=["A"], dtype=dtype)
        
        # 断言结果DataFrame与预期DataFrame是否相等
        tm.assert_frame_equal(result, expected)

        # GH 16804
        # 创建一个DataFrame，通过字典数据{'A': range(5)}，数据类型为dtype
        result = DataFrame({"A": range(5)}, dtype=dtype)
        
        # 断言结果DataFrame与预期DataFrame是否相等
        tm.assert_frame_equal(result, expected)

    def test_frame_from_list_subclass(self):
        # GH21226
        # 定义一个继承自list的子类List
        class List(list):
            pass

        # 创建一个预期的DataFrame，包含数据[[1, 2, 3], [4, 5, 6]]
        expected = DataFrame([[1, 2, 3], [4, 5, 6]])
        
        # 使用List类创建一个DataFrame，数据类型为List
        result = DataFrame(List([List([1, 2, 3]), List([4, 5, 6])]))
        
        # 断言结果DataFrame与预期DataFrame是否相等
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "extension_arr",
        [
            Categorical(list("aabbc")),
            SparseArray([1, np.nan, np.nan, np.nan]),
            IntervalArray([Interval(0, 1), Interval(1, 5)]),
            PeriodArray(pd.period_range(start="1/1/2017", end="1/1/2018", freq="M")),
        ],
    )
    # 使用 pytest 的参数化标记，为单元测试 test_constructor_with_extension_array 提供多个扩展数组类型的参数化输入
    def test_constructor_with_extension_array(self, extension_arr):
        # GH11363
        # 创建一个预期的DataFrame，包含一个Series，数据类型为extension_arr
        expected = DataFrame(Series(extension_arr))
        
        # 使用extension_arr创建一个DataFrame
        result = DataFrame(extension_arr)
        
        # 断言结果DataFrame与预期DataFrame是否相等
        tm.assert_frame_equal(result, expected)

    def test_datetime_date_tuple_columns_from_dict(self):
        # GH 10863
        # 获取当天日期
        v = date.today()
        # 创建一个包含元组列的DataFrame，列名为元组tup，Series的索引为range(3)，值为range(3)
        tup = v, v
        result = DataFrame({tup: Series(range(3), index=range(3))}, columns=[tup])
        
        # 创建一个预期的DataFrame，包含一个列名为元组tup的DataFrame，值为[0, 1, 2]
        expected = DataFrame([0, 1, 2], columns=Index(Series([tup])))
        
        # 断言结果DataFrame与预期DataFrame是否相等
        tm.assert_frame_equal(result, expected)
    def test_construct_with_two_categoricalindex_series(self):
        # 测试用例：使用两个分类索引的 Series 构造
        # GH 14600
        # 创建第一个 Series，指定自定义的分类索引
        s1 = Series([39, 6, 4], index=CategoricalIndex(["female", "male", "unknown"]))
        # 创建第二个 Series，指定另一个自定义的分类索引
        s2 = Series(
            [2, 152, 2, 242, 150],
            index=CategoricalIndex(["f", "female", "m", "male", "unknown"]),
        )
        # 构造 DataFrame，包含两个 Series，自动扩展列以匹配所有可能的分类值
        result = DataFrame([s1, s2])
        # 创建期望的 DataFrame，使用 NumPy 数组初始化，列名为分类值
        expected = DataFrame(
            np.array([[39, 6, 4, np.nan, np.nan], [152.0, 242.0, 150.0, 2.0, 2.0]]),
            columns=["female", "male", "unknown", "f", "m"],
        )
        # 使用测试工具比较两个 DataFrame 是否相等
        tm.assert_frame_equal(result, expected)

    def test_constructor_series_nonexact_categoricalindex(self):
        # 测试用例：使用非精确匹配的分类索引构造 Series
        # GH 42424
        # 创建 Series 对象，并将其划分为 10 个区间，并统计各区间内的值
        ser = Series(range(100))
        ser1 = cut(ser, 10).value_counts().head(5)
        ser2 = cut(ser, 10).value_counts().tail(5)
        # 构造 DataFrame，包含两个 Series，列名分别为 "1" 和 "2"
        result = DataFrame({"1": ser1, "2": ser2})
        # 创建期望的 DataFrame，包含两列，索引为 Interval 类型的分类索引
        index = CategoricalIndex(
            [
                Interval(-0.099, 9.9, closed="right"),
                Interval(9.9, 19.8, closed="right"),
                Interval(19.8, 29.7, closed="right"),
                Interval(29.7, 39.6, closed="right"),
                Interval(39.6, 49.5, closed="right"),
                Interval(49.5, 59.4, closed="right"),
                Interval(59.4, 69.3, closed="right"),
                Interval(69.3, 79.2, closed="right"),
                Interval(79.2, 89.1, closed="right"),
                Interval(89.1, 99, closed="right"),
            ],
            ordered=True,
        )
        expected = DataFrame(
            {"1": [10] * 5 + [np.nan] * 5, "2": [np.nan] * 5 + [10] * 5}, index=index
        )
        # 使用测试工具比较两个 DataFrame 是否相等
        tm.assert_frame_equal(expected, result)

    def test_from_M8_structured(self):
        # 测试用例：从结构化的 M8 类型数据创建 DataFrame
        dates = [(datetime(2012, 9, 9, 0, 0), datetime(2012, 9, 8, 15, 10))]
        # 使用 NumPy 的结构化数组创建 DataFrame
        arr = np.array(dates, dtype=[("Date", "M8[us]"), ("Forecasting", "M8[us]")])
        df = DataFrame(arr)

        # 断言 DataFrame 的列中的值与预期的日期时间值相等
        assert df["Date"][0] == dates[0][0]
        assert df["Forecasting"][0] == dates[0][1]

        # 从结构化数组中的一个列创建 Series，并断言其类型为 Timestamp，值与预期的日期时间值相等
        s = Series(arr["Date"])
        assert isinstance(s[0], Timestamp)
        assert s[0] == dates[0][0]

    def test_from_datetime_subclass(self):
        # 测试用例：从日期时间子类创建 DataFrame
        # GH21142 验证日期时间的子类是否也属于 datetime 类型
        class DatetimeSubclass(datetime):
            pass

        # 创建包含日期时间子类的 DataFrame
        data = DataFrame({"datetime": [DatetimeSubclass(2020, 1, 1, 1, 1)]})
        # 断言 DataFrame 的列的数据类型为 datetime64[us]
        assert data.datetime.dtype == "datetime64[us]"

    def test_with_mismatched_index_length_raises(self):
        # 测试用例：当索引长度不匹配时引发错误
        # GH#33437
        # 创建一个日期范围的 DatetimeIndex 对象
        dti = date_range("2016-01-01", periods=3, tz="US/Pacific")
        # 准备错误消息
        msg = "Shape of passed values|Passed arrays should have the same length"
        # 使用 pytest 检查在创建 DataFrame 时是否会引发 ValueError，且错误消息匹配预期
        with pytest.raises(ValueError, match=msg):
            DataFrame(dti, index=range(4))
    # 定义测试函数，用于测试DataFrame类构造函数是否正确处理datetime64类型的列
    def test_frame_ctor_datetime64_column(self):
        # 生成一个日期范围，频率为每10秒
        rng = date_range("1/1/2000 00:00:00", "1/1/2000 1:59:50", freq="10s")
        # 将日期范围转换为NumPy数组
        dates = np.asarray(rng)

        # 创建一个DataFrame对象，包含"A"列为标准正态分布随机数，"B"列为日期数组
        df = DataFrame(
            {"A": np.random.default_rng(2).standard_normal(len(rng)), "B": dates}
        )
        # 断言"B"列的数据类型是否为np.dtype("M8[ns]")，即datetime64类型
        assert np.issubdtype(df["B"].dtype, np.dtype("M8[ns]"))

    # 测试DataFrame类构造函数是否能正确推断多级索引
    def test_dataframe_constructor_infer_multiindex(self):
        # 定义包含索引值的列表
        index_lists = [["a", "a", "b", "b"], ["x", "y", "x", "y"]]

        # 创建一个DataFrame对象，数据为标准正态分布随机数，使用多级索引
        multi = DataFrame(
            np.random.default_rng(2).standard_normal((4, 4)),
            index=[np.array(x) for x in index_lists],
        )
        # 断言索引是否为MultiIndex类型
        assert isinstance(multi.index, MultiIndex)
        # 断言列不是MultiIndex类型
        assert not isinstance(multi.columns, MultiIndex)

        # 创建一个DataFrame对象，数据为标准正态分布随机数，列名为索引值列表
        multi = DataFrame(
            np.random.default_rng(2).standard_normal((4, 4)), columns=index_lists
        )
        # 断言列是否为MultiIndex类型
        assert isinstance(multi.columns, MultiIndex)

    # 使用pytest的参数化装饰器，测试构造函数处理列表和字符串时的行为
    @pytest.mark.parametrize(
        "input_vals",
        [
            ([1, 2]),  # 整数列表
            (["1", "2"]),  # 字符串列表
            (list(date_range("1/1/2011", periods=2, freq="h"))),  # 时间范围对象列表
            (list(date_range("1/1/2011", periods=2, freq="h", tz="US/Eastern"))),  # 含时区的时间范围对象列表
            ([Interval(left=0, right=5)]),  # 区间对象列表
        ],
    )
    def test_constructor_list_str(self, input_vals, string_dtype):
        # GH#16605
        # 确保当dtype为str、'str'或'U'时，数据元素被转换为字符串

        # 使用给定的dtype创建DataFrame对象
        result = DataFrame({"A": input_vals}, dtype=string_dtype)
        # 创建期望的DataFrame对象，使用astype将"A"列转换为给定的字符串dtype
        expected = DataFrame({"A": input_vals}).astype({"A": string_dtype})
        # 断言两个DataFrame对象是否相等
        tm.assert_frame_equal(result, expected)

    # 测试构造函数处理含有缺失值的列表和字符串时的行为
    def test_constructor_list_str_na(self, string_dtype):
        # 创建含有缺失值的DataFrame对象，使用给定的字符串dtype
        result = DataFrame({"A": [1.0, 2.0, None]}, dtype=string_dtype)
        # 创建期望的DataFrame对象，列"A"的dtype为object，含有与输入相同的缺失值
        expected = DataFrame({"A": ["1.0", "2.0", None]}, dtype=object)
        # 断言两个DataFrame对象是否相等
        tm.assert_frame_equal(result, expected)

    # 使用pytest的参数化装饰器，测试字典数据结构在不复制的情况下的行为
    @pytest.mark.parametrize("copy", [False, True])
    def test_dict_nocopy(
        self,
        copy,
        any_numeric_ea_dtype,
        any_numpy_dtype,
        ```
        ):
            # 创建一个 numpy 数组 `a`，其中包含 [1, 2]，数据类型为 `any_numpy_dtype`
            a = np.array([1, 2], dtype=any_numpy_dtype)
            # 创建一个 numpy 数组 `b`，其中包含 [3, 4]，数据类型为 `any_numpy_dtype`
            b = np.array([3, 4], dtype=any_numpy_dtype)
            # 如果 `b` 的数据类型是字符串或Unicode，跳过当前测试并显示相应的消息
            if b.dtype.kind in ["S", "U"]:
                pytest.skip(f"{b.dtype} get cast, making the checks below more cumbersome")

            # 创建一个 pandas 数组 `c`，其中包含 [1, 2]，数据类型为 `any_numeric_ea_dtype`
            c = pd.array([1, 2], dtype=any_numeric_ea_dtype)
            # 复制 `c` 数组以备份
            c_orig = c.copy()
            # 使用 `a`, `b`, `c` 创建一个 DataFrame `df`，并根据 `copy` 参数决定是否复制数据
            df = DataFrame({"a": a, "b": b, "c": c}, copy=copy)

            # 定义一个函数 `get_base`，用于获取对象的基础数据
            def get_base(obj):
                if isinstance(obj, np.ndarray):
                    return obj.base
                elif isinstance(obj.dtype, np.dtype):
                    # 例如 DatetimeArray, TimedeltaArray
                    return obj._ndarray.base
                else:
                    raise TypeError

            # 定义一个函数 `check_views`，用于检查 DataFrame 的视图是否保持一致性
            def check_views(c_only: bool = False):
                # 检查 `df["c"]` 后面的数据是否仍然是 `c`
                assert sum(x.values is c for x in df._mgr.blocks) == 1
                if c_only:
                    return

                # 检查 `df._mgr.blocks` 中是否有一个数组的基础数据是 `a`
                assert (
                    sum(
                        get_base(x.values) is a
                        for x in df._mgr.blocks
                        if isinstance(x.values.dtype, np.dtype)
                    )
                    == 1
                )
                # 检查 `df._mgr.blocks` 中是否有一个数组的基础数据是 `b`
                assert (
                    sum(
                        get_base(x.values) is b
                        for x in df._mgr.blocks
                        if isinstance(x.values.dtype, np.dtype)
                    )
                    == 1
                )

            # 如果 `copy` 是 False，则检查视图保持一致性
            if not copy:
                check_views()

            # 如果第一列的数据类型不是 "fciuO"，则设置 `warn` 为 `FutureWarning`
            if lib.is_np_dtype(df.dtypes.iloc[0], "fciuO"):
                warn = None
            else:
                warn = FutureWarning

            # 使用 `assert_produces_warning` 上下文管理器检查设置 iloc 后是否会产生警告
            with tm.assert_produces_warning(warn, match="incompatible dtype"):
                df.iloc[0, 0] = 0
                df.iloc[0, 1] = 0

            # 如果 `copy` 是 False，则再次检查视图保持一致性
            if not copy:
                check_views(True)

            # 设置 df.iloc[:, 2] 的值为 [45, 46]，数据类型与 `c` 相同
            df.iloc[:, 2] = pd.array([45, 46], dtype=c.dtype)
            # 断言 df 第三列的数据类型与 `c` 相同
            assert df.dtypes.iloc[2] == c.dtype

            # 如果 `copy` 是 True，则根据数据类型的不同进行断言
            if copy:
                if a.dtype.kind == "M":
                    assert a[0] == a.dtype.type(1, "ns")
                    assert b[0] == b.dtype.type(3, "ns")
                else:
                    assert a[0] == a.dtype.type(1)
                    assert b[0] == b.dtype.type(3)
                # FIXME(GH#35417): 在 GH#35417 之后启用
                assert c[0] == c_orig[0]  # 即 df.iloc[0, 2]=45 未更新 `c`
    # 测试从字典构建 Series，Series 包含扩展数据类型时，默认情况下 copy=True 也应该适用
    def test_construct_from_dict_ea_series(self):
        # 创建包含 Int64 数据类型的 Series
        ser = Series([1, 2, 3], dtype="Int64")
        # 从 Series 创建 DataFrame
        df = DataFrame({"a": ser})
        # 检查 Series 和 DataFrame 的数据是否共享内存
        assert not np.shares_memory(ser.values._data, df["a"].values._data)

    # 测试从带有名称的 Series 创建 DataFrame，并指定列名
    def test_from_series_with_name_with_columns(self):
        # 创建带有名称为 "foo" 的 Series，并指定列名为 "bar"
        result = DataFrame(Series(1, name="foo"), columns=["bar"])
        # 创建预期的 DataFrame，只包含列名 "bar"
        expected = DataFrame(columns=["bar"])
        # 检查两个 DataFrame 是否相等
        tm.assert_frame_equal(result, expected)

    # 测试从嵌套列表创建 DataFrame，指定列名为多级索引
    def test_nested_list_columns(self):
        # 创建嵌套列表的 DataFrame，指定多级列名
        result = DataFrame(
            [[1, 2, 3], [4, 5, 6]], columns=[["A", "A", "A"], ["a", "b", "c"]]
        )
        # 创建预期的 DataFrame，使用 MultiIndex 作为列名
        expected = DataFrame(
            [[1, 2, 3], [4, 5, 6]],
            columns=MultiIndex.from_tuples([("A", "a"), ("A", "b"), ("A", "c")]),
        )
        # 检查两个 DataFrame 是否相等
        tm.assert_frame_equal(result, expected)

    # 测试从包含 Period 或 Interval 的二维对象数组创建 DataFrame
    def test_from_2d_object_array_of_periods_or_intervals(self):
        # 创建 Period 对象
        pi = pd.period_range("2016-04-05", periods=3)
        # 转换为对象数组，并重塑为二维数组
        data = pi._data.astype(object).reshape(1, -1)
        # 从二维数组创建 DataFrame
        df = DataFrame(data)
        # 断言 DataFrame 的形状和数据类型
        assert df.shape == (1, 3)
        assert (df.dtypes == pi.dtype).all()
        assert (df == pi).all().all()

        # 创建 IntervalIndex 对象
        ii = pd.IntervalIndex.from_breaks([3, 4, 5, 6])
        # 转换为对象数组，并重塑为二维数组
        data2 = ii._data.astype(object).reshape(1, -1)
        # 从二维数组创建 DataFrame
        df2 = DataFrame(data2)
        # 断言 DataFrame 的形状和数据类型
        assert df2.shape == (1, 3)
        assert (df2.dtypes == ii.dtype).all()
        assert (df2 == ii).all().all()

        # 创建混合数据
        data3 = np.r_[data, data2, data, data2].T
        # 从混合数据创建 DataFrame
        df3 = DataFrame(data3)
        # 创建预期的 DataFrame
        expected = DataFrame({0: pi, 1: ii, 2: pi, 3: ii})
        # 检查两个 DataFrame 是否相等
        tm.assert_frame_equal(df3, expected)

    # 参数化测试，测试从二维数组创建 DataFrame 时出现错误的情况
    @pytest.mark.parametrize(
        "col_a, col_b",
        [
            ([[1], [2]], np.array([[1], [2]])),
            (np.array([[1], [2]]), [[1], [2]]),
            (np.array([[1], [2]]), np.array([[1], [2]])),
        ],
    )
    def test_error_from_2darray(self, col_a, col_b):
        # 断言创建 DataFrame 时出现错误
        msg = "Per-column arrays must each be 1-dimensional"
        with pytest.raises(ValueError, match=msg):
            DataFrame({"a": col_a, "b": col_b})

    # 测试从包含缺失值的字典创建 DataFrame，设置 copy=False
    def test_from_dict_with_missing_copy_false(self):
        # 创建 DataFrame，指定索引、列名和 copy=False
        df = DataFrame(index=[1, 2, 3], columns=["a", "b", "c"], copy=False)
        # 检查列 "a" 和 "b" 的数据是否共享内存
        assert not np.shares_memory(df["a"]._values, df["b"]._values)

        # 修改 DataFrame 中的值
        df.iloc[0, 0] = 0
        # 创建预期的 DataFrame
        expected = DataFrame(
            {
                "a": [0, np.nan, np.nan],
                "b": [np.nan, np.nan, np.nan],
                "c": [np.nan, np.nan, np.nan],
            },
            index=[1, 2, 3],
            dtype=object,
        )
        # 检查两个 DataFrame 是否相等
        tm.assert_frame_equal(df, expected)
    def test_construction_empty_array_multi_column_raises(self):
        # 标记为 GH#46822 的 GitHub 问题
        msg = r"Shape of passed values is \(0, 1\), indices imply \(0, 2\)"
        # 使用 pytest 来检查是否引发 ValueError，并匹配特定的错误消息
        with pytest.raises(ValueError, match=msg):
            # 创建一个空的 NumPy 数组，并尝试用其作为数据构造 DataFrame，指定列名为 ["a", "b"]
            DataFrame(data=np.array([]), columns=["a", "b"])

    def test_construct_with_strings_and_none(self):
        # 标记为 GH#32218 的 GitHub 问题
        # 使用字符串和 None 来构造 DataFrame，指定列名为 ["a"]，数据类型为字符串
        df = DataFrame(["1", "2", None], columns=["a"], dtype="str")
        # 创建一个预期的 DataFrame，包含一个列 "a"，其值为 ["1", "2", None]，数据类型为字符串
        expected = DataFrame({"a": ["1", "2", None]}, dtype="str")
        # 使用测试工具（tm.assert_frame_equal）比较实际的 DataFrame 和预期的 DataFrame 是否相等
        tm.assert_frame_equal(df, expected)

    def test_frame_string_inference(self):
        # 标记为 GH#54430 的 GitHub 问题
        # 如果缺少 pyarrow 库，则跳过这个测试
        pytest.importorskip("pyarrow")
        # 设置数据类型为 "string[pyarrow_numpy]"
        dtype = "string[pyarrow_numpy]"
        # 创建一个预期的 DataFrame，包含一个列 "a"，其值为 ["a", "b"]，数据类型为 dtype，并指定列索引为 dtype 类型的索引
        expected = DataFrame(
            {"a": ["a", "b"]}, dtype=dtype, columns=Index(["a"], dtype=dtype)
        )
        # 使用 future.infer_string 选项创建一个 DataFrame，包含一个列 "a"，其值为 ["a", "b"]
        with pd.option_context("future.infer_string", True):
            df = DataFrame({"a": ["a", "b"]})
        # 使用测试工具比较实际的 DataFrame 和预期的 DataFrame 是否相等
        tm.assert_frame_equal(df, expected)

        # 创建一个预期的 DataFrame，包含一个列 "a"，其值为 ["a", "b"]，数据类型为 dtype，列索引为 dtype 类型的索引，行索引为 dtype 类型的索引
        expected = DataFrame(
            {"a": ["a", "b"]},
            dtype=dtype,
            columns=Index(["a"], dtype=dtype),
            index=Index(["x", "y"], dtype=dtype),
        )
        # 使用 future.infer_string 选项创建一个 DataFrame，包含一个列 "a"，其值为 ["a", "b"]，行索引为 ["x", "y"]
        with pd.option_context("future.infer_string", True):
            df = DataFrame({"a": ["a", "b"]}, index=["x", "y"])
        # 使用测试工具比较实际的 DataFrame 和预期的 DataFrame 是否相等
        tm.assert_frame_equal(df, expected)

        # 创建一个预期的 DataFrame，包含一个列 "a"，其值为 ["a", 1]，数据类型为 "object"，列索引为 dtype 类型的索引
        expected = DataFrame(
            {"a": ["a", 1]}, dtype="object", columns=Index(["a"], dtype=dtype)
        )
        # 使用 future.infer_string 选项创建一个 DataFrame，包含一个列 "a"，其值为 ["a", 1]
        with pd.option_context("future.infer_string", True):
            df = DataFrame({"a": ["a", 1]})
        # 使用测试工具比较实际的 DataFrame 和预期的 DataFrame 是否相等
        tm.assert_frame_equal(df, expected)

        # 创建一个预期的 DataFrame，包含一个列 "a"，其值为 ["a", "b"]，数据类型为 "object"，列索引为 dtype 类型的索引
        expected = DataFrame(
            {"a": ["a", "b"]}, dtype="object", columns=Index(["a"], dtype=dtype)
        )
        # 使用 future.infer_string 选项创建一个 DataFrame，包含一个列 "a"，其值为 ["a", "b"]，数据类型为 "object"
        with pd.option_context("future.infer_string", True):
            df = DataFrame({"a": ["a", "b"]}, dtype="object")
        # 使用测试工具比较实际的 DataFrame 和预期的 DataFrame 是否相等
        tm.assert_frame_equal(df, expected)

    def test_frame_string_inference_array_string_dtype(self):
        # 标记为 GH#54496 的 GitHub 问题
        # 如果缺少 pyarrow 库，则跳过这个测试
        pytest.importorskip("pyarrow")
        # 设置数据类型为 "string[pyarrow_numpy]"
        dtype = "string[pyarrow_numpy]"
        # 创建一个预期的 DataFrame，包含一个列 "a"，其值为 ["a", "b"]，数据类型为 dtype，并指定列索引为 dtype 类型的索引
        expected = DataFrame(
            {"a": ["a", "b"]}, dtype=dtype, columns=Index(["a"], dtype=dtype)
        )
        # 使用 future.infer_string 选项创建一个 DataFrame，包含一个列 "a"，其值为 ["a", "b"] 的 NumPy 数组
        with pd.option_context("future.infer_string", True):
            df = DataFrame({"a": np.array(["a", "b"])})
        # 使用测试工具比较实际的 DataFrame 和预期的 DataFrame 是否相等
        tm.assert_frame_equal(df, expected)

        # 创建一个预期的 DataFrame，包含两列，第一列为 ["a", "b"]，第二列为 ["c", "d"]，数据类型为 dtype
        expected = DataFrame({0: ["a", "b"], 1: ["c", "d"]}, dtype=dtype)
        # 使用 future.infer_string 选项创建一个 DataFrame，包含一个 NumPy 数组 [["a", "c"], ["b", "d"]]
        with pd.option_context("future.infer_string", True):
            df = DataFrame(np.array([["a", "c"], ["b", "d"]]))
        # 使用测试工具比较实际的 DataFrame 和预期的 DataFrame 是否相等
        tm.assert_frame_equal(df, expected)

        # 创建一个预期的 DataFrame，包含两列 "a" 和 "b"，其值为 [["a", "c"], ["b", "d"]]，数据类型为 dtype
        expected = DataFrame(
            {"a": ["a", "b"], "b": ["c", "d"]},
            dtype=dtype,
            columns=Index(["a", "b"], dtype=dtype),
        )
        # 使用 future.infer_string 选项创建一个 DataFrame，包含一个 NumPy 数组 [["a", "c"], ["b", "d"]]，指定列名为 ["a", "b"]
        df = DataFrame(np.array([["a", "c"], ["b", "d"]]), columns=["a", "b"])
        # 使用测试工具比较实际的 DataFrame 和预期的 DataFrame 是否相等
        tm.assert_frame_equal(df, expected)
    # 定义测试方法，用于测试推断字符串和块维度
    def test_frame_string_inference_block_dim(self):
        # 引入依赖项 pyarrow，如果失败则跳过该测试
        pytest.importorskip("pyarrow")
        # 在上下文中设置选项，允许推断字符串类型
        with pd.option_context("future.infer_string", True):
            # 创建一个包含字符串数组的 DataFrame 对象
            df = DataFrame(np.array([["hello", "goodbye"], ["hello", "Hello"]]))
        # 断言第一个块的维度为 2
        assert df._mgr.blocks[0].ndim == 2

    # 使用参数化测试装饰器，测试 Pandas 对象的推断行为
    @pytest.mark.parametrize("klass", [Series, Index])
    def test_inference_on_pandas_objects(self, klass):
        # 创建一个包含 Timestamp 对象的 Series 或 Index 对象
        obj = klass([Timestamp("2019-12-31")], dtype=object)
        # 使用包含对象的 DataFrame，指定列名为 "a"
        result = DataFrame(obj, columns=["a"])
        # 断言第一列的数据类型为 np.object_
        assert result.dtypes.iloc[0] == np.object_

        # 使用包含对象的 DataFrame，自动推断列名
        result = DataFrame({"a": obj})
        # 断言第一列的数据类型为 np.object_
        assert result.dtypes.iloc[0] == np.object_

    # 测试字典创建的 DataFrame 的列索引类型
    def test_dict_keys_returns_rangeindex(self):
        # 创建一个字典对象并生成 DataFrame，列名为默认的 RangeIndex
        result = DataFrame({0: [1], 1: [2]}).columns
        # 期望的索引类型是 RangeIndex，长度为 2
        expected = RangeIndex(2)
        # 断言 result 和 expected 是相等的索引对象，精确匹配
        tm.assert_index_equal(result, expected, exact=True)

    # 使用参数化测试装饰器，测试不同构造函数的日期时间推断行为
    @pytest.mark.parametrize(
        "cons", [Series, Index, DatetimeIndex, DataFrame, pd.array, pd.to_datetime]
    )
    def test_construction_datetime_resolution_inference(self, cons):
        # 创建一个指定日期时间的 Timestamp 对象
        ts = Timestamp(2999, 1, 1)
        # 将该 Timestamp 对象本地化为指定时区 "US/Pacific"
        ts2 = ts.tz_localize("US/Pacific")

        # 使用给定的构造函数 cons 创建包含 ts 的对象
        obj = cons([ts])
        # 获取对象的数据类型
        res_dtype = tm.get_dtype(obj)
        # 断言结果数据类型为 "M8[us]"，即微秒精度的日期时间

        assert res_dtype == "M8[us]", res_dtype

        # 使用给定的构造函数 cons 创建包含 ts2 的对象
        obj2 = cons([ts2])
        # 获取对象的数据类型
        res_dtype2 = tm.get_dtype(obj2)
        # 断言结果数据类型为 "M8[us, US/Pacific]"，带有指定时区的日期时间

        assert res_dtype2 == "M8[us, US/Pacific]", res_dtype2
class TestDataFrameConstructorIndexInference:
    # 测试方法：从包含重叠月度周期索引的字典序列创建数据帧
    def test_frame_from_dict_of_series_overlapping_monthly_period_indexes(self):
        # 创建日期范围 rng1，从 "1/1/1999" 到 "1/1/2012"，频率为每月
        rng1 = pd.period_range("1/1/1999", "1/1/2012", freq="M")
        # 使用随机数生成器创建序列 s1，长度与 rng1 相同，索引为 rng1
        s1 = Series(np.random.default_rng(2).standard_normal(len(rng1)), rng1)

        # 创建日期范围 rng2，从 "1/1/1980" 到 "12/1/2001"，频率为每月
        rng2 = pd.period_range("1/1/1980", "12/1/2001", freq="M")
        # 使用随机数生成器创建序列 s2，长度与 rng2 相同，索引为 rng2
        s2 = Series(np.random.default_rng(2).standard_normal(len(rng2)), rng2)
        # 创建数据帧 df，包含列 "s1" 和 "s2"，各自对应序列 s1 和 s2
        df = DataFrame({"s1": s1, "s2": s2})

        # 期望的日期范围 exp，从 "1/1/1980" 到 "1/1/2012"，频率为每月
        exp = pd.period_range("1/1/1980", "1/1/2012", freq="M")
        # 断言数据帧 df 的索引与期望的日期范围 exp 相等
        tm.assert_index_equal(df.index, exp)

    # 测试方法：从包含混合时区感知索引的字典创建数据帧
    def test_frame_from_dict_with_mixed_tzaware_indexes(self):
        # GH#44091
        # 创建一个日期范围 dti，从 "2016-01-01" 开始，包含 3 个时间点
        dti = date_range("2016-01-01", periods=3)

        # 创建一个无时区信息的序列 ser1，索引为 dti
        ser1 = Series(range(3), index=dti)
        # 创建一个带有 UTC 时区信息的序列 ser2，索引为 dti，并转换为 UTC 时区
        ser2 = Series(range(3), index=dti.tz_localize("UTC"))
        # 创建一个带有 US/Central 时区信息的序列 ser3，索引为 dti，并转换为 US/Central 时区
        ser3 = Series(range(3), index=dti.tz_localize("US/Central"))
        # 创建一个无时区信息的序列 ser4，索引为 dti
        ser4 = Series(range(3))

        # 创建数据帧 df1，包含列 "A", "B", "C"，各自对应序列 ser2, ser3, ser4
        df1 = DataFrame({"A": ser2, "B": ser3, "C": ser4})
        # 期望的索引对象 exp_index，包含 ser2.index, ser3.index, ser4.index 的组合
        exp_index = Index(
            list(ser2.index) + list(ser3.index) + list(ser4.index), dtype=object
        )
        # 断言数据帧 df1 的索引与期望的索引对象 exp_index 相等
        tm.assert_index_equal(df1.index, exp_index)

        # 创建数据帧 df2，包含列 "A", "C", "B"，各自对应序列 ser2, ser4, ser3
        df2 = DataFrame({"A": ser2, "C": ser4, "B": ser3})
        # 期望的索引对象 exp_index3，包含 ser2.index, ser4.index, ser3.index 的组合
        exp_index3 = Index(
            list(ser2.index) + list(ser4.index) + list(ser3.index), dtype=object
        )
        # 断言数据帧 df2 的索引与期望的索引对象 exp_index3 相等
        tm.assert_index_equal(df2.index, exp_index3)

        # 创建数据帧 df3，包含列 "B", "A", "C"，各自对应序列 ser3, ser2, ser4
        df3 = DataFrame({"B": ser3, "A": ser2, "C": ser4})
        # 期望的索引对象 exp_index3，包含 ser3.index, ser2.index, ser4.index 的组合
        exp_index3 = Index(
            list(ser3.index) + list(ser2.index) + list(ser4.index), dtype=object
        )
        # 断言数据帧 df3 的索引与期望的索引对象 exp_index3 相等
        tm.assert_index_equal(df3.index, exp_index3)

        # 创建数据帧 df4，包含列 "C", "B", "A"，各自对应序列 ser4, ser3, ser2
        df4 = DataFrame({"C": ser4, "B": ser3, "A": ser2})
        # 期望的索引对象 exp_index4，包含 ser4.index, ser3.index, ser2.index 的组合
        exp_index4 = Index(
            list(ser4.index) + list(ser3.index) + list(ser2.index), dtype=object
        )
        # 断言数据帧 df4 的索引与期望的索引对象 exp_index4 相等
        tm.assert_index_equal(df4.index, exp_index4)

        # TODO: 不清楚是否希望引发这些异常（没有现有的测试），
        #  但这是实际行为 2021-12-22
        # 错误消息
        msg = "Cannot join tz-naive with tz-aware DatetimeIndex"
        # 使用 pytest 断言引发 TypeError 异常，并匹配错误消息 msg
        with pytest.raises(TypeError, match=msg):
            DataFrame({"A": ser2, "B": ser3, "C": ser4, "D": ser1})
        with pytest.raises(TypeError, match=msg):
            DataFrame({"A": ser2, "B": ser3, "D": ser1})
        with pytest.raises(TypeError, match=msg):
            DataFrame({"D": ser1, "A": ser2, "B": ser3})

    # 参数化测试：key_val, col_vals, col_type 参数化测试
    @pytest.mark.parametrize(
        "key_val, col_vals, col_type",
        [
            ["3", ["3", "4"], "utf8"],
            [3, [3, 4], "int8"],
        ],
    )
    # 定义一个测试方法，用于测试字典数据到箭头扩展列的转换
    def test_dict_data_arrow_column_expansion(self, key_val, col_vals, col_type):
        # GH 53617
        # 导入 pyarrow 库，如果导入失败则跳过该测试
        pa = pytest.importorskip("pyarrow")
        
        # 使用 ArrowExtensionArray 创建 Arrow 扩展数组 cols，其中包含字典类型的元素
        cols = pd.arrays.ArrowExtensionArray(
            pa.array(col_vals, type=pa.dictionary(pa.int8(), getattr(pa, col_type)()))
        )
        
        # 创建一个 DataFrame，其中包含一个指定键值的列和 Arrow 扩展数组 cols
        result = DataFrame({key_val: [1, 2]}, columns=cols)
        
        # 创建预期的 DataFrame，其中包含与 result 相同的键值列，但列数据为空值
        expected = DataFrame([[1, np.nan], [2, np.nan]], columns=cols)
        
        # 将预期 DataFrame 的第二列设置为对象类型的空值
        expected.isetitem(1, expected.iloc[:, 1].astype(object))
        
        # 使用测试框架中的方法验证 result 和 expected 的内容是否相等
        tm.assert_frame_equal(result, expected)
# 定义测试类 `TestDataFrameConstructorWithDtypeCoercion`，用于测试带类型强制转换的 DataFrame 构造器
class TestDataFrameConstructorWithDtypeCoercion:
    # 定义测试方法 `test_floating_values_integer_dtype`
    def test_floating_values_integer_dtype(self):
        # GH#40110：确保 DataFrame 处理数组式浮点数据和整数类型时与 Series 的行为一致

        # 生成一个 10x5 的标准正态分布随机数组
        arr = np.random.default_rng(2).standard_normal((10, 5))

        # GH#49599：在版本 2.0 中，当尝试将浮点数强制转换为整数时，引发 ValueError 而不再是
        # a) 静默忽略 dtype 并返回浮点数（旧 Series 的行为）或
        # b) 四舍五入（旧 DataFrame 的行为）
        msg = "Trying to coerce float values to integers"
        with pytest.raises(ValueError, match=msg):
            DataFrame(arr, dtype="i8")

        # 创建 DataFrame，将数组元素四舍五入并指定 dtype 为 "i8"
        df = DataFrame(arr.round(), dtype="i8")
        assert (df.dtypes == "i8").all()

        # 当存在 NaN 时，会采用不同的路径并给出不同的警告信息
        arr[0, 0] = np.nan
        msg = r"Cannot convert non-finite values \(NA or inf\) to integer"
        with pytest.raises(IntCastingNaNError, match=msg):
            DataFrame(arr, dtype="i8")
        with pytest.raises(IntCastingNaNError, match=msg):
            Series(arr[0], dtype="i8")
        # 未来（引发异常的）行为与使用 astype 转换得到的行为相匹配
        msg = r"Cannot convert non-finite values \(NA or inf\) to integer"
        with pytest.raises(IntCastingNaNError, match=msg):
            DataFrame(arr).astype("i8")
        with pytest.raises(IntCastingNaNError, match=msg):
            Series(arr[0]).astype("i8")


# 定义测试类 `TestDataFrameConstructorWithDatetimeTZ`
class TestDataFrameConstructorWithDatetimeTZ:
    # 使用 pytest 参数化装饰器，测试方法 `test_construction_preserves_tzaware_dtypes`，参数为时区 `tz`
    @pytest.mark.parametrize("tz", ["US/Eastern", "dateutil/US/Eastern"])
    def test_construction_preserves_tzaware_dtypes(self, tz):
        # GH#7822 后：确保在字典构造时保留时区信息

        # 创建一个日期范围，频率为每周五，从 2011/1/1 到 2012/1/1
        dr = date_range("2011/1/1", "2012/1/1", freq="W-FRI")
        # 将日期范围设为指定时区
        dr_tz = dr.tz_localize(tz)
        # 创建 DataFrame，包含字符串列 "A" 和带时区信息的日期列 "B"，索引为日期范围 `dr`
        df = DataFrame({"A": "foo", "B": dr_tz}, index=dr)
        # 验证列 "B" 的 dtype 是否与预期的带时区的 DatetimeTZDtype 一致
        tz_expected = DatetimeTZDtype("ns", dr_tz.tzinfo)
        assert df["B"].dtype == tz_expected

        # GH#2810（带时区的情况）：处理不同列含有不同时区信息的 DataFrame 构造
        datetimes_naive = [ts.to_pydatetime() for ts in dr]
        datetimes_with_tz = [ts.to_pydatetime() for ts in dr_tz]
        df = DataFrame({"dr": dr})
        df["dr_tz"] = dr_tz
        df["datetimes_naive"] = datetimes_naive
        df["datetimes_with_tz"] = datetimes_with_tz
        result = df.dtypes
        # 预期的结果，包含每列的 dtype 类型
        expected = Series(
            [
                np.dtype("datetime64[ns]"),
                DatetimeTZDtype(tz=tz),
                np.dtype("datetime64[us]"),
                DatetimeTZDtype(tz=tz, unit="us"),
            ],
            index=["dr", "dr_tz", "datetimes_naive", "datetimes_with_tz"],
        )
        # 验证实际结果与预期结果是否相等
        tm.assert_series_equal(result, expected)

    # 使用 pytest 参数化装饰器，测试方法 `test_construction_preserves_tzaware_dtypes`，参数为 `pydt`
    @pytest.mark.parametrize("pydt", [True, False])
    def test_constructor_data_aware_dtype_naive(self, tz_aware_fixture, pydt):
        # GH#25843, GH#41555, GH#33401
        # 获取时区感知的fixture
        tz = tz_aware_fixture
        # 创建一个带时区的Timestamp对象
        ts = Timestamp("2019", tz=tz)
        # 如果pydt为真，将Timestamp对象转换为Python的datetime对象
        if pydt:
            ts = ts.to_pydatetime()

        # 设置错误消息
        msg = (
            "Cannot convert timezone-aware data to timezone-naive dtype. "
            r"Use pd.Series\(values\).dt.tz_localize\(None\) instead."
        )
        # 断言DataFrame构造时引发值错误，并匹配特定错误消息
        with pytest.raises(ValueError, match=msg):
            DataFrame({0: [ts]}, dtype="datetime64[ns]")

        # 设置第二个错误消息
        msg2 = "Cannot unbox tzaware Timestamp to tznaive dtype"
        # 断言DataFrame构造时引发类型错误，并匹配特定错误消息
        with pytest.raises(TypeError, match=msg2):
            DataFrame({0: ts}, index=[0], dtype="datetime64[ns]")

        # 重复使用第一个错误消息进行断言
        with pytest.raises(ValueError, match=msg):
            DataFrame([ts], dtype="datetime64[ns]")

        # 使用包含对象数组的NumPy数组重复使用第一个错误消息进行断言
        with pytest.raises(ValueError, match=msg):
            DataFrame(np.array([ts], dtype=object), dtype="datetime64[ns]")

        # 再次使用第二个错误消息进行断言
        with pytest.raises(TypeError, match=msg2):
            DataFrame(ts, index=[0], columns=[0], dtype="datetime64[ns]")

        # 重复使用第一个错误消息进行断言，DataFrame嵌套Series
        with pytest.raises(ValueError, match=msg):
            DataFrame([Series([ts])], dtype="datetime64[ns]")

        # 重复使用第一个错误消息进行断言，DataFrame嵌套列表
        with pytest.raises(ValueError, match=msg):
            DataFrame([[ts]], columns=[0], dtype="datetime64[ns]")

    def test_from_dict(self):
        # 8260
        # 支持带时区的datetime64

        # 创建一个带有时区的日期索引
        idx = Index(date_range("20130101", periods=3, tz="US/Eastern"), name="foo")
        # 创建一个日期范围
        dr = date_range("20130110", periods=3)

        # 构造DataFrame对象，包含带有时区的datetime64类型数据
        df = DataFrame({"A": idx, "B": dr})
        # 断言列"A"的dtype为"M8[ns, US/Eastern"
        assert df["A"].dtype, "M8[ns, US/Eastern"
        # 断言列"A"的名称为"A"
        assert df["A"].name == "A"
        # 使用测试工具函数断言Series对象的相等性
        tm.assert_series_equal(df["A"], Series(idx, name="A"))
        tm.assert_series_equal(df["B"], Series(dr, name="B"))

    def test_from_index(self):
        # from index
        # 创建一个带有时区的日期索引
        idx2 = date_range("20130101", periods=3, tz="US/Eastern", name="foo")
        # 使用带有时区的日期索引创建DataFrame对象
        df2 = DataFrame(idx2)
        # 使用测试工具函数断言Series对象的相等性
        tm.assert_series_equal(df2["foo"], Series(idx2, name="foo"))
        # 使用带有时区的日期索引创建DataFrame对象，并断言Series对象的相等性
        df2 = DataFrame(Series(idx2))
        tm.assert_series_equal(df2["foo"], Series(idx2, name="foo"))

        # 创建一个没有名称的带有时区的日期索引
        idx2 = date_range("20130101", periods=3, tz="US/Eastern")
        # 使用带有时区的日期索引创建DataFrame对象，并断言Series对象的相等性
        df2 = DataFrame(idx2)
        tm.assert_series_equal(df2[0], Series(idx2, name=0))
        # 使用带有时区的日期索引创建DataFrame对象，并断言Series对象的相等性
        df2 = DataFrame(Series(idx2))
        tm.assert_series_equal(df2[0], Series(idx2, name=0))

    def test_frame_dict_constructor_datetime64_1680(self):
        # 创建一个日期范围
        dr = date_range("1/1/2012", periods=10)
        # 使用日期范围创建Series对象
        s = Series(dr, index=dr)

        # 构造DataFrame对象，包含字符串和日期数据
        DataFrame({"a": "foo", "b": s}, index=dr)
        # 构造DataFrame对象，包含字符串和日期数据，使用Series的值
        DataFrame({"a": "foo", "b": s.values}, index=dr)

    def test_frame_datetime64_mixed_index_ctor_1681(self):
        # 创建一个周五频率的日期范围
        dr = date_range("2011/1/1", "2012/1/1", freq="W-FRI")
        # 使用日期范围创建Series对象
        ts = Series(dr)

        # 构造DataFrame对象，包含字符串和日期数据
        d = DataFrame({"A": "foo", "B": ts}, index=dr)
        # 断言列"B"中的所有值都是缺失值
        assert d["B"].isna().all()
    def test_frame_timeseries_column(self):
        # GH19157
        # 创建一个时间范围，从指定时间开始，频率为分钟，时区为美国东部，共计3个时间点
        dr = date_range(
            start="20130101T10:00:00", periods=3, freq="min", tz="US/Eastern"
        )
        # 创建一个DataFrame，使用时间范围作为数据，列名为"timestamps"
        result = DataFrame(dr, columns=["timestamps"])
        # 创建预期的DataFrame，包含指定时区下的时间戳
        expected = DataFrame(
            {
                "timestamps": [
                    Timestamp("20130101T10:00:00", tz="US/Eastern"),
                    Timestamp("20130101T10:01:00", tz="US/Eastern"),
                    Timestamp("20130101T10:02:00", tz="US/Eastern"),
                ]
            },
            dtype="M8[ns, US/Eastern]",
        )
        # 使用测试工具函数比较结果DataFrame和预期DataFrame
        tm.assert_frame_equal(result, expected)

    def test_nested_dict_construction(self):
        # GH22227
        # 定义列名列表
        columns = ["Nevada", "Ohio"]
        # 创建嵌套字典数据结构
        pop = {
            "Nevada": {2001: 2.4, 2002: 2.9},
            "Ohio": {2000: 1.5, 2001: 1.7, 2002: 3.6},
        }
        # 使用嵌套字典和指定索引、列名创建DataFrame
        result = DataFrame(pop, index=[2001, 2002, 2003], columns=columns)
        # 创建预期的DataFrame，包含指定数据和结构
        expected = DataFrame(
            [(2.4, 1.7), (2.9, 3.6), (np.nan, np.nan)],
            columns=columns,
            index=Index([2001, 2002, 2003]),
        )
        # 使用测试工具函数比较结果DataFrame和预期DataFrame
        tm.assert_frame_equal(result, expected)

    def test_from_tzaware_object_array(self):
        # GH#26825 2D object array of tzaware timestamps should not raise
        # 创建一个时间范围，从指定时间开始，频率为3个时间点，时区为UTC
        dti = date_range("2016-04-05 04:30", periods=3, tz="UTC")
        # 将时间范围数据转换为对象数组，并重新形状为1行多列
        data = dti._data.astype(object).reshape(1, -1)
        # 使用对象数组创建DataFrame
        df = DataFrame(data)
        # 断言DataFrame的形状为(1, 3)
        assert df.shape == (1, 3)
        # 断言DataFrame的数据类型与时间范围的数据类型一致
        assert (df.dtypes == dti.dtype).all()
        # 断言DataFrame的内容与时间范围的内容一致
        assert (df == dti).all().all()

    def test_from_tzaware_mixed_object_array(self):
        # GH#26825
        # 创建包含不同时区的混合对象数组
        arr = np.array(
            [
                [
                    Timestamp("2013-01-01 00:00:00"),
                    Timestamp("2013-01-02 00:00:00"),
                    Timestamp("2013-01-03 00:00:00"),
                ],
                [
                    Timestamp("2013-01-01 00:00:00-0500", tz="US/Eastern"),
                    pd.NaT,
                    Timestamp("2013-01-03 00:00:00-0500", tz="US/Eastern"),
                ],
                [
                    Timestamp("2013-01-01 00:00:00+0100", tz="CET"),
                    pd.NaT,
                    Timestamp("2013-01-03 00:00:00+0100", tz="CET"),
                ],
            ],
            dtype=object,
        ).T
        # 使用对象数组创建DataFrame，指定列名
        res = DataFrame(arr, columns=["A", "B", "C"])
        # 创建预期的数据类型列表
        expected_dtypes = [
            "datetime64[s]",
            "datetime64[s, US/Eastern]",
            "datetime64[s, CET]",
        ]
        # 断言DataFrame的数据类型与预期的数据类型列表一致
        assert (res.dtypes == expected_dtypes).all()

    def test_from_2d_ndarray_with_dtype(self):
        # GH#12513
        # 创建一个二维数组，包含10个元素，形状为(5, 2)
        array_dim2 = np.arange(10).reshape((5, 2))
        # 使用二维数组创建DataFrame，指定数据类型为带时区的日期时间
        df = DataFrame(array_dim2, dtype="datetime64[ns, UTC]")
        # 创建预期的DataFrame，将其数据类型转换为带时区的日期时间
        expected = DataFrame(array_dim2).astype("datetime64[ns, UTC]")
        # 使用测试工具函数比较结果DataFrame和预期DataFrame
        tm.assert_frame_equal(df, expected)

    @pytest.mark.parametrize("typ", [set, frozenset])
    # 测试从集合构建时是否会引发异常
    def test_construction_from_set_raises(self, typ):
        # 引用GitHub上的问题链接
        # 创建包含元素 {1, 2, 3} 的 typ 类型的对象
        values = typ({1, 2, 3})
        # 构造异常消息
        msg = f"'{typ.__name__}' type is unordered"
        # 断言在构建 DataFrame 时会引发 TypeError 异常，异常消息匹配预期的消息
        with pytest.raises(TypeError, match=msg):
            DataFrame({"a": values})

        # 断言在构建 Series 时会引发 TypeError 异常，异常消息匹配预期的消息
        with pytest.raises(TypeError, match=msg):
            Series(values)

    # 测试从 ndarray 构建包含日期时间数据的 DataFrame
    def test_construction_from_ndarray_datetimelike(self):
        # 确保从 2D ndarray 构建时底层数组适当地封装为 DatetimeArray
        arr = np.arange(0, 12, dtype="datetime64[ns]").reshape(4, 3)
        df = DataFrame(arr)
        # 断言所有块都是 DatetimeArray 类型的值
        assert all(isinstance(block.values, DatetimeArray) for block in df._mgr.blocks)

    # 测试从 ndarray 构建时，如果 dtype 与列不匹配会引发 ValueError 异常
    def test_construction_from_ndarray_with_eadtype_mismatched_columns(self):
        arr = np.random.default_rng(2).standard_normal((10, 2))
        dtype = pd.array([2.0]).dtype
        msg = r"len\(arrays\) must match len\(columns\)"
        # 断言在构建 DataFrame 时会引发 ValueError 异常，异常消息匹配预期的消息
        with pytest.raises(ValueError, match=msg):
            DataFrame(arr, columns=["foo"], dtype=dtype)

        arr2 = pd.array([2.0, 3.0, 4.0])
        # 断言在构建 DataFrame 时会引发 ValueError 异常，异常消息匹配预期的消息
        with pytest.raises(ValueError, match=msg):
            DataFrame(arr2, columns=["foo", "bar"])

    # 测试当索引或列是集合时会引发 ValueError 异常
    def test_columns_indexes_raise_on_sets(self):
        data = [[1, 2, 3], [4, 5, 6]]
        # 断言在构建 DataFrame 时会引发 ValueError 异常，异常消息匹配预期的消息
        with pytest.raises(ValueError, match="index cannot be a set"):
            DataFrame(data, index={"a", "b"})
        # 断言在构建 DataFrame 时会引发 ValueError 异常，异常消息匹配预期的消息
        with pytest.raises(ValueError, match="columns cannot be a set"):
            DataFrame(data, columns={"a", "b", "c"})

    # 测试从字典构建 DataFrame，包含列并设置 NaN 值为标量
    def test_from_dict_with_columns_na_scalar(self):
        result = DataFrame({"a": pd.NaT}, columns=["a"], index=range(2))
        expected = DataFrame({"a": Series([pd.NaT, pd.NaT])})
        # 断言构建的 DataFrame 与预期结果相等
        tm.assert_frame_equal(result, expected)

    # TODO: make this not cast to object in pandas 3.0
    @pytest.mark.skipif(
        not np_version_gt2, reason="StringDType only available in numpy 2 and above"
    )
    # 参数化测试，验证从字典中包含字符串数组时的行为
    @pytest.mark.parametrize(
        "data",
        [
            {"a": ["a", "b", "c"], "b": [1.0, 2.0, 3.0], "c": ["d", "e", "f"]},
        ],
    )
    # 测试从 numpy 字符串数组构建 DataFrame 时的行为
    def test_np_string_array_object_cast(self, data):
        from numpy.dtypes import StringDType

        # 使用 StringDType 将数组 'a' 转换为 numpy 字符串类型
        data["a"] = np.array(data["a"], dtype=StringDType())
        # 构建 DataFrame
        res = DataFrame(data)
        # 断言列 'a' 的 dtype 是 np.object_
        assert res["a"].dtype == np.object_
        # 断言列 'a' 的值与原始数据 'a' 相等
        assert (res["a"] == data["a"]).all()
# 定义一个函数 get1，用于从对象中获取第一个元素
def get1(obj):  # TODO: make a helper in tm?
    # 如果对象是 Series 类型，则返回第一个元素
    if isinstance(obj, Series):
        return obj.iloc[0]
    else:
        # 否则，返回 DataFrame 的第一个元素
        return obj.iloc[0, 0]


# 定义一个测试类 TestFromScalar
class TestFromScalar:
    # 定义一个 pytest 的参数化 fixture，用于返回不同的参数（list、dict 或 None）
    @pytest.fixture(params=[list, dict, None])
    def box(self, request):
        return request.param

    # 定义一个 pytest 的 fixture，用于根据参数化结果构造数据框或者系列数据
    @pytest.fixture
    def constructor(self, frame_or_series, box):
        extra = {"index": range(2)}
        # 如果 frame_or_series 是 DataFrame 类型，则设置额外的列名为 ["A"]
        if frame_or_series is DataFrame:
            extra["columns"] = ["A"]

        # 根据 box 的不同取值，返回不同的构造函数
        if box is None:
            return functools.partial(frame_or_series, **extra)
        elif box is dict:
            if frame_or_series is Series:
                # 如果是 Series 类型，则返回一个根据字典构造的函数
                return lambda x, **kwargs: frame_or_series({0: x, 1: x}, **extra, **kwargs)
            else:
                # 否则，返回一个根据字典构造的函数，列名为 "A"
                return lambda x, **kwargs: frame_or_series({"A": x}, **extra, **kwargs)
        elif frame_or_series is Series:
            # 如果是 Series 类型，则返回一个根据列表构造的函数
            return lambda x, **kwargs: frame_or_series([x, x], **extra, **kwargs)
        else:
            # 否则，返回一个根据字典构造的函数，包含 "A" 列
            return lambda x, **kwargs: frame_or_series({"A": [x, x]}, **extra, **kwargs)

    # 参数化测试方法，测试从 NaT（Not a Time）标量创建对象，并验证类型和是否全为空值
    @pytest.mark.parametrize("dtype", ["M8[ns]", "m8[ns]"])
    def test_from_nat_scalar(self, dtype, constructor):
        obj = constructor(pd.NaT, dtype=dtype)
        assert np.all(obj.dtypes == dtype)
        assert np.all(obj.isna())

    # 测试从 Timedelta 标量创建对象，验证是否保留纳秒级别的精度
    def test_from_timedelta_scalar_preserves_nanos(self, constructor):
        td = Timedelta(1)

        obj = constructor(td, dtype="m8[ns]")
        assert get1(obj) == td

    # 测试从 Timestamp 标量创建对象，验证是否保留纳秒级别的精度
    def test_from_timestamp_scalar_preserves_nanos(self, constructor, fixed_now_ts):
        ts = fixed_now_ts + Timedelta(1)

        obj = constructor(ts, dtype="M8[ns]")
        assert get1(obj) == ts

    # 测试从 Timedelta64 标量创建对象，验证返回的对象类型是否为 np.timedelta64
    def test_from_timedelta64_scalar_object(self, constructor):
        td = Timedelta(1)
        td64 = td.to_timedelta64()

        obj = constructor(td64, dtype=object)
        assert isinstance(get1(obj), np.timedelta64)

    # 参数化测试方法，测试从不同标量类型创建日期时间对象时的类型不匹配情况
    @pytest.mark.parametrize("cls", [np.datetime64, np.timedelta64])
    def test_from_scalar_datetimelike_mismatched(self, constructor, cls):
        scalar = cls("NaT", "ns")
        dtype = {np.datetime64: "m8[ns]", np.timedelta64: "M8[ns]"}[cls]

        if cls is np.datetime64:
            msg1 = "Invalid type for timedelta scalar: <class 'numpy.datetime64'>"
        else:
            msg1 = "<class 'numpy.timedelta64'> is not convertible to datetime"
        msg = "|".join(["Cannot cast", msg1])

        # 使用 pytest 的断言来验证是否抛出预期的 TypeError 异常
        with pytest.raises(TypeError, match=msg):
            constructor(scalar, dtype=dtype)

        scalar = cls(4, "ns")
        with pytest.raises(TypeError, match=msg):
            constructor(scalar, dtype=dtype)

    # 参数化测试方法，测试从超出边界的纳秒级日期时间创建对象时的异常情况
    @pytest.mark.parametrize("cls", [datetime, np.datetime64])
    def test_from_out_of_bounds_ns_datetime(
        self, constructor, cls, request, box, frame_or_series
    # scalar that won't fit in nanosecond dt64, but will fit in microsecond
    scalar = datetime(9999, 1, 1)
    exp_dtype = "M8[us]"  # pydatetime objects default to this resolution

    if cls is np.datetime64:
        scalar = np.datetime64(scalar, "D")
        exp_dtype = "M8[s]"  # closest resolution to input
    result = constructor(scalar)

    item = get1(result)
    dtype = tm.get_dtype(result)

    assert type(item) is Timestamp
    assert item.asm8.dtype == exp_dtype
    assert dtype == exp_dtype

@pytest.mark.skip_ubsan
def test_out_of_s_bounds_datetime64(self, constructor):
    # Define a scalar datetime64 that is at its maximum value
    scalar = np.datetime64(np.iinfo(np.int64).max, "D")
    result = constructor(scalar)
    item = get1(result)
    assert type(item) is np.datetime64
    dtype = tm.get_dtype(result)
    assert dtype == object

@pytest.mark.parametrize("cls", [timedelta, np.timedelta64])
def test_from_out_of_bounds_ns_timedelta(
    self, constructor, cls, request, box, frame_or_series
):
    # Define a scalar timedelta that won't fit in nanosecond td64, but fits in microsecond
    if box is list or (frame_or_series is Series and box is dict):
        mark = pytest.mark.xfail(
            reason="TimedeltaArray constructor has been updated to cast td64 "
            "to non-nano, but TimedeltaArray._from_sequence has not",
            strict=True,
        )
        request.applymarker(mark)

    scalar = datetime(9999, 1, 1) - datetime(1970, 1, 1)
    exp_dtype = "m8[us]"  # smallest resolution that fits
    if cls is np.timedelta64:
        scalar = np.timedelta64(scalar, "D")
        exp_dtype = "m8[s]"  # closest resolution to input
    result = constructor(scalar)

    item = get1(result)
    dtype = tm.get_dtype(result)

    assert type(item) is Timedelta
    assert item.asm8.dtype == exp_dtype
    assert dtype == exp_dtype

@pytest.mark.skip_ubsan
@pytest.mark.parametrize("cls", [np.datetime64, np.timedelta64])
def test_out_of_s_bounds_timedelta64(self, constructor, cls):
    # Define a scalar datetime64 or timedelta64 that is at its maximum value
    scalar = cls(np.iinfo(np.int64).max, "D")
    result = constructor(scalar)
    item = get1(result)
    assert type(item) is cls
    dtype = tm.get_dtype(result)
    assert dtype == object

def test_tzaware_data_tznaive_dtype(self, constructor, box, frame_or_series):
    tz = "US/Eastern"
    ts = Timestamp("2019", tz=tz)

    if box is None or (frame_or_series is DataFrame and box is dict):
        msg = "Cannot unbox tzaware Timestamp to tznaive dtype"
        err = TypeError
    else:
        msg = (
            "Cannot convert timezone-aware data to timezone-naive dtype. "
            r"Use pd.Series\(values\).dt.tz_localize\(None\) instead."
        )
        err = ValueError

    # Test that an error is raised when trying to convert timezone-aware Timestamp to tznaive dtype
    with pytest.raises(err, match=msg):
        constructor(ts, dtype="M8[ns]")
# TODO: better location for this test?
# 创建一个测试类 TestAllowNonNano，用于测试非纳秒精度的日期时间索引的行为
class TestAllowNonNano:
    # Until 2.0, we do not preserve non-nano dt64/td64 when passed as ndarray,
    #  but do preserve it when passed as DTA/TDA

    # 定义一个参数化 fixture，用于生成测试数据，参数为 True 和 False
    @pytest.fixture(params=[True, False])
    def as_td(self, request):
        return request.param

    # 定义一个 fixture，根据参数 as_td 返回不同类型的日期时间数组
    @pytest.fixture
    def arr(self, as_td):
        # 创建一个包含 0 到 4 的整数数组，转换为 np.int64 类型，并视图转换为 "M8[s]" 类型
        values = np.arange(5).astype(np.int64).view("M8[s]")
        if as_td:
            # 如果 as_td 为 True，则将数组值转换为时间差数组并返回
            values = values - values[0]
            return TimedeltaArray._simple_new(values, dtype=values.dtype)
        else:
            # 如果 as_td 为 False，则将数组值转换为日期时间数组并返回
            return DatetimeArray._simple_new(values, dtype=values.dtype)

    # 测试索引对象的行为是否符合预期
    def test_index_allow_non_nano(self, arr):
        # 创建一个索引对象并断言其数据类型与输入数组相同
        idx = Index(arr)
        assert idx.dtype == arr.dtype

    # 测试时间增量索引对象的行为是否符合预期
    def test_dti_tdi_allow_non_nano(self, arr, as_td):
        # 根据参数 as_td 创建相应类型的时间索引对象，并断言其数据类型与输入数组相同
        if as_td:
            idx = pd.TimedeltaIndex(arr)
        else:
            idx = DatetimeIndex(arr)
        assert idx.dtype == arr.dtype

    # 测试序列对象的行为是否符合预期
    def test_series_allow_non_nano(self, arr):
        # 创建一个序列对象并断言其数据类型与输入数组相同
        ser = Series(arr)
        assert ser.dtype == arr.dtype

    # 测试数据框对象的行为是否符合预期
    def test_frame_allow_non_nano(self, arr):
        # 创建一个数据框对象并断言第一列的数据类型与输入数组相同
        df = DataFrame(arr)
        assert df.dtypes[0] == arr.dtype

    # 测试从字典创建的数据框对象的行为是否符合预期
    def test_frame_from_dict_allow_non_nano(self, arr):
        # 使用输入数组创建一个数据框对象，并断言第一列的数据类型与输入数组相同
        df = DataFrame({0: arr})
        assert df.dtypes[0] == arr.dtype
```