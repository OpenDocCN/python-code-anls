# `D:\src\scipysrc\pandas\pandas\tests\base\test_conversion.py`

```
import numpy as np  # 导入 NumPy 库
import pytest  # 导入 pytest 库

from pandas.core.dtypes.dtypes import DatetimeTZDtype  # 从 Pandas 库中导入 DatetimeTZDtype 类型

import pandas as pd  # 导入 Pandas 库
from pandas import (  # 从 Pandas 中导入多个模块和类
    CategoricalIndex,
    Series,
    Timedelta,
    Timestamp,
    date_range,
)
import pandas._testing as tm  # 导入 Pandas 测试模块
from pandas.core.arrays import (  # 从 Pandas 核心数组模块中导入多个数组类型
    DatetimeArray,
    IntervalArray,
    NumpyExtensionArray,
    PeriodArray,
    SparseArray,
    TimedeltaArray,
)
from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics  # 导入 Pandas 字符串箭头数组模块


class TestToIterable:
    # test that we convert an iterable to python types
    dtypes = [  # 定义数据类型列表
        ("int8", int),
        ("int16", int),
        ("int32", int),
        ("int64", int),
        ("uint8", int),
        ("uint16", int),
        ("uint32", int),
        ("uint64", int),
        ("float16", float),
        ("float32", float),
        ("float64", float),
        ("datetime64[ns]", Timestamp),
        ("datetime64[ns, US/Eastern]", Timestamp),
        ("timedelta64[ns]", Timedelta),
    ]

    @pytest.mark.parametrize("dtype, rdtype", dtypes)  # 使用 pytest 的参数化装饰器，传入数据类型和期望类型
    @pytest.mark.parametrize(  # 使用 pytest 的参数化装饰器，传入方法列表和标识列表
        "method",
        [
            lambda x: x.tolist(),  # 将对象转换为列表的方法
            lambda x: x.to_list(),  # 将对象转换为列表的方法（Pandas 方法）
            lambda x: list(x),  # 将对象转换为列表的方法
            lambda x: list(x.__iter__()),  # 将对象迭代转换为列表的方法
        ],
        ids=["tolist", "to_list", "list", "iter"],  # 标识每个方法的名称
    )
    def test_iterable(self, index_or_series, method, dtype, rdtype):
        # gh-10904
        # gh-13258
        # coerce iteration to underlying python / pandas types
        typ = index_or_series  # 将输入的索引或序列赋值给变量 typ
        if dtype == "float16" and issubclass(typ, pd.Index):  # 如果数据类型是 float16 并且 typ 是 pd.Index 的子类
            with pytest.raises(NotImplementedError, match="float16 indexes are not "):  # 使用 pytest 来捕获预期的错误信息
                typ([1], dtype=dtype)  # 尝试创建一个 float16 类型的索引
            return  # 如果创建成功则返回
        s = typ([1], dtype=dtype)  # 使用 typ 创建一个包含单个元素的序列 s，指定数据类型为 dtype
        result = method(s)[0]  # 使用指定的方法处理序列 s，并获取结果的第一个元素
        assert isinstance(result, rdtype)  # 断言结果的类型符合预期的 rdtype 类型

    @pytest.mark.parametrize(  # 使用 pytest 的参数化装饰器，传入数据类型、期望类型和对象
        "dtype, rdtype, obj",
        [
            ("object", object, "a"),  # 对象是字符串 "a"，数据类型是 object，期望类型是 object
            ("object", int, 1),  # 对象是整数 1，数据类型是 object，期望类型是 int
            ("category", object, "a"),  # 对象是字符串 "a"，数据类型是 category，期望类型是 object
            ("category", int, 1),  # 对象是整数 1，数据类型是 category，期望类型是 int
        ],
    )
    @pytest.mark.parametrize(  # 使用 pytest 的参数化装饰器，传入方法列表和标识列表
        "method",
        [
            lambda x: x.tolist(),  # 将对象转换为列表的方法
            lambda x: x.to_list(),  # 将对象转换为列表的方法（Pandas 方法）
            lambda x: list(x),  # 将对象转换为列表的方法
            lambda x: list(x.__iter__()),  # 将对象迭代转换为列表的方法
        ],
        ids=["tolist", "to_list", "list", "iter"],  # 标识每个方法的名称
    )
    def test_iterable_object_and_category(
        self, index_or_series, method, dtype, rdtype, obj
    ):
        # gh-10904
        # gh-13258
        # coerce iteration to underlying python / pandas types
        typ = index_or_series  # 将输入的索引或序列赋值给变量 typ
        s = typ([obj], dtype=dtype)  # 使用 typ 创建一个包含单个元素的序列 s，指定数据类型为 dtype
        result = method(s)[0]  # 使用指定的方法处理序列 s，并获取结果的第一个元素
        assert isinstance(result, rdtype)  # 断言结果的类型符合预期的 rdtype 类型

    @pytest.mark.parametrize("dtype, rdtype", dtypes)  # 使用 pytest 的参数化装饰器，传入数据类型和期望类型
    # 定义测试方法，用于验证迭代返回的项是否是正确的包装标量
    def test_iterable_items(self, dtype, rdtype):
        # 标记：gh-13258
        # 测试 items 方法是否正确返回包装后的标量
        # 仅适用于 Series 类型
        s = Series([1], dtype=dtype)
        # 获取迭代器的下一个项，并解包得到结果
        _, result = next(iter(s.items()))
        # 断言结果的类型符合预期的 rdtype
        assert isinstance(result, rdtype)

        _, result = next(iter(s.items()))
        # 再次断言结果的类型符合预期的 rdtype
        assert isinstance(result, rdtype)

    @pytest.mark.parametrize(
        "dtype, rdtype", dtypes + [("object", int), ("category", int)]
    )
    # 定义参数化测试方法，用于验证映射操作的迭代结果
    def test_iterable_map(self, index_or_series, dtype, rdtype):
        # 标记：gh-13236
        # 将迭代结果强制转换为底层的 Python / Pandas 类型
        typ = index_or_series
        # 如果 dtype 是 'float16' 且 typ 是 pd.Index 的子类
        if dtype == "float16" and issubclass(typ, pd.Index):
            # 使用 pytest 检查是否会抛出 NotImplementedError，并匹配特定的错误信息
            with pytest.raises(NotImplementedError, match="float16 indexes are not "):
                typ([1], dtype=dtype)
            return
        # 创建一个包含单个元素的 typ 对象，并指定 dtype
        s = typ([1], dtype=dtype)
        # 对 s 应用 map(type) 方法并获取第一个结果
        result = s.map(type)[0]
        # 如果 rdtype 不是元组，则转换为元组类型
        if not isinstance(rdtype, tuple):
            rdtype = (rdtype,)
        # 断言结果在 rdtype 中
        assert result in rdtype

    @pytest.mark.parametrize(
        "method",
        [
            lambda x: x.tolist(),
            lambda x: x.to_list(),
            lambda x: list(x),
            lambda x: list(x.__iter__()),
        ],
        ids=["tolist", "to_list", "list", "iter"],
    )
    # 定义参数化测试方法，用于测试类别数据和日期时间数据的迭代结果
    def test_categorial_datetimelike(self, method):
        # 创建一个分类索引对象，包含两个时间戳
        i = CategoricalIndex([Timestamp("1999-12-31"), Timestamp("2000-12-31")])

        # 应用给定的方法（method）并获取第一个结果
        result = method(i)[0]
        # 断言结果的类型是 Timestamp
        assert isinstance(result, Timestamp)

    # 定义测试方法，用于验证处理 datetime64 数据的迭代结果
    def test_iter_box_dt64(self, unit):
        # 创建包含两个时间戳的 Series 对象，并将其转换为指定单位的 datetime64 数据
        vals = [Timestamp("2011-01-01"), Timestamp("2011-01-02")]
        ser = Series(vals).dt.as_unit(unit)
        # 断言序列的 dtype 符合预期的 datetime64 单位
        assert ser.dtype == f"datetime64[{unit}]"
        # 遍历序列中的每一项，并逐项进行断言
        for res, exp in zip(ser, vals):
            assert isinstance(res, Timestamp)
            assert res.tz is None
            assert res == exp
            assert res.unit == unit

    # 定义测试方法，用于验证处理带时区的 datetime64 数据的迭代结果
    def test_iter_box_dt64tz(self, unit):
        # 创建包含带时区信息的两个时间戳的 Series 对象，并将其转换为指定单位的 datetime64 数据
        vals = [
            Timestamp("2011-01-01", tz="US/Eastern"),
            Timestamp("2011-01-02", tz="US/Eastern"),
        ]
        ser = Series(vals).dt.as_unit(unit)

        # 断言序列的 dtype 符合预期的 datetime64 单位和时区
        assert ser.dtype == f"datetime64[{unit}, US/Eastern]"
        # 遍历序列中的每一项，并逐项进行断言
        for res, exp in zip(ser, vals):
            assert isinstance(res, Timestamp)
            assert res.tz == exp.tz
            assert res == exp
            assert res.unit == unit

    # 定义测试方法，用于验证处理 timedelta64 数据的迭代结果
    def test_iter_box_timedelta64(self, unit):
        # 创建包含两个 timedelta 的 Series 对象，并将其转换为指定单位的 timedelta64 数据
        vals = [Timedelta("1 days"), Timedelta("2 days")]
        ser = Series(vals).dt.as_unit(unit)
        # 断言序列的 dtype 符合预期的 timedelta64 单位
        assert ser.dtype == f"timedelta64[{unit}]"
        # 遍历序列中的每一项，并逐项进行断言
        for res, exp in zip(ser, vals):
            assert isinstance(res, Timedelta)
            assert res == exp
            assert res.unit == unit
    # 定义一个测试方法，用于测试迭代器在处理时间段时的行为
    def test_iter_box_period(self):
        # 创建两个时间段对象并放入列表中
        vals = [pd.Period("2011-01-01", freq="M"), pd.Period("2011-01-02", freq="M")]
        # 用时间段对象列表创建一个 Pandas Series
        s = Series(vals)
        # 断言 Series 的数据类型为 "Period[M]"
        assert s.dtype == "Period[M]"
        # 使用 zip 函数同时迭代 s 中的元素 res 和 vals 中的元素 exp
        for res, exp in zip(s, vals):
            # 断言 res 是一个 pd.Period 对象
            assert isinstance(res, pd.Period)
            # 断言 res 的频率为 "ME"（这里是一个故意制造错误的期望，期望会失败）
            assert res.freq == "ME"
            # 断言 res 等于 exp，即与期望的时间段对象相同
            assert res == exp
@pytest.mark.parametrize(
    "arr, expected_type, dtype",
    [
        # 测试用例1: 整数类型的 NumPy 数组
        (np.array([0, 1], dtype=np.int64), np.ndarray, "int64"),
        # 测试用例2: 对象类型的 NumPy 数组
        (np.array(["a", "b"]), np.ndarray, "object"),
        # 测试用例3: Pandas 的分类数据
        (pd.Categorical(["a", "b"]), pd.Categorical, "category"),
        # 测试用例4: 带时区的日期时间索引
        (
            pd.DatetimeIndex(["2017", "2018"], tz="US/Central"),
            DatetimeArray,
            "datetime64[ns, US/Central]",
        ),
        # 测试用例5: 周期索引
        (
            pd.PeriodIndex([2018, 2019], freq="Y"),
            PeriodArray,
            pd.core.dtypes.dtypes.PeriodDtype("Y-DEC"),
        ),
        # 测试用例6: 区间索引
        (pd.IntervalIndex.from_breaks([0, 1, 2]), IntervalArray, "interval"),
        # 测试用例7: 日期时间索引
        (
            pd.DatetimeIndex(["2017", "2018"]),
            DatetimeArray,
            "datetime64[ns]",
        ),
        # 测试用例8: 时间增量索引
        (
            pd.TimedeltaIndex([10**10]),
            TimedeltaArray,
            "m8[ns]",
        ),
    ],
)
def test_values_consistent(arr, expected_type, dtype, using_infer_string):
    if using_infer_string and dtype == "object":
        expected_type = ArrowStringArrayNumpySemantics
    # 创建 Series 对象并获取其内部值
    l_values = Series(arr)._values
    # 创建索引对象并获取其内部值
    r_values = pd.Index(arr)._values
    # 断言 Series 的值类型与期望类型一致
    assert type(l_values) is expected_type
    # 断言 Series 的值类型与索引的值类型一致
    assert type(l_values) is type(r_values)
    # 使用 Pandas 提供的测试工具断言 l_values 和 r_values 相等
    tm.assert_equal(l_values, r_values)


@pytest.mark.parametrize("arr", [np.array([1, 2, 3])])
def test_numpy_array(arr):
    # 创建 Series 对象
    ser = Series(arr)
    # 获取 Series 的扩展数组
    result = ser.array
    # 创建预期的 NumPy 扩展数组
    expected = NumpyExtensionArray(arr)
    # 使用 Pandas 提供的测试工具断言两者相等
    tm.assert_extension_array_equal(result, expected)


def test_numpy_array_all_dtypes(any_numpy_dtype):
    # 创建指定数据类型的 Series 对象
    ser = Series(dtype=any_numpy_dtype)
    # 获取 Series 的数组表示
    result = ser.array
    # 根据数据类型种类进行断言
    if np.dtype(any_numpy_dtype).kind == "M":
        assert isinstance(result, DatetimeArray)
    elif np.dtype(any_numpy_dtype).kind == "m":
        assert isinstance(result, TimedeltaArray)
    else:
        assert isinstance(result, NumpyExtensionArray)


@pytest.mark.parametrize(
    "arr, attr",
    [
        # 测试用例1: Pandas 分类数据的 _codes 属性
        (pd.Categorical(["a", "b"]), "_codes"),
        # 测试用例2: 周期数组的 _ndarray 属性
        (PeriodArray._from_sequence(["2000", "2001"], dtype="period[D]"), "_ndarray"),
        # 测试用例3: Pandas 整数数组的 _data 属性
        (pd.array([0, np.nan], dtype="Int64"), "_data"),
        # 测试用例4: 区间数组的 _left 属性
        (IntervalArray.from_breaks([0, 1]), "_left"),
        # 测试用例5: 稀疏数组的 _sparse_values 属性
        (SparseArray([0, 1]), "_sparse_values"),
        # 测试用例6: 从序列创建的日期时间数组的 _ndarray 属性
        (
            DatetimeArray._from_sequence(np.array([1, 2], dtype="datetime64[ns]")),
            "_ndarray",
        ),
        # 测试用例7: 带时区信息的日期时间数组的 _ndarray 属性
        (
            DatetimeArray._from_sequence(
                np.array(
                    ["2000-01-01T12:00:00", "2000-01-02T12:00:00"], dtype="M8[ns]"
                ),
                dtype=DatetimeTZDtype(tz="US/Central"),
            ),
            "_ndarray",
        ),
    ],
)
def test_array(arr, attr, index_or_series):
    # 获取 index_or_series 对象
    box = index_or_series
    # 使用 box 函数创建对象并获取其数组表示
    result = box(arr, copy=False).array
    # 如果存在属性，获取属性值并进行断言
    if attr:
        arr = getattr(arr, attr)
        result = getattr(result, attr)
    # 断言结果与原始数组相等
    assert result is arr


def test_array_multiindex_raises():
    # 创建多级索引对象
    idx = pd.MultiIndex.from_product([["A"], ["a", "b"]])
    # 定义错误信息字符串，用于匹配 pytest 抛出的 ValueError 异常
    msg = "MultiIndex has no single backing array"
    # 使用 pytest 提供的上下文管理器，期望捕获到 ValueError 异常，并检查其消息是否与 msg 变量匹配
    with pytest.raises(ValueError, match=msg):
        # 访问 MultiIndex 对象的 array 属性，预期会触发 ValueError 异常
        idx.array
# 使用 pytest 的 mark.parametrize 装饰器定义参数化测试，测试输入 arr 和期望输出 expected
@pytest.mark.parametrize(
    "arr, expected",
    [
        # 测试 np.array([1, 2], dtype=np.int64) 的情况
        (np.array([1, 2], dtype=np.int64), np.array([1, 2], dtype=np.int64)),
        # 测试 pd.Categorical(["a", "b"]) 的情况
        (pd.Categorical(["a", "b"]), np.array(["a", "b"], dtype=object)),
        # 测试 pd.core.arrays.period_array(["2000", "2001"], freq="D") 的情况
        (
            pd.core.arrays.period_array(["2000", "2001"], freq="D"),
            np.array([pd.Period("2000", freq="D"), pd.Period("2001", freq="D")]),
        ),
        # 测试 pd.array([0, np.nan], dtype="Int64") 的情况
        (pd.array([0, np.nan], dtype="Int64"), np.array([0, np.nan])),
        # 测试 IntervalArray.from_breaks([0, 1, 2]) 的情况
        (
            IntervalArray.from_breaks([0, 1, 2]),
            np.array([pd.Interval(0, 1), pd.Interval(1, 2)], dtype=object),
        ),
        # 测试 SparseArray([0, 1]) 的情况
        (SparseArray([0, 1]), np.array([0, 1], dtype=np.int64)),
        # 测试 DatetimeArray._from_sequence(np.array(["2000", "2001"], dtype="M8[ns]")) 的情况
        (
            DatetimeArray._from_sequence(np.array(["2000", "2001"], dtype="M8[ns]")),
            np.array(["2000", "2001"], dtype="M8[ns]"),
        ),
        # 测试 DatetimeArray._from_sequence(...) 的情况，处理带时区的日期时间
        (
            DatetimeArray._from_sequence(
                np.array(["2000-01-01T06:00:00", "2000-01-02T06:00:00"], dtype="M8[ns]")
            )
            .tz_localize("UTC")
            .tz_convert("US/Central"),
            np.array(
                [
                    Timestamp("2000-01-01", tz="US/Central"),
                    Timestamp("2000-01-02", tz="US/Central"),
                ]
            ),
        ),
        # 测试 TimedeltaArray._from_sequence(...) 的情况，处理时间差
        (
            TimedeltaArray._from_sequence(
                np.array([0, 3600000000000], dtype="i8").view("m8[ns]")
            ),
            np.array([0, 3600000000000], dtype="m8[ns]"),
        ),
        # 测试 pd.Categorical(date_range("2016-01-01", periods=2, tz="US/Pacific")) 的情况
        (
            pd.Categorical(date_range("2016-01-01", periods=2, tz="US/Pacific")),
            np.array(
                [
                    Timestamp("2016-01-01", tz="US/Pacific"),
                    Timestamp("2016-01-02", tz="US/Pacific"),
                ]
            ),
        ),
    ],
)
# 定义测试函数 test_to_numpy，测试转换为 numpy 数组的功能
def test_to_numpy(arr, expected, index_or_series_or_array, request):
    # 获取测试参数 box，并忽略产生的警告
    box = index_or_series_or_array

    with tm.assert_produces_warning(None):
        # 将 arr 转换为 box 对象
        thing = box(arr)

    # 测试调用 to_numpy 方法得到的结果，并断言其与期望结果 expected 相等
    result = thing.to_numpy()
    tm.assert_numpy_array_equal(result, expected)

    # 将 thing 转换为 numpy 数组，并再次断言其与期望结果 expected 相等
    result = np.asarray(thing)
    tm.assert_numpy_array_equal(result, expected)


# 使用 pytest 的 mark.parametrize 装饰器定义参数化测试，测试 as_series 和 arr 的组合
@pytest.mark.parametrize("as_series", [True, False])
@pytest.mark.parametrize(
    "arr", [np.array([1, 2, 3], dtype="int64"), np.array(["a", "b", "c"], dtype=object)]
)
# 定义测试函数 test_to_numpy_copy，测试在不同条件下的拷贝行为
def test_to_numpy_copy(arr, as_series, using_infer_string):
    # 创建 pd.Index 对象 obj，根据 as_series 决定是否转换为 Series 对象
    obj = pd.Index(arr, copy=False)
    if as_series:
        obj = Series(obj.values, copy=False)

    # 默认情况下不进行拷贝，测试调用 to_numpy 方法得到的结果
    result = obj.to_numpy()
    # 根据 using_infer_string 和 arr 的 dtype 来断言内存共享行为
    if using_infer_string and arr.dtype == object:
        assert np.shares_memory(arr, result) is False
    else:
        assert np.shares_memory(arr, result) is True

    # 测试调用 to_numpy(copy=False) 方法得到的结果
    result = obj.to_numpy(copy=False)
    # 根据 using_infer_string 和 arr 的 dtype 来断言内存共享行为
    if using_infer_string and arr.dtype == object:
        assert np.shares_memory(arr, result) is False
    # 如果条件不满足，则断言 arr 和 result 共享内存
    else:
        assert np.shares_memory(arr, result) is True

    # 使用 copy=True 选项将 obj 转换为 NumPy 数组，并将结果赋给 result
    result = obj.to_numpy(copy=True)
    # 断言 arr 和 result 不共享内存
    assert np.shares_memory(arr, result) is False
@pytest.mark.parametrize("as_series", [True, False])
def test_to_numpy_dtype(as_series):
    tz = "US/Eastern"  # 设置时区为美国东部
    obj = pd.DatetimeIndex(["2000", "2001"], tz=tz)  # 创建包含两个日期的DatetimeIndex对象，带有时区信息
    if as_series:
        obj = Series(obj)  # 如果as_series为True，则将obj转换为Series对象

    # 默认情况下保留时区信息
    result = obj.to_numpy()  # 转换为numpy数组
    expected = np.array(
        [Timestamp("2000", tz=tz), Timestamp("2001", tz=tz)], dtype=object
    )
    tm.assert_numpy_array_equal(result, expected)  # 断言结果与预期相等

    result = obj.to_numpy(dtype="object")  # 指定dtype为"object"进行转换
    tm.assert_numpy_array_equal(result, expected)  # 断言结果与预期相等

    result = obj.to_numpy(dtype="M8[ns]")  # 指定dtype为"M8[ns]"进行转换
    expected = np.array(["2000-01-01T05", "2001-01-01T05"], dtype="M8[ns]")
    tm.assert_numpy_array_equal(result, expected)  # 断言结果与预期相等


@pytest.mark.parametrize(
    "values, dtype, na_value, expected",
    [
        ([1, 2, None], "float64", 0, [1.0, 2.0, 0.0]),  # 测试值、dtype为"float64"、na_value为0的情况
        (
            [Timestamp("2000"), Timestamp("2000"), pd.NaT],  # 测试包含Timestamp和NaT的情况
            None,  # dtype为None
            Timestamp("2000"),  # na_value为Timestamp("2000")
            [np.datetime64("2000-01-01T00:00:00", "s")] * 3,  # 预期结果为3个相同的datetime64对象
        ),
    ],
)
def test_to_numpy_na_value_numpy_dtype(
    index_or_series, values, dtype, na_value, expected
):
    obj = index_or_series(values)  # 使用index_or_series函数创建对象
    result = obj.to_numpy(dtype=dtype, na_value=na_value)  # 转换为numpy数组，指定dtype和na_value
    expected = np.array(expected)  # 创建预期的numpy数组
    tm.assert_numpy_array_equal(result, expected)  # 断言结果与预期相等


@pytest.mark.parametrize(
    "data, multiindex, dtype, na_value, expected",
    [
        (
            [1, 2, None, 4],  # 测试数据包含None的情况
            [(0, "a"), (0, "b"), (1, "b"), (1, "c")],  # 测试多索引为元组的情况
            float,  # dtype为float
            None,  # na_value为None
            [1.0, 2.0, np.nan, 4.0],  # 预期结果包含NaN
        ),
        (
            [1, 2, None, 4],  # 同上，但na_value为np.nan
            [(0, "a"), (0, "b"), (1, "b"), (1, "c")],
            float,
            np.nan,
            [1.0, 2.0, np.nan, 4.0],
        ),
        (
            [1.0, 2.0, np.nan, 4.0],  # 测试浮点数和NaN的情况
            [("a", 0), ("a", 1), ("a", 2), ("b", 0)],  # 测试多索引为元组的情况
            int,  # dtype为int
            0,  # na_value为0
            [1, 2, 0, 4],  # 预期结果为整数数组
        ),
        (
            [Timestamp("2000"), Timestamp("2000"), pd.NaT],  # 测试包含Timestamp和NaT的情况
            [(0, Timestamp("2021")), (0, Timestamp("2022")), (1, Timestamp("2000"))],  # 测试多索引为元组的情况
            None,  # dtype为None
            Timestamp("2000"),  # na_value为Timestamp("2000")
            [np.datetime64("2000-01-01T00:00:00", "s")] * 3,  # 预期结果为3个相同的datetime64对象
        ),
    ],
)
def test_to_numpy_multiindex_series_na_value(
    data, multiindex, dtype, na_value, expected
):
    index = pd.MultiIndex.from_tuples(multiindex)  # 创建MultiIndex对象
    series = Series(data, index=index)  # 创建Series对象
    result = series.to_numpy(dtype=dtype, na_value=na_value)  # 转换为numpy数组，指定dtype和na_value
    expected = np.array(expected)  # 创建预期的numpy数组
    tm.assert_numpy_array_equal(result, expected)  # 断言结果与预期相等


def test_to_numpy_kwargs_raises():
    # numpy
    s = Series([1, 2, 3])  # 创建整数Series对象
    msg = r"to_numpy\(\) got an unexpected keyword argument 'foo'"  # 预期的错误消息
    with pytest.raises(TypeError, match=msg):  # 断言调用to_numpy(foo=True)时会抛出TypeError，错误消息匹配msg
        s.to_numpy(foo=True)

    # extension
    s = Series([1, 2, 3], dtype="Int64")  # 创建带有Int64扩展类型的Series对象
    with pytest.raises(TypeError, match=msg):  # 同上，断言调用to_numpy(foo=True)时会抛出TypeError，错误消息匹配msg
        s.to_numpy(foo=True)


@pytest.mark.parametrize(
    "data",
    [
        # 创建包含字典的列表，每个字典有两个键值对
        {"a": [1, 2, 3], "b": [1, 2, None]},
        # 创建包含字典的列表，使用 NumPy 数组作为值
        {"a": np.array([1, 2, 3]), "b": np.array([1, 2, np.nan])},
        # 创建包含字典的列表，使用 Pandas Series 作为值
        {"a": pd.array([1, 2, 3]), "b": pd.array([1, 2, None])},
    ],
@pytest.mark.parametrize("dtype, na_value", [(float, np.nan), (object, None)])
# 使用 pytest 的参数化装饰器，设置测试参数 dtype 和 na_value
def test_to_numpy_dataframe_na_value(data, dtype, na_value):
    # 创建 DataFrame 对象，将传入的数据作为内容
    df = pd.DataFrame(data)
    # 调用 DataFrame 的 to_numpy 方法，将 DataFrame 转换为 NumPy 数组
    result = df.to_numpy(dtype=dtype, na_value=na_value)
    # 创建预期的 NumPy 数组，用于与结果进行比较
    expected = np.array([[1, 1], [2, 2], [3, na_value]], dtype=dtype)
    # 使用测试框架的方法检查两个 NumPy 数组是否相等
    tm.assert_numpy_array_equal(result, expected)


@pytest.mark.parametrize(
    "data, expected_data",
    [
        (
            {"a": pd.array([1, 2, None])},
            [[1.0], [2.0], [np.nan]],
        ),
        (
            {"a": [1, 2, 3], "b": [1, 2, 3]},
            [[1, 1], [2, 2], [3, 3]],
        ),
    ],
)
# 使用 pytest 的参数化装饰器，设置测试参数 data 和 expected_data
def test_to_numpy_dataframe_single_block(data, expected_data):
    # 创建 DataFrame 对象，将传入的数据作为内容
    df = pd.DataFrame(data)
    # 调用 DataFrame 的 to_numpy 方法，将 DataFrame 转换为 NumPy 数组
    result = df.to_numpy(dtype=float, na_value=np.nan)
    # 创建预期的 NumPy 数组，用于与结果进行比较
    expected = np.array(expected_data, dtype=float)
    # 使用测试框架的方法检查两个 NumPy 数组是否相等
    tm.assert_numpy_array_equal(result, expected)


def test_to_numpy_dataframe_single_block_no_mutate():
    # 创建包含 NumPy 数组的 DataFrame 对象
    result = pd.DataFrame(np.array([1.0, 2.0, np.nan]))
    # 创建预期的 DataFrame 对象
    expected = pd.DataFrame(np.array([1.0, 2.0, np.nan]))
    # 调用 DataFrame 的 to_numpy 方法，但没有指定 na_value，不会改变结果
    result.to_numpy(na_value=0.0)
    # 使用测试框架的方法检查两个 DataFrame 是否相等
    tm.assert_frame_equal(result, expected)


class TestAsArray:
    @pytest.mark.parametrize("tz", [None, "US/Central"])
    # 使用 pytest 的参数化装饰器，设置测试参数 tz
    def test_asarray_object_dt64(self, tz):
        # 创建包含时间序列的 Series 对象，设置时区
        ser = Series(date_range("2000", periods=2, tz=tz))

        with tm.assert_produces_warning(None):
            # 调用 np.asarray 将 Series 转换为对象类型的 NumPy 数组，测试无警告
            result = np.asarray(ser, dtype=object)

        # 创建预期的 NumPy 数组，包含时间戳对象
        expected = np.array(
            [Timestamp("2000-01-01", tz=tz), Timestamp("2000-01-02", tz=tz)]
        )
        # 使用测试框架的方法检查两个 NumPy 数组是否相等
        tm.assert_numpy_array_equal(result, expected)

    def test_asarray_tz_naive(self):
        # 创建包含时间序列的 Series 对象，没有设置时区
        ser = Series(date_range("2000", periods=2))
        # 创建预期的 NumPy 数组，包含日期字符串
        expected = np.array(["2000-01-01", "2000-01-02"], dtype="M8[ns]")
        # 调用 np.asarray 将 Series 转换为 NumPy 数组
        result = np.asarray(ser)
        # 使用测试框架的方法检查两个 NumPy 数组是否相等
        tm.assert_numpy_array_equal(result, expected)

    def test_asarray_tz_aware(self):
        # 设置时区
        tz = "US/Central"
        # 创建包含时间序列的 Series 对象，设置时区
        ser = Series(date_range("2000", periods=2, tz=tz))
        # 创建预期的 NumPy 数组，包含带时区的日期字符串
        expected = np.array(["2000-01-01T06", "2000-01-02T06"], dtype="M8[ns]")
        # 调用 np.asarray 将 Series 转换为 datetime64 类型的 NumPy 数组
        result = np.asarray(ser, dtype="datetime64[ns]")
        # 使用测试框架的方法检查两个 NumPy 数组是否相等
        tm.assert_numpy_array_equal(result, expected)

        # 使用旧的 dtype，测试时不产生警告
        result = np.asarray(ser, dtype="M8[ns]")
        # 使用测试框架的方法检查两个 NumPy 数组是否相等
        tm.assert_numpy_array_equal(result, expected)
```