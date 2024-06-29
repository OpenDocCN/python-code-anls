# `D:\src\scipysrc\pandas\pandas\tests\copy_view\test_constructors.py`

```
import numpy as np  # 导入 NumPy 库，用于处理数组和数值计算

import pytest  # 导入 pytest 库，用于编写和运行测试用例

import pandas as pd  # 导入 Pandas 库，用于数据处理和分析
from pandas import (  # 导入 Pandas 的子模块和类
    DataFrame,  # 用于处理带标签的二维数据
    DatetimeIndex,  # 日期时间索引类
    Index,  # 通用索引类
    Period,  # 表示时期的类
    PeriodIndex,  # 时期索引类
    Series,  # 用于处理带标签的一维数据
    Timedelta,  # 表示时间增量的类
    TimedeltaIndex,  # 时间增量索引类
    Timestamp,  # 表示时间戳的类
)

import pandas._testing as tm  # 导入 Pandas 内部测试工具模块
from pandas.tests.copy_view.util import get_array  # 从测试工具中导入获取数组函数

# -----------------------------------------------------------------------------
# Series / DataFrame 构造函数的复制/视图行为测试


@pytest.mark.parametrize("dtype", [None, "int64"])
def test_series_from_series(dtype):
    # 场景：从另一个 Series 对象构造 Series 遵循 CoW 规则：
    # 返回一个新对象，因此不会传播变异
    ser = Series([1, 2, 3], name="name")

    # 默认情况下 copy=False -> 新 Series 是原始 Series 的浅复制/视图
    result = Series(ser, dtype=dtype)

    # 浅复制仍然共享内存
    assert np.shares_memory(get_array(ser), get_array(result))

    assert result._mgr.blocks[0].refs.has_reference()

    # 修改新的 Series 复制不会改变原始 Series
    result.iloc[0] = 0
    assert ser.iloc[0] == 1
    # 修改触发了写时复制 -> 不再共享内存
    assert not np.shares_memory(get_array(ser), get_array(result))

    # 当修改原始 Series 时同样适用
    result = Series(ser, dtype=dtype)

    # 修改原始 Series 不会改变新的 Series
    ser.iloc[0] = 0
    assert result.iloc[0] == 1


def test_series_from_series_with_reindex():
    # 场景：从另一个 Series 构造 Series 并指定可能需要重新索引值的索引
    ser = Series([1, 2, 3], name="name")

    # 传递一个不需要实际重新索引值的索引 -> 如果不使用 CoW，我们得到一个实际的可变视图
    for index in [
        ser.index,
        ser.index.copy(),
        list(ser.index),
        ser.index.rename("idx"),
    ]:
        result = Series(ser, index=index)
        assert np.shares_memory(ser.values, result.values)
        result.iloc[0] = 0
        assert ser.iloc[0] == 1

    # 确保如果需要实际重新索引，则没有引用
    # （修改结果不会触发 CoW）
    result = Series(ser, index=[0, 1, 2, 3])
    assert not np.shares_memory(ser.values, result.values)
    assert not result._mgr.blocks[0].refs.has_reference()


@pytest.mark.parametrize("dtype", [None, "int64"])
@pytest.mark.parametrize("idx", [None, pd.RangeIndex(start=0, stop=3, step=1)])
@pytest.mark.parametrize(
    "arr", [np.array([1, 2, 3], dtype="int64"), pd.array([1, 2, 3], dtype="Int64")]
)
def test_series_from_array(idx, dtype, arr):
    ser = Series(arr, dtype=dtype, index=idx)
    ser_orig = ser.copy()
    data = getattr(arr, "_data", arr)
    assert not np.shares_memory(get_array(ser), data)

    arr[0] = 100
    tm.assert_series_equal(ser, ser_orig)


@pytest.mark.parametrize("copy", [True, False, None])
def test_series_from_array_different_dtype(copy):
    # 测试不同数据类型的 Series 构造
    pass  # 该测试当前没有实现任何具体的逻辑，只是一个占位符
    # 创建一个 NumPy 数组，包含元素 [1, 2, 3]，数据类型为 int64
    arr = np.array([1, 2, 3], dtype="int64")
    # 使用 NumPy 的 Series 函数，将 arr 转换为 Pandas 的 Series 对象，
    # 指定数据类型为 int32，并设置 copy 参数（假设 copy 变量已定义）
    ser = Series(arr, dtype="int32", copy=copy)
    # 断言：验证 get_array(ser) 返回的数组与 arr 不共享内存
    assert not np.shares_memory(get_array(ser), arr)
@pytest.mark.parametrize(
    "idx",
    [
        Index([1, 2]),  # 创建一个整数索引对象
        DatetimeIndex([Timestamp("2019-12-31"), Timestamp("2020-12-31")]),  # 创建一个日期时间索引对象
        PeriodIndex([Period("2019-12-31"), Period("2020-12-31")]),  # 创建一个周期索引对象
        TimedeltaIndex([Timedelta("1 days"), Timedelta("2 days")]),  # 创建一个时间增量索引对象
    ],
)
def test_series_from_index(idx):
    ser = Series(idx)  # 使用索引对象创建一个 Series 对象
    expected = idx.copy(deep=True)  # 深拷贝索引对象作为预期结果
    assert np.shares_memory(get_array(ser), get_array(idx))  # 检查 Series 是否与索引对象共享内存
    assert not ser._mgr._has_no_reference(0)  # 检查 Series 内部管理器是否有引用
    ser.iloc[0] = ser.iloc[1]  # 修改 Series 的第一个元素为第二个元素的值
    tm.assert_index_equal(idx, expected)  # 断言索引对象与预期对象相等


def test_series_from_index_different_dtypes():
    idx = Index([1, 2, 3], dtype="int64")  # 创建一个指定数据类型的整数索引对象
    ser = Series(idx, dtype="int32")  # 使用指定数据类型创建一个 Series 对象
    assert not np.shares_memory(get_array(ser), get_array(idx))  # 检查 Series 是否与索引对象不共享内存
    assert ser._mgr._has_no_reference(0)  # 检查 Series 内部管理器是否没有引用


def test_series_from_block_manager_different_dtype():
    ser = Series([1, 2, 3], dtype="int64")  # 创建一个指定数据类型的 Series 对象
    msg = "Passing a SingleBlockManager to Series"
    with tm.assert_produces_warning(DeprecationWarning, match=msg):  # 检查是否产生特定警告
        ser2 = Series(ser._mgr, dtype="int32")  # 使用不同数据类型创建一个 Series 对象
    assert not np.shares_memory(get_array(ser), get_array(ser2))  # 检查两个 Series 对象是否不共享内存
    assert ser2._mgr._has_no_reference(0)  # 检查新 Series 对象的内部管理器是否没有引用


@pytest.mark.parametrize("use_mgr", [True, False])
@pytest.mark.parametrize("columns", [None, ["a"]])
def test_dataframe_constructor_mgr_or_df(columns, use_mgr):
    df = DataFrame({"a": [1, 2, 3]})  # 创建一个 DataFrame 对象
    df_orig = df.copy()  # 复制原始 DataFrame 对象

    if use_mgr:
        data = df._mgr  # 使用 DataFrame 的内部管理器作为数据
        warn = DeprecationWarning
    else:
        data = df  # 使用 DataFrame 对象作为数据
        warn = None
    msg = "Passing a BlockManager to DataFrame"
    with tm.assert_produces_warning(warn, match=msg, check_stacklevel=False):  # 检查是否产生特定警告
        new_df = DataFrame(data)  # 使用给定数据创建一个新的 DataFrame 对象

    assert np.shares_memory(get_array(df, "a"), get_array(new_df, "a"))  # 检查两个 DataFrame 的列是否共享内存
    new_df.iloc[0] = 100  # 修改新 DataFrame 的第一行数据

    assert not np.shares_memory(get_array(df, "a"), get_array(new_df, "a"))  # 检查修改后两个 DataFrame 的列是否不共享内存
    tm.assert_frame_equal(df, df_orig)  # 断言原始 DataFrame 与复制后的 DataFrame 相等


@pytest.mark.parametrize("dtype", [None, "int64", "Int64"])
@pytest.mark.parametrize("index", [None, [0, 1, 2]])
@pytest.mark.parametrize("columns", [None, ["a", "b"], ["a", "b", "c"]])
def test_dataframe_from_dict_of_series(columns, index, dtype):
    # Case: constructing a DataFrame from Series objects with copy=False
    # has to do a lazy following CoW rules
    # (the default for DataFrame(dict) is still to copy to ensure consolidation)
    s1 = Series([1, 2, 3])  # 创建第一个 Series 对象
    s2 = Series([4, 5, 6])  # 创建第二个 Series 对象
    s1_orig = s1.copy()  # 复制第一个 Series 对象作为原始对象
    expected = DataFrame(
        {"a": [1, 2, 3], "b": [4, 5, 6]}, index=index, columns=columns, dtype=dtype  # 创建预期的 DataFrame 对象
    )

    result = DataFrame(
        {"a": s1, "b": s2}, index=index, columns=columns, dtype=dtype, copy=False  # 使用指定参数创建新的 DataFrame 对象
    )

    # the shallow copy still shares memory
    assert np.shares_memory(get_array(result, "a"), get_array(s1))  # 检查新 DataFrame 的列与原始 Series 是否共享内存

    # mutating the new dataframe doesn't mutate original
    result.iloc[0, 0] = 10  # 修改新 DataFrame 的第一行第一列的值
    assert not np.shares_memory(get_array(result, "a"), get_array(s1))  # 检查修改后新 DataFrame 的列与原始 Series 是否不共享内存
    tm.assert_series_equal(s1, s1_orig)  # 断言原始 Series 与复制的 Series 相等

    # the same when modifying the parent series
    # 创建一个 Pandas Series 对象 s1，包含整数列表 [1, 2, 3]
    s1 = Series([1, 2, 3])
    
    # 创建另一个 Pandas Series 对象 s2，包含整数列表 [4, 5, 6]
    s2 = Series([4, 5, 6])
    
    # 使用 s1 和 s2 创建一个 Pandas DataFrame 对象 result，指定索引、列和数据类型，并且不复制数据
    result = DataFrame(
        {"a": s1, "b": s2}, index=index, columns=columns, dtype=dtype, copy=False
    )
    
    # 修改 s1 中第一个元素的值为 10
    s1.iloc[0] = 10
    
    # 断言 Pandas 库函数 np.shares_memory 返回 False，验证 result 的列 'a' 和 s1 的数组不共享内存
    assert not np.shares_memory(get_array(result, "a"), get_array(s1))
    
    # 使用 Pandas 测试模块 tm 断言 result 等于预期的 DataFrame 对象 expected
    tm.assert_frame_equal(result, expected)
@pytest.mark.parametrize("dtype", [None, "int64"])
# 参数化测试，测试数据类型为None或者"int64"
def test_dataframe_from_dict_of_series_with_reindex(dtype):
    # Case: constructing a DataFrame from Series objects with copy=False
    # and passing an index that requires an actual (no-view) reindex -> need
    # to ensure the result doesn't have refs set up to unnecessarily trigger
    # a copy on write
    # 创建一个DataFrame，从Series对象构建，使用copy=False，并传递一个需要实际重新索引的索引 -> 需要确保结果没有设置引用来不必要地触发写时复制
    s1 = Series([1, 2, 3])
    # 创建第一个Series对象
    s2 = Series([4, 5, 6])
    # 创建第二个Series对象
    df = DataFrame({"a": s1, "b": s2}, index=[1, 2, 3], dtype=dtype, copy=False)
    # 用s1和s2创建DataFrame对象，指定索引为[1, 2, 3]，指定数据类型为dtype，copy=False

    # df should own its memory, so mutating shouldn't trigger a copy
    # df应该拥有自己的内存，因此变异不应该触发复制
    arr_before = get_array(df, "a")
    # 获取df中列'a'的数组表示
    assert not np.shares_memory(arr_before, get_array(s1))
    # 断言df的列'a'与s1的数组不共享内存
    df.iloc[0, 0] = 100
    # 修改df的第一行第一列的值为100
    arr_after = get_array(df, "a")
    # 再次获取df中列'a'的数组表示
    assert np.shares_memory(arr_before, arr_after)
    # 断言修改后的df的列'a'与修改前的df的列'a'共享内存


@pytest.mark.parametrize(
    "data, dtype", [([1, 2], None), ([1, 2], "int64"), (["a", "b"], None)]
)
# 参数化测试，测试数据和数据类型的组合
def test_dataframe_from_series_or_index(data, dtype, index_or_series):
    # 根据输入的数据和数据类型创建索引或者Series对象
    obj = index_or_series(data, dtype=dtype)
    # 复制原始对象
    obj_orig = obj.copy()
    # 用obj创建DataFrame对象
    df = DataFrame(obj, dtype=dtype)
    # 断言obj的数组表示和df的第一列数组表示共享内存
    assert np.shares_memory(get_array(obj), get_array(df, 0))
    # 断言df的内部管理器不具有对第一列的无引用
    assert not df._mgr._has_no_reference(0)

    df.iloc[0, 0] = data[-1]
    # 修改df的第一行第一列的值为data的最后一个元素
    tm.assert_equal(obj, obj_orig)
    # 使用tm模块来断言obj和obj_orig相等


def test_dataframe_from_series_or_index_different_dtype(index_or_series):
    # 根据指定数据和数据类型创建索引或Series对象
    obj = index_or_series([1, 2], dtype="int64")
    # 用obj创建DataFrame对象，指定数据类型为int32
    df = DataFrame(obj, dtype="int32")
    # 断言obj的数组表示和df的第一列数组表示不共享内存
    assert not np.shares_memory(get_array(obj), get_array(df, 0))
    # 断言df的内部管理器具有对第一列的无引用


def test_dataframe_from_series_dont_infer_datetime():
    # 创建包含时间戳的Series对象，数据类型为object
    ser = Series([Timestamp("2019-12-31"), Timestamp("2020-12-31")], dtype=object)
    # 用ser创建DataFrame对象
    df = DataFrame(ser)
    # 断言df的第一列的数据类型为object
    assert df.dtypes.iloc[0] == np.dtype(object)
    # 断言ser的数组表示和df的第一列数组表示共享内存
    assert np.shares_memory(get_array(ser), get_array(df, 0))
    # 断言df的内部管理器不具有对第一列的无引用


@pytest.mark.parametrize("index", [None, [0, 1, 2]])
# 参数化测试，测试索引为None或者[0, 1, 2]
def test_dataframe_from_dict_of_series_with_dtype(index):
    # 变体，但现在传递了会导致复制的数据类型
    # -> 需要确保结果没有设置引用来不必要地触发写时复制
    # 创建第一个Series对象，包含浮点数
    s1 = Series([1.0, 2.0, 3.0])
    # 创建第二个Series对象
    s2 = Series([4, 5, 6])
    # 用s1和s2创建DataFrame对象，指定索引为index，数据类型为int64，copy=False
    df = DataFrame({"a": s1, "b": s2}, index=index, dtype="int64", copy=False)

    # df should own its memory, so mutating shouldn't trigger a copy
    # df应该拥有自己的内存，因此变异不应该触发复制
    arr_before = get_array(df, "a")
    # 获取df中列'a'的数组表示
    assert not np.shares_memory(arr_before, get_array(s1))
    # 断言df的列'a'与s1的数组不共享内存
    df.iloc[0, 0] = 100
    # 修改df的第一行第一列的值为100
    arr_after = get_array(df, "a")
    # 再次获取df中列'a'的数组表示
    assert np.shares_memory(arr_before, arr_after)


@pytest.mark.parametrize("copy", [False, None, True])
# 参数化测试，测试copy参数为False, None, True
def test_frame_from_numpy_array(copy):
    # 创建一个numpy数组
    arr = np.array([[1, 2], [3, 4]])
    # 用numpy数组创建DataFrame对象，指定copy参数
    df = DataFrame(arr, copy=copy)

    if copy is not False or copy is True:
        # 如果copy不是False或者True，断言df的第一列数组表示与arr不共享内存
        assert not np.shares_memory(get_array(df, 0), arr)
    else:
        # 否则，断言df的第一列数组表示与arr共享内存
        assert np.shares_memory(get_array(df, 0), arr)


def test_frame_from_dict_of_index():
    # 创建一个Index对象
    idx = Index([1, 2, 3])
    # 复制idx对象
    expected = idx.copy(deep=True)
    # 用idx创建DataFrame对象，指定copy=False
    df = DataFrame({"a": idx}, copy=False)
    # 使用 NumPy 函数检查两个数组是否共享内存，确保 DataFrame 的列 "a" 和索引 idx._values 共享内存
    assert np.shares_memory(get_array(df, "a"), idx._values)
    
    # 检查 DataFrame 的内部管理器是否没有对第一个位置（0）的数据的引用
    assert not df._mgr._has_no_reference(0)

    # 将 DataFrame 的第一行第一列的值设置为 100
    df.iloc[0, 0] = 100
    
    # 使用测试模块（tm）的方法来断言索引 idx 是否等于预期的索引（expected）
    tm.assert_index_equal(idx, expected)
```