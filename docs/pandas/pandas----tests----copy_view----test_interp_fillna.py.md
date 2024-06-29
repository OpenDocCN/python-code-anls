# `D:\src\scipysrc\pandas\pandas\tests\copy_view\test_interp_fillna.py`

```
import numpy as np
import pytest

from pandas import (
    NA,
    DataFrame,
    Interval,
    NaT,
    Series,
    Timestamp,
    interval_range,
)
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array

# 使用 pytest 的 parametrize 装饰器，指定 method 参数的多个取值
@pytest.mark.parametrize("method", ["pad", "nearest", "linear"])
def test_interpolate_no_op(method):
    # 创建一个包含列 'a' 的 DataFrame 对象
    df = DataFrame({"a": [1, 2]})
    # 复制原始 DataFrame 对象
    df_orig = df.copy()

    # 根据 method 的不同取值进行不同的测试
    if method == "pad":
        # 如果 method 是 'pad'，则抛出 ValueError 异常，并匹配指定的消息
        msg = f"Can not interpolate with method={method}"
        with pytest.raises(ValueError, match=msg):
            df.interpolate(method=method)
    else:
        # 对 DataFrame 进行插值操作，结果赋值给 result
        result = df.interpolate(method=method)
        # 断言结果数组与原始数组共享内存
        assert np.shares_memory(get_array(result, "a"), get_array(df, "a"))

        # 修改结果的第一个元素
        result.iloc[0, 0] = 100

        # 断言修改后，结果数组与原始数组不再共享内存
        assert not np.shares_memory(get_array(result, "a"), get_array(df, "a"))
        # 断言经过插值后的 DataFrame 与原始 DataFrame 相等
        tm.assert_frame_equal(df, df_orig)

# 使用 pytest 的 parametrize 装饰器，指定 func 参数的多个取值
@pytest.mark.parametrize("func", ["ffill", "bfill"])
def test_interp_fill_functions(func):
    # 检查这些函数与 interpolate 方法的代码路径是否相同
    df = DataFrame({"a": [1, 2]})
    df_orig = df.copy()

    # 调用 DataFrame 的 ffill 或 bfill 方法
    result = getattr(df, func)()

    # 断言结果数组与原始数组共享内存
    assert np.shares_memory(get_array(result, "a"), get_array(df, "a"))
    # 修改结果的第一个元素
    result.iloc[0, 0] = 100

    # 断言修改后，结果数组与原始数组不再共享内存
    assert not np.shares_memory(get_array(result, "a"), get_array(df, "a"))
    # 断言经过填充后的 DataFrame 与原始 DataFrame 相等
    tm.assert_frame_equal(df, df_orig)

# 使用 pytest 的 parametrize 装饰器，指定 func 和 vals 参数的多个取值
@pytest.mark.parametrize("func", ["ffill", "bfill"])
@pytest.mark.parametrize(
    "vals", [[1, np.nan, 2], [Timestamp("2019-12-31"), NaT, Timestamp("2020-12-31")]]
)
def test_interpolate_triggers_copy(vals, func):
    # 创建一个包含列 'a' 的 DataFrame 对象，列值为 vals
    df = DataFrame({"a": vals})
    # 调用 DataFrame 的 ffill 或 bfill 方法
    result = getattr(df, func)()

    # 断言结果数组与原始数组不共享内存
    assert not np.shares_memory(get_array(result, "a"), get_array(df, "a"))
    # 检查在触发复制时没有引用
    assert result._mgr._has_no_reference(0)

# 使用 pytest 的 parametrize 装饰器，指定 vals 参数的多个取值
@pytest.mark.parametrize(
    "vals", [[1, np.nan, 2], [Timestamp("2019-12-31"), NaT, Timestamp("2020-12-31")]]
)
def test_interpolate_inplace_no_reference_no_copy(vals):
    # 创建一个包含列 'a' 的 DataFrame 对象，列值为 vals
    df = DataFrame({"a": vals})
    # 获取列 'a' 的数组视图
    arr = get_array(df, "a")
    # 在原地进行线性插值
    df.interpolate(method="linear", inplace=True)

    # 断言原始数组与插值后的数组共享内存
    assert np.shares_memory(arr, get_array(df, "a"))
    # 检查在触发复制时没有引用
    assert df._mgr._has_no_reference(0)

# 使用 pytest 的 parametrize 装饰器，指定 vals 参数的多个取值
@pytest.mark.parametrize(
    "vals", [[1, np.nan, 2], [Timestamp("2019-12-31"), NaT, Timestamp("2020-12-31")]]
)
def test_interpolate_inplace_with_refs(vals):
    # 创建一个包含列 'a' 的 DataFrame 对象，列值为 [1, np.nan, 2]
    df = DataFrame({"a": [1, np.nan, 2]})
    # 复制原始 DataFrame 对象
    df_orig = df.copy()
    # 获取列 'a' 的数组视图
    arr = get_array(df, "a")
    # 获取 DataFrame 的视图
    view = df[:]
    # 在原地进行线性插值
    df.interpolate(method="linear", inplace=True)
    # 检查是否在插值时触发了复制，并且没有任何引用留下
    assert not np.shares_memory(arr, get_array(df, "a"))
    # 断言插值前后原始 DataFrame 与视图相等
    tm.assert_frame_equal(df_orig, view)
    assert df._mgr._has_no_reference(0)
    assert view._mgr._has_no_reference(0)

# 使用 pytest 的 parametrize 装饰器，指定 func 参数的多个取值
@pytest.mark.parametrize("func", ["ffill", "bfill"])
@pytest.mark.parametrize("dtype", ["float64", "Float64"])
# 测试插值填充函数是否能够就地修改数据框中的列
def test_interp_fill_functions_inplace(func, dtype):
    # 创建一个包含空值的数据框
    df = DataFrame({"a": [1, np.nan, 2]}, dtype=dtype)
    # 备份原始数据框
    df_orig = df.copy()
    # 获取列"a"的数组表示
    arr = get_array(df, "a")
    # 创建数据框的视图
    view = df[:]

    # 调用指定的插值填充函数并设置 inplace=True 以便就地修改
    getattr(df, func)(inplace=True)

    # 检查插值操作是否触发了复制操作，并确保没有任何引用留在原始数据框上
    assert not np.shares_memory(arr, get_array(df, "a"))
    # 检查视图与原始数据框是否相等
    tm.assert_frame_equal(df_orig, view)
    # 检查经过操作后数据框是否不再有引用指向第一个块
    assert df._mgr._has_no_reference(0)
    # 检查视图经过操作后是否不再有引用指向第一个块
    assert view._mgr._has_no_reference(0)


# 测试无法使用对象类型进行插值的情况
def test_interpolate_cannot_with_object_dtype():
    # 创建一个包含对象类型列的数据框
    df = DataFrame({"a": ["a", np.nan, "c"], "b": 1})

    # 准备错误消息
    msg = "DataFrame cannot interpolate with object dtype"
    # 检查插值时是否会引发预期的类型错误异常
    with pytest.raises(TypeError, match=msg):
        df.interpolate()


# 测试对象类型列进行插值时不会进行转换操作
def test_interpolate_object_convert_no_op():
    # 创建一个包含对象类型列的数据框
    df = DataFrame({"a": ["a", "b", "c"], "b": 1})
    # 获取列"a"的数组表示
    arr_a = get_array(df, "a")

    # 检查在执行Copy-on-Write（CoW）时是否会进行复制操作，预期应不复制
    assert df._mgr._has_no_reference(0)
    # 检查在插值操作之前和之后，列"a"的数组是否共享内存
    assert np.shares_memory(arr_a, get_array(df, "a"))


# 测试对象类型列进行插值时会触发复制操作
def test_interpolate_object_convert_copies():
    # 创建一个包含数值和空值的数据框
    df = DataFrame({"a": [1, np.nan, 2.5], "b": 1})
    # 获取列"a"的数组表示
    arr_a = get_array(df, "a")
    
    # 准备错误消息
    msg = "Can not interpolate with method=pad"
    # 检查在使用方法"pad"进行插值时是否会引发预期的值错误异常，并检查是否未就地修改
    with pytest.raises(ValueError, match=msg):
        df.interpolate(method="pad", inplace=True, downcast="infer")

    # 检查经过操作后数据框是否不再有引用指向第一个块
    assert df._mgr._has_no_reference(0)
    # 检查在插值操作之前和之后，列"a"的数组是否不再共享内存
    assert not np.shares_memory(arr_a, get_array(df, "a"))


# 测试插值时下转引用是否触发复制操作
def test_interpolate_downcast_reference_triggers_copy():
    # 创建一个包含数值和空值的数据框
    df = DataFrame({"a": [1, np.nan, 2.5], "b": 1})
    # 备份原始数据框
    df_orig = df.copy()
    # 获取列"a"的数组表示
    arr_a = get_array(df, "a")
    # 创建数据框的视图
    view = df[:]

    # 准备错误消息
    msg = "Can not interpolate with method=pad"
    # 检查在使用方法"pad"进行插值时是否会引发预期的值错误异常，并检查是否未就地修改
    with pytest.raises(ValueError, match=msg):
        df.interpolate(method="pad", inplace=True, downcast="infer")
        assert df._mgr._has_no_reference(0)
        assert not np.shares_memory(arr_a, get_array(df, "a"))

    # 检查视图与原始数据框是否相等
    tm.assert_frame_equal(df_orig, view)


# 测试填充空值操作
def test_fillna():
    # 创建一个包含空值的数据框
    df = DataFrame({"a": [1.5, np.nan], "b": 1})
    # 备份原始数据框
    df_orig = df.copy()

    # 使用指定的值填充空值，并返回新的数据框
    df2 = df.fillna(5.5)
    # 检查在填充操作后，列"b"的数组是否共享内存
    assert np.shares_memory(get_array(df, "b"), get_array(df2, "b"))
    # 修改填充后的数据框，并检查原始数据框是否保持不变
    df2.iloc[0, 1] = 100
    tm.assert_frame_equal(df_orig, df)


# 测试使用字典进行填充操作
def test_fillna_dict():
    # 创建一个包含空值的数据框
    df = DataFrame({"a": [1.5, np.nan], "b": 1})
    # 备份原始数据框
    df_orig = df.copy()

    # 使用字典对指定的列进行填充，并返回新的数据框
    df2 = df.fillna({"a": 100.5})
    # 检查在填充操作后，列"b"的数组是否共享内存
    assert np.shares_memory(get_array(df, "b"), get_array(df2, "b"))
    # 检查在填充操作后，列"a"的数组是否不再共享内存
    assert not np.shares_memory(get_array(df, "a"), get_array(df2, "a"))

    # 修改填充后的数据框，并检查原始数据框是否保持不变
    df2.iloc[0, 1] = 100
    tm.assert_frame_equal(df_orig, df)


# 测试就地填充操作
def test_fillna_inplace():
    # 创建一个包含空值的数据框
    df = DataFrame({"a": [1.5, np.nan], "b": 1})
    # 获取列"a"和"b"的数组表示
    arr_a = get_array(df, "a")
    arr_b = get_array(df, "b")

    # 使用指定的值就地填充空值
    df.fillna(5.5, inplace=True)
    # 检查在填充操作后，列"a"的数组是否与原始数组共享内存
    assert np.shares_memory(get_array(df, "a"), arr_a)
    # 检查在填充操作后，列"b"的数组是否与原始数组共享内存
    assert np.shares_memory(get_array(df, "b"), arr_b)
    # 检查经过操作后数据框是否不再有引用指向第一个块
    assert df._mgr._has_no_reference(0)
    # 检查经过操作后数据框是否不再有引用指向第二个块
    assert df._mgr._has_no_reference(1)


# 测试就地填充操作是否会触发复制操作
def test_fillna_inplace_reference():
    # 创建一个包含空值的数据框
    df = DataFrame({"a": [1.5, np.nan], "b": 1})
    # 创建原始数据框的副本
    df_orig = df.copy()
    
    # 使用自定义函数从数据框中获取列 "a" 的数组
    arr_a = get_array(df, "a")
    
    # 使用自定义函数从数据框中获取列 "b" 的数组
    arr_b = get_array(df, "b")
    
    # 创建数据框的视图
    view = df[:]
    
    # 将数据框中的缺失值填充为 5.5，并在原地修改数据框
    df.fillna(5.5, inplace=True)
    
    # 断言：确保修改后的数组 "a" 不再与原始数组 arr_a 共享内存
    assert not np.shares_memory(get_array(df, "a"), arr_a)
    
    # 断言：确保修改后的数组 "b" 仍与原始数组 arr_b 共享内存
    assert np.shares_memory(get_array(df, "b"), arr_b)
    
    # 断言：确保数据视图 view 没有对第一个块的引用
    assert view._mgr._has_no_reference(0)
    
    # 断言：确保数据框 df 没有对第一个块的引用
    assert df._mgr._has_no_reference(0)
    
    # 使用测试工具函数比较视图和原始数据框，确保它们相等
    tm.assert_frame_equal(view, df_orig)
    
    # 创建预期的数据框，包含特定的列和值
    expected = DataFrame({"a": [1.5, 5.5], "b": 1})
    
    # 使用测试工具函数比较当前数据框和预期数据框，确保它们相等
    tm.assert_frame_equal(df, expected)
# 测试填充缺失值的方法，直接在原对象上进行填充，而非创建新对象
def test_fillna_interval_inplace_reference():
    # 明确设置数据类型，避免在设置 NaN 时隐式转换
    ser = Series(
        interval_range(start=0, end=5), name="a", dtype="interval[float64, right]"
    )
    # 将第二个元素设置为 NaN
    ser.iloc[1] = np.nan

    # 备份原始序列
    ser_orig = ser.copy()
    # 创建视图对象，与原序列共享数据
    view = ser[:]
    # 在原序列上填充缺失值为指定区间
    ser.fillna(value=Interval(left=0, right=5), inplace=True)

    # 断言：确认视图对象与原序列的数据内存不共享
    assert not np.shares_memory(
        get_array(ser, "a").left.values, get_array(view, "a").left.values
    )
    # 断言：确认视图对象与原始序列保持一致
    tm.assert_series_equal(view, ser_orig)


# 测试对空参数进行填充操作
def test_fillna_series_empty_arg():
    # 创建包含 NaN 的序列
    ser = Series([1, np.nan, 2])
    # 备份原始序列
    ser_orig = ser.copy()
    # 对序列进行空字典填充
    result = ser.fillna({})
    # 断言：确认原序列与填充后的结果共享数据内存
    assert np.shares_memory(get_array(ser), get_array(result))

    # 修改原序列第一个元素
    ser.iloc[0] = 100.5
    # 断言：确认原始序列与填充后的结果保持一致
    tm.assert_series_equal(ser_orig, result)


# 测试对空参数进行填充操作，并在原对象上进行修改
def test_fillna_series_empty_arg_inplace():
    # 创建包含 NaN 的序列
    ser = Series([1, np.nan, 2])
    # 获取原序列的数据数组
    arr = get_array(ser)
    # 在原序列上进行空字典填充
    ser.fillna({}, inplace=True)

    # 断言：确认填充后的结果与原序列共享数据内存
    assert np.shares_memory(get_array(ser), arr)
    # 断言：确认序列对象中没有引用
    assert ser._mgr._has_no_reference(0)


# 测试对特定数据类型的数据框进行填充操作
def test_fillna_ea_noop_shares_memory(any_numeric_ea_and_arrow_dtype):
    # 创建包含 NaN 的数据框
    df = DataFrame({"a": [1, NA, 3], "b": 1}, dtype=any_numeric_ea_and_arrow_dtype)
    # 备份原始数据框
    df_orig = df.copy()
    # 使用特定值填充 NaN
    df2 = df.fillna(100)

    # 断言：确认填充后的数据框与原数据框的 'a' 列数据内存不共享
    assert not np.shares_memory(get_array(df, "a"), get_array(df2, "a"))

    # 断言：确认填充后的数据框与原数据框的 'b' 列数据内存共享
    assert np.shares_memory(get_array(df, "b"), get_array(df2, "b"))
    # 断言：确认填充后的数据框对象中存在引用
    assert not df2._mgr._has_no_reference(1)
    # 断言：确认填充后的数据框与原数据框保持一致
    tm.assert_frame_equal(df_orig, df)

    # 修改填充后数据框的元素
    df2.iloc[0, 1] = 100
    # 断言：确认修改后的数据框的 'b' 列数据内存不共享
    assert not np.shares_memory(get_array(df, "b"), get_array(df2, "b"))
    # 断言：确认填充后数据框对象中不存在引用
    assert df2._mgr._has_no_reference(1)
    # 断言：确认原数据框对象中不存在引用
    assert df._mgr._has_no_reference(1)
    # 断言：确认填充后的数据框与原数据框保持一致
    tm.assert_frame_equal(df_orig, df)


# 测试在原数据框对象上进行填充操作
def test_fillna_inplace_ea_noop_shares_memory(any_numeric_ea_and_arrow_dtype):
    # 创建包含 NaN 的数据框
    df = DataFrame({"a": [1, NA, 3], "b": 1}, dtype=any_numeric_ea_and_arrow_dtype)
    # 备份原始数据框
    df_orig = df.copy()
    # 创建数据框的视图
    view = df[:]
    # 在原数据框上进行填充操作
    df.fillna(100, inplace=True)

    # 断言：确认填充后的数据框与视图的 'a' 列数据内存不共享
    assert not np.shares_memory(get_array(df, "a"), get_array(view, "a"))

    # 断言：确认填充后的数据框与视图的 'b' 列数据内存共享
    assert np.shares_memory(get_array(df, "b"), get_array(view, "b"))
    # 断言：确认填充后的数据框对象中存在引用
    assert not df._mgr._has_no_reference(1)
    # 断言：确认视图对象中存在引用
    assert not view._mgr._has_no_reference(1)

    # 修改填充后数据框的元素
    df.iloc[0, 1] = 100
    # 断言：确认填充后的数据框与原数据框保持一致
    tm.assert_frame_equal(df_orig, view)


# 测试链式赋值情况下的填充操作
def test_fillna_chained_assignment():
    # 创建包含 NaN 的数据框
    df = DataFrame({"a": [1, np.nan, 2], "b": 1})
    # 备份原始数据框
    df_orig = df.copy()
    # 断言：确认在链式赋值情况下进行填充会引发异常
    with tm.raises_chained_assignment_error():
        df["a"].fillna(100, inplace=True)
    # 断言：确认数据框保持不变
    tm.assert_frame_equal(df, df_orig)

    # 断言：确认在链式赋值情况下对多列进行填充会引发异常
    with tm.raises_chained_assignment_error():
        df[["a"]].fillna(100, inplace=True)
    # 断言：确认数据框保持不变
    tm.assert_frame_equal(df, df_orig)


# 测试插值、前向填充和后向填充时的链式赋值情况
@pytest.mark.parametrize("func", ["interpolate", "ffill", "bfill"])
def test_interpolate_chained_assignment(func):
    # 创建包含 NaN 的数据框
    df = DataFrame({"a": [1, np.nan, 2], "b": 1})
    # 备份原始数据框
    df_orig = df.copy()
    # 断言：确认在链式赋值情况下进行插值、前向填充或后向填充会引发异常
    with tm.raises_chained_assignment_error():
        getattr(df["a"], func)(inplace=True)
    # 断言：确认数据框保持不变
    tm.assert_frame_equal(df, df_orig)

    # 断言：确认在链式赋值情况下对单列进行插值、前向填充或后向填充会引发异常
    with tm.raises_chained_assignment_error():
        getattr(df[["a"]], func)(inplace=True)
    # 使用测试框架中的断言函数，比较数据框 df 和 df_orig 是否相等
    tm.assert_frame_equal(df, df_orig)
```