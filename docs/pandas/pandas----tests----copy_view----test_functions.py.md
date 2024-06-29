# `D:\src\scipysrc\pandas\pandas\tests\copy_view\test_functions.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算
import pytest  # 导入 pytest 库，用于编写和运行测试用例

from pandas import (  # 从 pandas 库中导入以下模块：
    DataFrame,  # DataFrame：用于处理表格数据的主要数据结构
    Index,  # Index：用于表示索引的数据结构
    Series,  # Series：用于表示一维标记数组的数据结构
    concat,  # concat：用于沿指定轴连接 pandas 对象
    merge,  # merge：用于数据库样式的数据连接操作
)
import pandas._testing as tm  # 导入 pandas 内部测试工具
from pandas.tests.copy_view.util import get_array  # 导入从测试工具获取数组的函数


def test_concat_frames():
    df = DataFrame({"b": ["a"] * 3})  # 创建包含'b'列的 DataFrame 对象
    df2 = DataFrame({"a": ["a"] * 3})  # 创建包含'a'列的 DataFrame 对象
    df_orig = df.copy()  # 复制原始 DataFrame 对象
    result = concat([df, df2], axis=1)  # 沿列轴连接 df 和 df2，生成新的 DataFrame 对象

    assert np.shares_memory(get_array(result, "b"), get_array(df, "b"))  # 断言'b'列数据共享内存
    assert np.shares_memory(get_array(result, "a"), get_array(df2, "a"))  # 断言'a'列数据共享内存

    result.iloc[0, 0] = "d"  # 修改结果 DataFrame 的第一行第一列元素
    assert not np.shares_memory(get_array(result, "b"), get_array(df, "b"))  # 断言'b'列数据不再共享内存
    assert np.shares_memory(get_array(result, "a"), get_array(df2, "a"))  # 断言'a'列数据仍然共享内存

    result.iloc[0, 1] = "d"  # 修改结果 DataFrame 的第一行第二列元素
    assert not np.shares_memory(get_array(result, "a"), get_array(df2, "a"))  # 断言'a'列数据不再共享内存
    tm.assert_frame_equal(df, df_orig)  # 使用测试工具断言 df 和 df_orig 相等


def test_concat_frames_updating_input():
    df = DataFrame({"b": ["a"] * 3})  # 创建包含'b'列的 DataFrame 对象
    df2 = DataFrame({"a": ["a"] * 3})  # 创建包含'a'列的 DataFrame 对象
    result = concat([df, df2], axis=1)  # 沿列轴连接 df 和 df2，生成新的 DataFrame 对象

    assert np.shares_memory(get_array(result, "b"), get_array(df, "b"))  # 断言'b'列数据共享内存
    assert np.shares_memory(get_array(result, "a"), get_array(df2, "a"))  # 断言'a'列数据共享内存

    expected = result.copy()  # 复制结果 DataFrame 对象
    df.iloc[0, 0] = "d"  # 修改 df 的第一行第一列元素
    assert not np.shares_memory(get_array(result, "b"), get_array(df, "b"))  # 断言'b'列数据不再共享内存
    assert np.shares_memory(get_array(result, "a"), get_array(df2, "a"))  # 断言'a'列数据仍然共享内存

    df2.iloc[0, 0] = "d"  # 修改 df2 的第一行第一列元素
    assert not np.shares_memory(get_array(result, "a"), get_array(df2, "a"))  # 断言'a'列数据不再共享内存
    tm.assert_frame_equal(result, expected)  # 使用测试工具断言 result 和 expected 相等


def test_concat_series():
    ser = Series([1, 2], name="a")  # 创建名为'a'的 Series 对象
    ser2 = Series([3, 4], name="b")  # 创建名为'b'的 Series 对象
    ser_orig = ser.copy()  # 复制原始 Series 对象
    ser2_orig = ser2.copy()  # 复制原始 Series 对象
    result = concat([ser, ser2], axis=1)  # 沿列轴连接 ser 和 ser2，生成新的 DataFrame 对象

    assert np.shares_memory(get_array(result, "a"), ser.values)  # 断言'a'列数据共享内存
    assert np.shares_memory(get_array(result, "b"), ser2.values)  # 断言'b'列数据共享内存

    result.iloc[0, 0] = 100  # 修改结果 DataFrame 的第一行第一列元素
    assert not np.shares_memory(get_array(result, "a"), ser.values)  # 断言'a'列数据不再共享内存
    assert np.shares_memory(get_array(result, "b"), ser2.values)  # 断言'b'列数据仍然共享内存

    result.iloc[0, 1] = 1000  # 修改结果 DataFrame 的第一行第二列元素
    assert not np.shares_memory(get_array(result, "b"), ser2.values)  # 断言'b'列数据不再共享内存
    tm.assert_series_equal(ser, ser_orig)  # 使用测试工具断言 ser 和 ser_orig 相等
    tm.assert_series_equal(ser2, ser2_orig)  # 使用测试工具断言 ser2 和 ser2_orig 相等


def test_concat_frames_chained():
    df1 = DataFrame({"a": [1, 2, 3], "b": [0.1, 0.2, 0.3]})  # 创建包含'a'和'b'列的 DataFrame 对象
    df2 = DataFrame({"c": [4, 5, 6]})  # 创建包含'c'列的 DataFrame 对象
    df3 = DataFrame({"d": [4, 5, 6]})  # 创建包含'd'列的 DataFrame 对象
    result = concat([concat([df1, df2], axis=1), df3], axis=1)  # 链式连接 df1、df2 和 df3，沿列轴生成新的 DataFrame 对象
    expected = result.copy()  # 复制结果 DataFrame 对象

    assert np.shares_memory(get_array(result, "a"), get_array(df1, "a"))  # 断言'a'列数据共享内存
    assert np.shares_memory(get_array(result, "c"), get_array(df2, "c"))  # 断言'c'列数据共享内存
    assert np.shares_memory(get_array(result, "d"), get_array(df3, "d"))  # 断言'd'列数据共享内存

    df1.iloc[0, 0] = 100  # 修改 df1 的第一行第一列元素
    assert not np.shares_memory(get_array(result, "a"), get_array(df1, "a"))  # 断言'a'列数据不再共享内存

    tm.assert_frame_equal(result, expected)  # 使用测试工具断言 result 和 expected 相等


def test_concat_series_chained():
    ser1 = Series([1, 2, 3], name="a")  # 创建名为'a'的 Series 对象
    ser2 = Series([4, 5, 6], name="c")  # 创建名为'c'的 Series 对象
    ser3 = Series([4, 5, 6], name="d")  # 创建名为'd'的 Series 对象
    # 将三个序列按列连接，形成一个结果序列
    result = concat([concat([ser1, ser2], axis=1), ser3], axis=1)
    
    # 创建一个预期结果的副本
    expected = result.copy()
    
    # 断言：检查结果序列的特定列与原始序列共享内存
    assert np.shares_memory(get_array(result, "a"), get_array(ser1, "a"))
    assert np.shares_memory(get_array(result, "c"), get_array(ser2, "c"))
    assert np.shares_memory(get_array(result, "d"), get_array(ser3, "d"))
    
    # 修改原始序列 ser1 的第一个元素
    ser1.iloc[0] = 100
    
    # 断言：检查修改后的序列不再与结果序列的特定列共享内存
    assert not np.shares_memory(get_array(result, "a"), get_array(ser1, "a"))
    
    # 使用测试框架断言：检查 result 和 expected 是否相等
    tm.assert_frame_equal(result, expected)
# 定义一个测试函数，用于验证拼接 Series 时更新输入数据的行为
def test_concat_series_updating_input():
    # 创建两个 Series 对象，每个对象分别包含两个整数，分别命名为 "a" 和 "b"
    ser = Series([1, 2], name="a")
    ser2 = Series([3, 4], name="b")
    # 创建期望的 DataFrame 对象，包含两列："a" 和 "b"
    expected = DataFrame({"a": [1, 2], "b": [3, 4]})
    # 执行拼接操作，将两个 Series 沿着列方向拼接成 DataFrame
    result = concat([ser, ser2], axis=1)

    # 断言：检查拼接后的结果中 "a" 列的数据是否与原始 ser 对象共享内存
    assert np.shares_memory(get_array(result, "a"), get_array(ser, "a"))
    # 断言：检查拼接后的结果中 "b" 列的数据是否与原始 ser2 对象共享内存
    assert np.shares_memory(get_array(result, "b"), get_array(ser2, "b"))

    # 修改 ser 中的第一个元素
    ser.iloc[0] = 100
    # 断言：检查修改后的 ser 对象的 "a" 列数据是否不再与拼接结果中的共享内存
    assert not np.shares_memory(get_array(result, "a"), get_array(ser, "a"))
    # 断言：检查拼接后的结果中 "b" 列的数据是否仍与原始 ser2 对象共享内存
    assert np.shares_memory(get_array(result, "b"), get_array(ser2, "b"))
    # 断言：检查整个拼接结果是否与期望的 DataFrame 对象相等
    tm.assert_frame_equal(result, expected)

    # 修改 ser2 中的第一个元素
    ser2.iloc[0] = 1000
    # 断言：检查修改后的 ser2 对象的 "b" 列数据是否不再与拼接结果中的共享内存
    assert not np.shares_memory(get_array(result, "b"), get_array(ser2, "b"))
    # 断言：检查整个拼接结果是否与期望的 DataFrame 对象相等
    tm.assert_frame_equal(result, expected)


# 定义一个测试函数，用于验证拼接不同类型的 Series 和 DataFrame 时的行为
def test_concat_mixed_series_frame():
    # 创建一个 DataFrame 对象，包含两列 "a" 和 "c"
    df = DataFrame({"a": [1, 2, 3], "c": 1})
    # 创建一个 Series 对象，包含三个整数，命名为 "d"
    ser = Series([4, 5, 6], name="d")
    # 执行拼接操作，将 DataFrame 和 Series 沿着列方向拼接成新的 DataFrame
    result = concat([df, ser], axis=1)
    # 复制拼接结果作为期望结果
    expected = result.copy()

    # 断言：检查拼接后的结果中 "a" 列的数据是否与原始 df 对象共享内存
    assert np.shares_memory(get_array(result, "a"), get_array(df, "a"))
    # 断言：检查拼接后的结果中 "c" 列的数据是否与原始 df 对象共享内存
    assert np.shares_memory(get_array(result, "c"), get_array(df, "c"))
    # 断言：检查拼接后的结果中 "d" 列的数据是否与原始 ser 对象共享内存
    assert np.shares_memory(get_array(result, "d"), get_array(ser, "d"))

    # 修改 ser 中的第一个元素
    ser.iloc[0] = 100
    # 断言：检查修改后的 ser 对象的 "d" 列数据是否不再与拼接结果中的共享内存
    assert not np.shares_memory(get_array(result, "d"), get_array(ser, "d"))

    # 修改 df 中的第一个元素的第一个列
    df.iloc[0, 0] = 100
    # 断言：检查修改后的 df 对象的 "a" 列数据是否不再与拼接结果中的共享内存
    assert not np.shares_memory(get_array(result, "a"), get_array(df, "a"))
    # 断言：检查整个拼接结果是否与期望的 DataFrame 对象相等
    tm.assert_frame_equal(result, expected)


# 定义一个测试函数，用于验证拼接 DataFrame 时使用 copy 关键字的行为
def test_concat_copy_keyword():
    # 创建两个 DataFrame 对象，每个对象包含一列整数
    df = DataFrame({"a": [1, 2]})
    df2 = DataFrame({"b": [1.5, 2.5]})

    # 执行拼接操作，将两个 DataFrame 沿着列方向拼接成新的 DataFrame
    result = concat([df, df2], axis=1)

    # 断言：检查拼接后的结果中 "a" 列的数据是否与原始 df 对象共享内存
    assert np.shares_memory(get_array(df, "a"), get_array(result, "a"))
    # 断言：检查拼接后的结果中 "b" 列的数据是否与原始 df2 对象共享内存
    assert np.shares_memory(get_array(df2, "b"), get_array(result, "b"))


# 使用 pytest 的参数化装饰器，定义一个参数化测试函数，测试 DataFrame 的合并操作
@pytest.mark.parametrize(
    "func",
    [
        lambda df1, df2, **kwargs: df1.merge(df2, **kwargs),
        lambda df1, df2, **kwargs: merge(df1, df2, **kwargs),
    ],
)
def test_merge_on_key(func):
    # 创建两个 DataFrame 对象，每个对象包含两列数据
    df1 = DataFrame({"key": ["a", "b", "c"], "a": [1, 2, 3]})
    df2 = DataFrame({"key": ["a", "b", "c"], "b": [4, 5, 6]})
    # 复制原始的 df1 和 df2 对象，用于后续的断言比较
    df1_orig = df1.copy()
    df2_orig = df2.copy()

    # 执行合并操作，根据 "key" 列将两个 DataFrame 拼接成一个新的 DataFrame
    result = func(df1, df2, on="key")

    # 断言：检查合并后的结果中 "a" 列的数据是否与原始 df1 对象共享内存
    assert np.shares_memory(get_array(result, "a"), get_array(df1, "a"))
    # 断言：检查合并后的结果中 "b" 列的数据是否与原始 df2 对象共享内存
    assert np.shares_memory(get_array(result, "b"), get_array(df2, "b"))
    # 断言：检查合并后的结果中 "key" 列的数据是否与原始 df1 对象共享内存
    assert np.shares_memory(get_array(result, "key"), get_array(df1, "key"))
    # 断言：检查合并后的结果中 "key" 列的数据是否不再与原始 df2 对象共享内存
    assert not np.shares_memory(get_array(result, "key"), get_array(df2, "key"))

    # 修改合并结果中的第一行第二列的数据为 0
    result.iloc[0, 1] = 0
    # 断言：检查修改后的合并结果中的 "a" 列的数据是否不再与原始 df1 对象共享内存
    assert not np.shares_memory(get_array(result, "a"), get_array(df1, "a"))
    # 断言：检查合并后的结果中 "b" 列的数据是否仍与原始 df2 对象共享内存
    assert np.shares_memory(get_array(result, "b"), get_array(df2, "b"))

    # 修改合并结果中的第一行第三列的数据为 0
    result.iloc[0, 2] = 0
    # 断言：检查修改后的合并结果中的 "b" 列的数据是否不
    # 确认result中列"a"的数组与df1中列"a"的数组共享内存
    assert np.shares_memory(get_array(result, "a"), get_array(df1, "a"))
    
    # 确认result中列"b"的数组与df2中列"b"的数组共享内存
    assert np.shares_memory(get_array(result, "b"), get_array(df2, "b"))
    
    # 将result的第一行第一列元素设为0
    result.iloc[0, 0] = 0
    
    # 确认result中列"a"的数组不再与df1中列"a"的数组共享内存
    assert not np.shares_memory(get_array(result, "a"), get_array(df1, "a"))
    
    # 确认result中列"b"的数组仍与df2中列"b"的数组共享内存
    assert np.shares_memory(get_array(result, "b"), get_array(df2, "b"))
    
    # 将result的第一行第二列元素设为0
    result.iloc[0, 1] = 0
    
    # 确认result中列"b"的数组不再与df2中列"b"的数组共享内存
    assert not np.shares_memory(get_array(result, "b"), get_array(df2, "b"))
    
    # 检查df1与df1_orig是否相等
    tm.assert_frame_equal(df1, df1_orig)
    
    # 检查df2与df2_orig是否相等
    tm.assert_frame_equal(df2, df2_orig)
# 使用 pytest 的 parametrize 装饰器，为 test_merge_on_key_enlarging_one 函数参数化多个测试用例
@pytest.mark.parametrize(
    "func, how",
    [
        # 定义一个匿名函数，用于执行右连接操作，将 df2 合并到 df1 上
        (lambda df1, df2, **kwargs: merge(df2, df1, on="key", **kwargs), "right"),
        # 定义一个匿名函数，用于执行左连接操作，将 df1 合并到 df2 上
        (lambda df1, df2, **kwargs: merge(df1, df2, on="key", **kwargs), "left"),
    ],
)
# 定义测试函数 test_merge_on_key_enlarging_one，用于测试数据框合并时的内存共享和复制情况
def test_merge_on_key_enlarging_one(func, how):
    # 创建 DataFrame df1 包含列 'key' 和 'a'
    df1 = DataFrame({"key": ["a", "b", "c"], "a": [1, 2, 3]})
    # 创建 DataFrame df2 包含列 'key' 和 'b'
    df2 = DataFrame({"key": ["a", "b"], "b": [4, 5]})
    # 复制 df1 和 df2 以备后续比较
    df1_orig = df1.copy()
    df2_orig = df2.copy()

    # 调用指定的合并函数 func 进行数据框合并
    result = func(df1, df2, how=how)

    # 断言：验证 'result' 的 'a' 列是否与 'df1' 的 'a' 列共享内存
    assert np.shares_memory(get_array(result, "a"), get_array(df1, "a"))
    # 断言：验证 'result' 的 'b' 列是否与 'df2' 的 'b' 列未共享内存
    assert not np.shares_memory(get_array(result, "b"), get_array(df2, "b"))
    # 断言：验证 df2 是否已经没有被其他对象引用，以便进行内存管理的清理
    assert df2._mgr._has_no_reference(1)
    assert df2._mgr._has_no_reference(0)
    # 断言：验证 'result' 的 'key' 列是否与 'df1' 的 'key' 列在左连接时共享内存
    assert np.shares_memory(get_array(result, "key"), get_array(df1, "key")) is (how == "left")
    # 断言：验证 'result' 的 'key' 列是否与 'df2' 的 'key' 列未共享内存
    assert not np.shares_memory(get_array(result, "key"), get_array(df2, "key"))

    # 根据连接类型修改 'result' 数据框的元素，以验证数据框是否发生复制
    if how == "left":
        result.iloc[0, 1] = 0
    else:
        result.iloc[0, 2] = 0
    # 断言：验证 'result' 的 'a' 列是否与 'df1' 的 'a' 列未共享内存
    assert not np.shares_memory(get_array(result, "a"), get_array(df1, "a"))
    # 使用测试工具 tm 进行断言：验证 df1 和 df2 是否与其初始副本相等
    tm.assert_frame_equal(df1, df1_orig)
    tm.assert_frame_equal(df2, df2_orig)


# 定义测试函数 test_merge_copy_keyword，用于测试数据框合并时复制关键字的效果
def test_merge_copy_keyword():
    # 创建 DataFrame df 包含列 'a'
    df = DataFrame({"a": [1, 2]})
    # 创建 DataFrame df2 包含列 'b'
    df2 = DataFrame({"b": [3, 4.5]})

    # 调用 DataFrame 的 merge 方法，根据索引进行合并
    result = df.merge(df2, left_index=True, right_index=True)

    # 断言：验证 'result' 的 'a' 列是否与 'df' 的 'a' 列共享内存
    assert np.shares_memory(get_array(df, "a"), get_array(result, "a"))
    # 断言：验证 'result' 的 'b' 列是否与 'df2' 的 'b' 列共享内存
    assert np.shares_memory(get_array(df2, "b"), get_array(result, "b"))


# 定义测试函数 test_join_on_key，用于测试数据框按索引键进行连接操作
def test_join_on_key():
    # 创建索引为 ['a', 'b', 'c'] 的索引对象 df_index
    df_index = Index(["a", "b", "c"], name="key")

    # 创建 DataFrame df1 包含列 'a'，并按 df_index 进行索引
    df1 = DataFrame({"a": [1, 2, 3]}, index=df_index.copy(deep=True))
    # 创建 DataFrame df2 包含列 'b'，并按 df_index 进行索引
    df2 = DataFrame({"b": [4, 5, 6]}, index=df_index.copy(deep=True))

    # 复制 df1 和 df2 以备后续比较
    df1_orig = df1.copy()
    df2_orig = df2.copy()

    # 调用 DataFrame 的 join 方法，根据 'key' 列进行连接
    result = df1.join(df2, on="key")

    # 断言：验证 'result' 的 'a' 列是否与 'df1' 的 'a' 列共享内存
    assert np.shares_memory(get_array(result, "a"), get_array(df1, "a"))
    # 断言：验证 'result' 的 'b' 列是否与 'df2' 的 'b' 列共享内存
    assert np.shares_memory(get_array(result, "b"), get_array(df2, "b"))
    # 断言：验证 'result' 的索引是否与 'df1' 的索引共享内存
    assert np.shares_memory(get_array(result.index), get_array(df1.index))
    # 断言：验证 'result' 的索引是否与 'df2' 的索引未共享内存
    assert not np.shares_memory(get_array(result.index), get_array(df2.index))

    # 修改 'result' 数据框的元素，以验证数据框是否发生复制
    result.iloc[0, 0] = 0
    # 断言：验证 'result' 的 'a' 列是否与 'df1' 的 'a' 列未共享内存
    assert not np.shares_memory(get_array(result, "a"), get_array(df1, "a"))
    # 断言：验证 'result' 的 'b' 列是否与 'df2' 的 'b' 列共享内存
    assert np.shares_memory(get_array(result, "b"), get_array(df2, "b"))

    # 修改 'result' 数据框的元素，以验证数据框是否发生复制
    result.iloc[0, 1] = 0
    # 断言：验证 'result' 的 'b' 列是否与 'df2' 的 'b' 列未共享内存
    assert not np.shares_memory(get_array(result, "b"), get_array(df2, "b"))

    # 使用测试工具 tm 进行断言：验证 df1 和 df2 是否与其初始副本相等
    tm.assert_frame_equal(df1, df1_orig)
    tm.assert_frame_equal(df2, df2_orig)


# 定义测试函数 test_join_multiple_dataframes_on_key，用于测试多个数据框按索引键进行连接操作
def test_join_multiple_dataframes_on_key():
    # 创建索引为 ['a', 'b', 'c'] 的索引对象 df_index
    df_index = Index(["a", "b", "c"], name="key")

    # 创建 DataFrame df1 包含列 'a'，并按 df_index 进行索引
    df1 = DataFrame({"a": [1, 2, 3]}, index=df_index.copy(deep=True))
    # 创建包含多个 DataFrame 的列表 dfs_list，每个 DataFrame 包含列 'b' 和 'c'，并按 df_index 进行索引
    dfs_list = [
        DataFrame({"b": [4, 5, 6]}, index=df_index.copy(deep=True)),
        DataFrame({"c": [7, 8, 9]}, index=df_index.copy(deep=True)),
    ]

    # 复制 df1 和 dfs_list 中的每个 DataFrame 以备后续比较
    # 断言结果数组的 "b" 列和第一个数据帧的 "b" 列共享内存
    assert np.shares_memory(get_array(result, "b"), get_array(dfs_list[0], "b"))
    # 断言结果数组的 "c" 列和第二个数据帧的 "c" 列共享内存
    assert np.shares_memory(get_array(result, "c"), get_array(dfs_list[1], "c"))
    # 断言结果数组的索引和第一个数据帧的索引共享内存
    assert np.shares_memory(get_array(result.index), get_array(df1.index))
    # 断言结果数组的索引和第一个数据帧列表中的第一个数据帧的索引不共享内存
    assert not np.shares_memory(get_array(result.index), get_array(dfs_list[0].index))
    # 断言结果数组的索引和第一个数据帧列表中的第二个数据帧的索引不共享内存
    assert not np.shares_memory(get_array(result.index), get_array(dfs_list[1].index))

    # 将结果数组的第一行第一列元素设为 0
    result.iloc[0, 0] = 0
    # 断言结果数组的 "a" 列和原始数据帧 df1 的 "a" 列不共享内存
    assert not np.shares_memory(get_array(result, "a"), get_array(df1, "a"))
    # 断言结果数组的 "b" 列和第一个数据帧的 "b" 列共享内存
    assert np.shares_memory(get_array(result, "b"), get_array(dfs_list[0], "b"))
    # 断言结果数组的 "c" 列和第二个数据帧的 "c" 列共享内存
    assert np.shares_memory(get_array(result, "c"), get_array(dfs_list[1], "c"))

    # 将结果数组的第一行第二列元素设为 0
    result.iloc[0, 1] = 0
    # 断言结果数组的 "b" 列和第一个数据帧的 "b" 列不共享内存
    assert not np.shares_memory(get_array(result, "b"), get_array(dfs_list[0], "b"))
    # 断言结果数组的 "c" 列和第二个数据帧的 "c" 列共享内存
    assert np.shares_memory(get_array(result, "c"), get_array(dfs_list[1], "c"))

    # 将结果数组的第一行第三列元素设为 0
    result.iloc[0, 2] = 0
    # 断言结果数组的 "c" 列和第二个数据帧的 "c" 列不共享内存
    assert not np.shares_memory(get_array(result, "c"), get_array(dfs_list[1], "c"))

    # 使用测试框架断言 df1 与其原始版本 df1_orig 相等
    tm.assert_frame_equal(df1, df1_orig)
    # 使用测试框架逐一断言每个数据帧与其对应的原始版本在数据上的相等性
    for df, df_orig in zip(dfs_list, dfs_list_orig):
        tm.assert_frame_equal(df, df_orig)
```