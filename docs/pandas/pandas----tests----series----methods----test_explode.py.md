# `D:\src\scipysrc\pandas\pandas\tests\series\methods\test_explode.py`

```
# 导入必要的库：numpy用于数值计算，pytest用于单元测试
import numpy as np
import pytest

# 导入pandas库，并导入其测试模块_pandas._testing as tm
import pandas as pd
import pandas._testing as tm


# 定义单元测试函数test_basic，测试pd.Series对象的explode方法
def test_basic():
    # 创建一个pd.Series对象s，包含多种数据类型的元素，包括列表、NaN、空列表和元组
    s = pd.Series([[0, 1, 2], np.nan, [], (3, 4)], index=list("abcd"), name="foo")
    # 对s调用explode方法，将列表展开为单独的元素
    result = s.explode()
    # 创建期望的pd.Series对象expected，展开后的预期结果
    expected = pd.Series(
        [0, 1, 2, np.nan, np.nan, 3, 4], index=list("aaabcdd"), dtype=object, name="foo"
    )
    # 使用_pandas._testing模块的assert_series_equal函数断言result和expected相等
    tm.assert_series_equal(result, expected)


# 定义单元测试函数test_mixed_type，测试pd.Series对象包含不同类型元素的explode方法
def test_mixed_type():
    # 创建一个pd.Series对象s，包含不同类型的元素，包括列表、NaN、None、数组和另一个pd.Series对象
    s = pd.Series(
        [[0, 1, 2], np.nan, None, np.array([]), pd.Series(["a", "b"])], name="foo"
    )
    # 对s调用explode方法，展开其中的列表
    result = s.explode()
    # 创建期望的pd.Series对象expected，展开后的预期结果
    expected = pd.Series(
        [0, 1, 2, np.nan, None, np.nan, "a", "b"],
        index=[0, 0, 0, 1, 2, 3, 4, 4],
        dtype=object,
        name="foo",
    )
    # 使用_pandas._testing模块的assert_series_equal函数断言result和expected相等
    tm.assert_series_equal(result, expected)


# 定义单元测试函数test_empty，测试pd.Series对象为空时的explode方法
def test_empty():
    # 创建一个空的pd.Series对象s
    s = pd.Series(dtype=object)
    # 对s调用explode方法，预期结果与原对象相同
    result = s.explode()
    expected = s.copy()
    # 使用_pandas._testing模块的assert_series_equal函数断言result和expected相等
    tm.assert_series_equal(result, expected)


# 定义单元测试函数test_nested_lists，测试pd.Series对象包含嵌套列表的explode方法
def test_nested_lists():
    # 创建一个pd.Series对象s，包含嵌套的列表元素
    s = pd.Series([[[1, 2, 3]], [1, 2], 1])
    # 对s调用explode方法，展开其中的嵌套列表
    result = s.explode()
    # 创建期望的pd.Series对象expected，展开后的预期结果
    expected = pd.Series([[1, 2, 3], 1, 2, 1], index=[0, 1, 1, 2])
    # 使用_pandas._testing模块的assert_series_equal函数断言result和expected相等
    tm.assert_series_equal(result, expected)


# 定义单元测试函数test_multi_index，测试pd.Series对象包含多级索引的explode方法
def test_multi_index():
    # 创建一个pd.Series对象s，包含多级索引和列表元素
    s = pd.Series(
        [[0, 1, 2], np.nan, [], (3, 4)],
        name="foo",
        index=pd.MultiIndex.from_product([list("ab"), range(2)], names=["foo", "bar"]),
    )
    # 对s调用explode方法，展开其中的列表
    result = s.explode()
    # 创建期望的pd.Series对象expected，展开后的预期结果
    index = pd.MultiIndex.from_tuples(
        [("a", 0), ("a", 0), ("a", 0), ("a", 1), ("b", 0), ("b", 1), ("b", 1)],
        names=["foo", "bar"],
    )
    expected = pd.Series(
        [0, 1, 2, np.nan, np.nan, 3, 4], index=index, dtype=object, name="foo"
    )
    # 使用_pandas._testing模块的assert_series_equal函数断言result和expected相等
    tm.assert_series_equal(result, expected)


# 定义单元测试函数test_large，测试pd.Series对象包含大量元素的explode方法
def test_large():
    # 创建一个包含多个range对象的pd.Series对象s，并对其调用两次explode方法，展开其中的所有元素
    s = pd.Series([range(256)]).explode()
    result = s.explode()
    # 使用_pandas._testing模块的assert_series_equal函数断言result和s相等
    tm.assert_series_equal(result, s)


# 定义单元测试函数test_invert_array，测试pd.DataFrame对象应用函数后使用explode方法
def test_invert_array():
    # 创建一个包含日期范围的pd.DataFrame对象df
    df = pd.DataFrame({"a": pd.date_range("20190101", periods=3, tz="UTC")})
    # 对df应用lambda函数，返回每行的日期数组，然后对结果调用explode方法
    listify = df.apply(lambda x: x.array, axis=1)
    result = listify.explode()
    # 使用_pandas._testing模块的assert_series_equal函数断言result和df["a"].rename()相等
    tm.assert_series_equal(result, df["a"].rename())


# 使用pytest的参数化装饰器标记，定义单元测试函数test_non_object_dtype，测试不同数据类型的pd.Series对象的explode方法
@pytest.mark.parametrize(
    "data", [[1, 2, 3], pd.date_range("2019", periods=3, tz="UTC")]
)
def test_non_object_dtype(data):
    # 根据参数data创建一个pd.Series对象ser，并对其调用explode方法
    ser = pd.Series(data)
    result = ser.explode()
    # 使用_pandas._testing模块的assert_series_equal函数断言result和ser相等
    tm.assert_series_equal(result, ser)


# 定义单元测试函数test_typical_usecase，测试pd.DataFrame对象的typical use case
def test_typical_usecase():
    # 创建一个包含字符串列和整数列的pd.DataFrame对象df
    df = pd.DataFrame(
        [{"var1": "a,b,c", "var2": 1}, {"var1": "d,e,f", "var2": 2}],
        columns=["var1", "var2"],
    )
    # 对df的var1列应用split方法和explode方法，展开以逗号分隔的字符串
    exploded = df.var1.str.split(",").explode()
    # 将展开后的结果与df["var2"]列连接，创建一个新的pd.DataFrame对象result
    result = df[["var2"]].join(exploded)
    # 创建期望的pd.DataFrame对象expected，包含展开后的预期结果
    expected = pd.DataFrame(
        {"var2": [1, 1, 1, 2, 2, 2], "var1": list("abcdef")},
        columns=["var2", "var1"],
        index=[0, 0, 0, 1, 1, 1],
    )
    # 使用_pandas._testing模块的assert_frame_equal函数断言result和expected相等
    tm.assert_frame_equal(result, expected)


# 定义单元测试函数test_nested_EA，测试pd.Series对象包含嵌套日期范围的explode方法
def test_nested_EA():
    # 创建一个包含嵌套日期范围的pd.Series对象s，并对其调用explode方法
    s = pd.Series(
        [
            pd.date_range("20170101", periods=3, tz="UTC"),
            pd.date_range("20170104", periods=3, tz="UTC"),
        ]
    )
    result = s.explode()
    # 创建一个预期的 Pandas Series 对象，包含从 2017-01-01 开始，以 UTC 时区为基准的日期范围，索引为 [0, 0, 0, 1, 1, 1]
    expected = pd.Series(
        pd.date_range("20170101", periods=6, tz="UTC"), index=[0, 0, 0, 1, 1, 1]
    )
    # 使用 Pandas Testing 模块中的 assert_series_equal 函数比较 result 和 expected 两个 Series 对象是否相等
    tm.assert_series_equal(result, expected)
# 测试处理重复索引的情况
def test_duplicate_index():
    # GH 28005
    # 创建一个包含列表的序列，指定索引为 [0, 0]
    s = pd.Series([[1, 2], [3, 4]], index=[0, 0])
    # 对序列执行 explode 操作，展开列表中的元素
    result = s.explode()
    # 预期的展开后的序列，包含每个元素及其对应的重复索引
    expected = pd.Series([1, 2, 3, 4], index=[0, 0, 0, 0], dtype=object)
    # 使用测试工具库检查 result 是否等于 expected
    tm.assert_series_equal(result, expected)


# 测试忽略索引的情况
def test_ignore_index():
    # GH 34932
    # 创建一个包含列表的序列，没有指定索引
    s = pd.Series([[1, 2], [3, 4]])
    # 对序列执行 explode 操作，并忽略原有的索引
    result = s.explode(ignore_index=True)
    # 预期的展开后的序列，包含每个元素并重新生成索引
    expected = pd.Series([1, 2, 3, 4], index=[0, 1, 2, 3], dtype=object)
    # 使用测试工具库检查 result 是否等于 expected
    tm.assert_series_equal(result, expected)


# 测试处理集合类型的情况
def test_explode_sets():
    # https://github.com/pandas-dev/pandas/issues/35614
    # 创建一个包含集合的序列，指定索引为 [1]
    s = pd.Series([{"a", "b", "c"}], index=[1])
    # 对序列执行 explode 操作，并对结果进行排序
    result = s.explode().sort_values()
    # 预期的展开后的序列，每个元素为集合中的单个元素，保持原有索引
    expected = pd.Series(["a", "b", "c"], index=[1, 1, 1])
    # 使用测试工具库检查 result 是否等于 expected
    tm.assert_series_equal(result, expected)


# 测试处理标量值的情况，并可以忽略索引
def test_explode_scalars_can_ignore_index():
    # https://github.com/pandas-dev/pandas/issues/40487
    # 创建一个包含标量值的序列，指定字符串索引
    s = pd.Series([1, 2, 3], index=["a", "b", "c"])
    # 对序列执行 explode 操作，并忽略原有的索引
    result = s.explode(ignore_index=True)
    # 预期的展开后的序列，不再保留原有索引
    expected = pd.Series([1, 2, 3])
    # 使用测试工具库检查 result 是否等于 expected
    tm.assert_series_equal(result, expected)


# 使用参数化测试，测试处理 pyarrow 列表类型的情况
@pytest.mark.parametrize("ignore_index", [True, False])
def test_explode_pyarrow_list_type(ignore_index):
    # GH 53602
    # 导入 pytest 并引入 pyarrow 库，如不存在则跳过测试
    pa = pytest.importorskip("pyarrow")

    # 创建包含 pyarrow 列表类型数据的序列
    data = [
        [None, None],
        [1],
        [],
        [2, 3],
        None,
    ]
    # 创建具有指定 dtype 的序列
    ser = pd.Series(data, dtype=pd.ArrowDtype(pa.list_(pa.int64())))
    # 对序列执行 explode 操作，并根据参数决定是否忽略索引
    result = ser.explode(ignore_index=ignore_index)
    # 预期的展开后的序列，根据参数是否保留原有索引
    expected = pd.Series(
        data=[None, None, 1, None, 2, 3, None],
        index=None if ignore_index else [0, 0, 1, 2, 3, 3, 4],
        dtype=pd.ArrowDtype(pa.int64()),
    )
    # 使用测试工具库检查 result 是否等于 expected
    tm.assert_series_equal(result, expected)


# 使用参数化测试，测试处理 pyarrow 非列表类型的情况
@pytest.mark.parametrize("ignore_index", [True, False])
def test_explode_pyarrow_non_list_type(ignore_index):
    # 导入 pytest 并引入 pyarrow 库，如不存在则跳过测试
    pa = pytest.importorskip("pyarrow")
    # 创建包含 pyarrow 非列表类型数据的序列
    data = [1, 2, 3]
    # 创建具有指定 dtype 的序列
    ser = pd.Series(data, dtype=pd.ArrowDtype(pa.int64()))
    # 对序列执行 explode 操作，并根据参数决定是否忽略索引
    result = ser.explode(ignore_index=ignore_index)
    # 预期的展开后的序列，根据参数是否保留原有索引
    expected = pd.Series([1, 2, 3], dtype="int64[pyarrow]", index=[0, 1, 2])
    # 使用测试工具库检查 result 是否等于 expected
    tm.assert_series_equal(result, expected)
```