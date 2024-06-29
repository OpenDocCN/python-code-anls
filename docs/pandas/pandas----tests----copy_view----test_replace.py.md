# `D:\src\scipysrc\pandas\pandas\tests\copy_view\test_replace.py`

```
# 导入必要的库
import numpy as np  # 导入 NumPy 库，并使用 np 作为别名
import pytest  # 导入 pytest 库

# 从 pandas 库中导入特定模块
from pandas import (
    Categorical,  # 导入 Categorical 类
    DataFrame,  # 导入 DataFrame 类
)

# 导入 pandas 内部测试工具模块
import pandas._testing as tm

# 导入自定义的数组获取函数
from pandas.tests.copy_view.util import get_array


@pytest.mark.parametrize(
    "replace_kwargs",  # 参数化测试，replace_kwargs 是测试用例的参数
    [
        {"to_replace": {"a": 1, "b": 4}, "value": -1},  # 替换指定值的测试用例
        # 测试 CoW（Copy-on-Write）是否能避免复制未更改的列块
        {"to_replace": {"a": 1}, "value": -1},  # 替换指定值的测试用例
        {"to_replace": {"b": 4}, "value": -1},  # 替换指定值的测试用例
        {"to_replace": {"b": {4: 1}}},  # 替换指定值的测试用例
        # TODO: 在进一步优化中添加这些测试用例
        # 我们需要查看哪些列在掩码中被替换，这可能是昂贵的操作
        # {"to_replace": {"b": 1}},
        # 1
    ],
)
def test_replace(replace_kwargs):
    df = DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": ["foo", "bar", "baz"]})  # 创建测试用 DataFrame
    df_orig = df.copy()  # 复制原始 DataFrame

    df_replaced = df.replace(**replace_kwargs)  # 执行替换操作，并返回替换后的 DataFrame

    # 检查是否共享内存，以确保 CoW 功能正常（对于列 'b'）
    if (df_replaced["b"] == df["b"]).all():
        assert np.shares_memory(get_array(df_replaced, "b"), get_array(df, "b"))
    assert np.shares_memory(get_array(df_replaced, "c"), get_array(df, "c"))

    # 修改挤压后的 DataFrame 的列 'c'，触发该列块的 CoW 操作
    df_replaced.loc[0, "c"] = -1
    assert not np.shares_memory(get_array(df_replaced, "c"), get_array(df, "c"))

    # 如果替换了列 'a' 中的值
    if "a" in replace_kwargs["to_replace"]:
        arr = get_array(df_replaced, "a")
        df_replaced.loc[0, "a"] = 100
        assert np.shares_memory(get_array(df_replaced, "a"), arr)
    tm.assert_frame_equal(df, df_orig)  # 断言修改后的 DataFrame 与原始 DataFrame 相等


def test_replace_regex_inplace_refs():
    df = DataFrame({"a": ["aaa", "bbb"]})  # 创建包含字符串的 DataFrame
    df_orig = df.copy()  # 复制原始 DataFrame
    view = df[:]  # 创建 DataFrame 的视图
    arr = get_array(df, "a")  # 获取列 'a' 的数组表示

    # 执行正则表达式替换，并设置 inplace=True
    df.replace(to_replace=r"^a.*$", value="new", inplace=True, regex=True)

    # 检查是否不共享内存，以确保 CoW 功能正常
    assert not np.shares_memory(arr, get_array(df, "a"))
    assert df._mgr._has_no_reference(0)
    tm.assert_frame_equal(view, df_orig)  # 断言视图与原始 DataFrame 相等


def test_replace_regex_inplace():
    df = DataFrame({"a": ["aaa", "bbb"]})  # 创建包含字符串的 DataFrame
    arr = get_array(df, "a")  # 获取列 'a' 的数组表示

    # 执行正则表达式替换，并设置 inplace=True
    df.replace(to_replace=r"^a.*$", value="new", inplace=True, regex=True)

    # 检查是否不共享内存，以确保 CoW 功能正常
    assert df._mgr._has_no_reference(0)
    assert np.shares_memory(arr, get_array(df, "a"))

    df_orig = df.copy()  # 复制原始 DataFrame
    df2 = df.replace(to_replace=r"^b.*$", value="new", regex=True)  # 执行另一次替换操作
    tm.assert_frame_equal(df_orig, df)  # 断言修改后的 DataFrame 与原始 DataFrame 相等
    assert not np.shares_memory(get_array(df2, "a"), get_array(df, "a"))


def test_replace_regex_inplace_no_op():
    df = DataFrame({"a": [1, 2]})  # 创建包含整数的 DataFrame
    arr = get_array(df, "a")  # 获取列 'a' 的数组表示

    # 执行正则表达式替换，并设置 inplace=True
    df.replace(to_replace=r"^a.$", value="new", inplace=True, regex=True)

    # 检查是否不共享内存，以确保 CoW 功能正常
    assert df._mgr._has_no_reference(0)
    assert np.shares_memory(arr, get_array(df, "a"))

    df_orig = df.copy()  # 复制原始 DataFrame
    df2 = df.replace(to_replace=r"^x.$", value="new", regex=True)  # 执行另一次替换操作
    tm.assert_frame_equal(df_orig, df)  # 断言修改后的 DataFrame 与原始 DataFrame 相等
    assert np.shares_memory(get_array(df2, "a"), get_array(df, "a"))


def test_replace_mask_all_false_second_block():
    df = DataFrame({"a": [1.5, 2, 3], "b": 100.5, "c": 1, "d": 2})  # 创建包含多种数据类型的 DataFrame
    df_orig = df.copy()  # 复制原始 DataFrame
    # 使用 Pandas 的 replace 方法，将 DataFrame df 中所有的 1.5 替换为 55.5
    df2 = df.replace(to_replace=1.5, value=55.5)

    # 断言：通过块分割可以避免复制 b
    assert np.shares_memory(get_array(df, "c"), get_array(df2, "c"))
    # 断言：确保 get_array 函数获取的 "a" 列数据不共享内存
    assert not np.shares_memory(get_array(df, "a"), get_array(df2, "a"))

    # 在 df2 中的第一行第 "c" 列设置值为 1
    df2.loc[0, "c"] = 1
    # 断言：确保 df 保持不变
    tm.assert_frame_equal(df, df_orig)  # 原始数据没有改变

    # 断言：确保 get_array 函数获取的 "c" 列数据不共享内存
    assert not np.shares_memory(get_array(df, "c"), get_array(df2, "c"))
    # 断言：确保 get_array 函数获取的 "d" 列数据共享内存
    assert np.shares_memory(get_array(df, "d"), get_array(df2, "d"))
# 定义一个函数，用于测试 DataFrame 的替换操作，强制转换单列
def test_replace_coerce_single_column():
    # 创建一个 DataFrame 对象，包含两列："a" 列为浮点数，"b" 列为浮点数 100.5
    df = DataFrame({"a": [1.5, 2, 3], "b": 100.5})
    # 备份原始 DataFrame
    df_orig = df.copy()

    # 使用 replace 方法将所有出现的 1.5 替换为字符串 "a"，生成新的 DataFrame df2
    df2 = df.replace(to_replace=1.5, value="a")
    # 断言 "b" 列的数据与原始 DataFrame 的 "b" 列数据共享内存
    assert np.shares_memory(get_array(df, "b"), get_array(df2, "b"))
    # 断言 "a" 列的数据与 df2 的 "a" 列数据不共享内存
    assert not np.shares_memory(get_array(df, "a"), get_array(df2, "a"))

    # 修改 df2 中的数据，不影响原始 DataFrame
    df2.loc[0, "b"] = 0.5
    # 使用 assert_frame_equal 断言 df 与 df_orig 完全相等，即原始 DataFrame 未被修改
    tm.assert_frame_equal(df, df_orig)  # Original is unchanged
    # 断言 "b" 列的数据与 df2 的 "b" 列数据不共享内存
    assert not np.shares_memory(get_array(df, "b"), get_array(df2, "b"))


# 定义一个函数，用于测试 replace 方法中 to_replace 参数类型错误的情况
def test_replace_to_replace_wrong_dtype():
    # 创建一个 DataFrame 对象，包含两列："a" 列为浮点数，"b" 列为浮点数 100.5
    df = DataFrame({"a": [1.5, 2, 3], "b": 100.5})
    # 备份原始 DataFrame
    df_orig = df.copy()

    # 使用 replace 方法将所有出现的字符串 "xxx" 替换为浮点数 1.5，生成新的 DataFrame df2
    df2 = df.replace(to_replace="xxx", value=1.5)

    # 断言 "b" 列的数据与原始 DataFrame 的 "b" 列数据共享内存
    assert np.shares_memory(get_array(df, "b"), get_array(df2, "b"))
    # 断言 "a" 列的数据与 df2 的 "a" 列数据共享内存
    assert np.shares_memory(get_array(df, "a"), get_array(df2, "a"))

    # 修改 df2 中的数据，不影响原始 DataFrame
    df2.loc[0, "b"] = 0.5
    # 使用 assert_frame_equal 断言 df 与 df_orig 完全相等，即原始 DataFrame 未被修改
    tm.assert_frame_equal(df, df_orig)  # Original is unchanged
    # 断言 "b" 列的数据与 df2 的 "b" 列数据不共享内存
    assert not np.shares_memory(get_array(df, "b"), get_array(df2, "b"))


# 定义一个函数，用于测试 replace 方法中处理分类数据的情况
def test_replace_list_categorical():
    # 创建一个 DataFrame 对象，包含一列 "a"，数据为分类类型 ["a", "b", "c"]
    df = DataFrame({"a": ["a", "b", "c"]}, dtype="category")
    # 获取 "a" 列的数据的底层数组
    arr = get_array(df, "a")

    # 使用 replace 方法将所有出现的 "c" 替换为 "a"，并将结果应用于原 DataFrame，即在原地替换
    df.replace(["c"], value="a", inplace=True)
    # 断言底层数据的编码（codes）与原 DataFrame 的 "a" 列的编码共享内存
    assert np.shares_memory(arr.codes, get_array(df, "a").codes)
    # 断言 DataFrame 的管理器中不包含任何引用
    assert df._mgr._has_no_reference(0)

    # 备份原始 DataFrame
    df_orig = df.copy()
    # 使用 replace 方法将所有出现的 "b" 替换为 "a"
    df.replace(["b"], value="a")
    # 应用 lambda 函数将 df 的 "a" 列分类数据重命名，将 "b" 替换为 "d"，生成新的 DataFrame df2
    df2 = df.apply(lambda x: x.cat.rename_categories({"b": "d"}))
    # 断言底层数据的编码（codes）与 df2 的 "a" 列的编码不共享内存
    assert not np.shares_memory(arr.codes, get_array(df2, "a").codes)

    # 使用 assert_frame_equal 断言 df 与 df_orig 完全相等
    tm.assert_frame_equal(df, df_orig)


# 定义一个函数，测试 replace 方法中处理分类数据且在原地替换的情况
def test_replace_list_inplace_refs_categorical():
    # 创建一个 DataFrame 对象，包含一列 "a"，数据为分类类型 ["a", "b", "c"]
    df = DataFrame({"a": ["a", "b", "c"]}, dtype="category")
    # 创建一个 df 的视图 view
    view = df[:]
    # 备份原始 DataFrame
    df_orig = df.copy()
    # 使用 replace 方法将所有出现的 "c" 替换为 "a"，并将结果应用于原 DataFrame，即在原地替换
    df.replace(["c"], value="a", inplace=True)
    # 使用 assert_frame_equal 断言 df_orig 与 view 完全相等
    tm.assert_frame_equal(df_orig, view)


# 使用 pytest 的参数化测试，测试 replace 方法中的原地替换操作
@pytest.mark.parametrize("to_replace", [1.5, [1.5], []])
def test_replace_inplace(to_replace):
    # 创建一个 DataFrame 对象，包含一列 "a"，数据为 [1.5, 2, 3]
    df = DataFrame({"a": [1.5, 2, 3]})
    # 获取 "a" 列的底层数组
    arr_a = get_array(df, "a")
    # 备份原始 DataFrame
    df_orig = df.copy()

    # 使用 replace 方法将所有出现的 1.5 替换为 15.5，并将结果应用于原 DataFrame，即在原地替换
    df.replace(to_replace=1.5, value=15.5, inplace=True)

    # 断言 "a" 列的数据与底层数组 arr_a 共享内存
    assert np.shares_memory(get_array(df, "a"), arr_a)
    # 断言 DataFrame 的管理器中不包含任何引用
    assert df._mgr._has_no_reference(0)


# 使用 pytest 的参数化测试，测试 replace 方法中的原地替换操作且引用的检查
@pytest.mark.parametrize("to_replace", [1.5, [1.5]])
def test_replace_inplace_reference(to_replace):
    # 创建一个 DataFrame 对象，包含一列 "a"，数据为 [1.5, 2, 3]
    df = DataFrame({"a": [1.5, 2, 3]})
    # 获取 "a" 列的底层数组
    arr_a = get_array(df, "a")
    # 创建一个 df 的视图 view
    view = df[:]
    # 备份原始 DataFrame
    df_orig = df.copy()

    # 使用 replace 方法将所有出现的 to_replace 替换为 15.5，并将结果应用于原 DataFrame，即在原地替换
    df.replace(to_replace=to_replace, value=15.5, inplace=True)

    # 断言 "a" 列的数据与底层数组 arr_a 不共享内存
    assert not np.shares_memory(get_array(df, "a"), arr_a)
    # 断言 DataFrame 的管理器中不包含任何引用
    assert df._mgr._has_no_reference(0)
    # 断言 view 的管理器中不包含任何引用
    assert view._mgr._has_no_reference(0)


# 使用 pytest 的参数化测试，测试 replace 方法中处理非数值类型的原地替换操作
@pytest.mark.parametrize("to_replace", ["a", 100.5])
def test_replace_inplace_reference_no_op(to_replace):
    # 创建一个 DataFrame 对象，包
    # 复制DataFrame，并赋值给df_orig变量
    df_orig = df.copy()
    
    # 使用get_array函数从DataFrame df中获取列"a"的数组，并赋值给arr_a变量
    arr_a = get_array(df, "a")
    
    # 使用切片操作创建DataFrame的视图，并赋值给view变量
    view = df[:]
    
    # 在DataFrame df中将to_replace的值替换为1，并原地修改（inplace=True）
    df.replace(to_replace=to_replace, value=1, inplace=True)
    
    # 断言：确保修改后的df的列"a"的数组与arr_a的codes没有共享内存
    assert not np.shares_memory(get_array(df, "a").codes, arr_a.codes)
    
    # 断言：确保df的数据结构的内部管理器（_mgr）中的第一个块没有任何引用
    assert df._mgr._has_no_reference(0)
    
    # 断言：确保view的数据结构的内部管理器（_mgr）中的第一个块没有任何引用
    assert view._mgr._has_no_reference(0)
    
    # 使用测试工具tm.assert_frame_equal断言：确保view和df_orig是相等的DataFrame
    tm.assert_frame_equal(view, df_orig)
# 定义测试函数，用于测试在分类数据上进行替换操作是否正确
def test_replace_categorical_inplace():
    # 创建包含一个分类列的数据框
    df = DataFrame({"a": Categorical([1, 2, 3])})
    # 获取列'a'对应的底层数组
    arr_a = get_array(df, "a")
    # 在原地将所有值为1的元素替换为1（实际上没有变化）
    df.replace(to_replace=1, value=1, inplace=True)

    # 断言：底层数组的内存地址相同
    assert np.shares_memory(get_array(df, "a").codes, arr_a.codes)
    # 断言：数据框的第一个管理器中不包含引用
    assert df._mgr._has_no_reference(0)

    # 期望的数据框，没有发生任何变化
    expected = DataFrame({"a": Categorical([1, 2, 3])})
    # 断言：数据框和期望的数据框相等
    tm.assert_frame_equal(df, expected)


# 定义测试函数，用于测试在分类数据上进行替换操作是否正确（非原地版本）
def test_replace_categorical():
    # 创建包含一个分类列的数据框
    df = DataFrame({"a": Categorical([1, 2, 3])})
    # 备份原始数据框
    df_orig = df.copy()
    # 使用非原地方式将所有值为1的元素替换为1（实际上没有变化）
    df2 = df.replace(to_replace=1, value=1)

    # 断言：数据框的第一个管理器中不包含引用
    assert df._mgr._has_no_reference(0)
    # 断言：新数据框的第一个管理器中不包含引用
    assert df2._mgr._has_no_reference(0)
    # 断言：底层数组的内存地址不同
    assert not np.shares_memory(get_array(df, "a").codes, get_array(df2, "a").codes)
    # 断言：数据框和原始数据框相等
    tm.assert_frame_equal(df, df_orig)

    # 获取新数据框列'a'对应的底层数组
    arr_a = get_array(df2, "a").codes
    # 修改新数据框的第一行第一列的值为2.0
    df2.iloc[0, 0] = 2.0
    # 断言：底层数组的内存地址与之前获取的底层数组相同
    assert np.shares_memory(get_array(df2, "a").codes, arr_a)


# 使用pytest的参数化装饰器定义测试函数，用于测试where和mask方法的屏蔽操作是否正确（原地版本）
@pytest.mark.parametrize("method", ["where", "mask"])
def test_masking_inplace(method):
    # 创建包含一个数值列的数据框
    df = DataFrame({"a": [1.5, 2, 3]})
    # 备份原始数据框
    df_orig = df.copy()
    # 获取列'a'对应的底层数组
    arr_a = get_array(df, "a")
    # 备份数据框视图
    view = df[:]

    # 获取方法名对应的方法对象
    method = getattr(df, method)
    # 在原地对列'a'进行屏蔽操作，将大于1.6的元素替换为-1
    method(df["a"] > 1.6, -1, inplace=True)

    # 断言：底层数组的内存地址不同
    assert not np.shares_memory(get_array(df, "a"), arr_a)
    # 断言：数据框的第一个管理器中不包含引用
    assert df._mgr._has_no_reference(0)
    # 断言：数据框视图的第一个管理器中不包含引用
    assert view._mgr._has_no_reference(0)
    # 断言：数据框和原始数据框相等
    tm.assert_frame_equal(view, df_orig)


# 定义测试函数，用于测试替换空列表操作是否正确
def test_replace_empty_list():
    # 创建包含一个整数列的数据框
    df = DataFrame({"a": [1, 2]})

    # 使用原地方式将空列表替换为空列表（实际上没有变化）
    df2 = df.replace([], [])
    # 断言：底层数组的内存地址相同
    assert np.shares_memory(get_array(df2, "a"), get_array(df, "a"))
    # 断言：数据框的第一个管理器中包含引用
    assert not df._mgr._has_no_reference(0)

    # 获取列'a'对应的底层数组
    arr_a = get_array(df, "a")
    # 再次使用原地方式将空列表替换为空列表（实际上没有变化）
    df.replace([], [])
    # 断言：底层数组的内存地址相同
    assert np.shares_memory(get_array(df, "a"), arr_a)
    # 断言：数据框的第一个管理器中包含引用
    assert not df._mgr._has_no_reference(0)
    # 断言：新数据框的第一个管理器中包含引用
    assert not df2._mgr._has_no_reference(0)


# 使用pytest的参数化装饰器定义测试函数，用于测试在对象列表上进行替换操作是否正确（原地版本）
@pytest.mark.parametrize("value", ["d", None])
def test_replace_object_list_inplace(value):
    # 创建包含一个字符串列的数据框
    df = DataFrame({"a": ["a", "b", "c"]})
    # 获取列'a'对应的底层数组
    arr = get_array(df, "a")
    # 在原地将所有值为'c'的元素替换为指定的值（可能为'd'或None）
    df.replace(["c"], value, inplace=True)
    # 断言：底层数组的内存地址相同
    assert np.shares_memory(arr, get_array(df, "a"))
    # 断言：数据框的第一个管理器中不包含引用
    assert df._mgr._has_no_reference(0)


# 定义测试函数，用于测试替换多个元素列表操作是否正确（原地版本）
def test_replace_list_multiple_elements_inplace():
    # 创建包含一个整数列的数据框
    df = DataFrame({"a": [1, 2, 3]})
    # 获取列'a'对应的底层数组
    arr = get_array(df, "a")
    # 在原地将所有值为1或2的元素替换为4
    df.replace([1, 2], 4, inplace=True)
    # 断言：底层数组的内存地址相同
    assert np.shares_memory(arr, get_array(df, "a"))
    # 断言：数据框的第一个管理器中不包含引用
    assert df._mgr._has_no_reference(0)


# 定义测试函数，用于测试替换列表中的None操作是否正确
def test_replace_list_none():
    # 创建包含一个字符串列的数据框
    df = DataFrame({"a": ["a", "b", "c"]})

    # 备份原始数据框
    df_orig = df.copy()
    # 使用替换操作将所有值为'b'的元素替换为None
    df2 = df.replace(["b"], value=None)
    # 断言：数据框和原始数据框相等
    tm.assert_frame_equal(df, df_orig)
    # 断言：底层数组的内存地址不同
    assert not np.shares_memory(get_array(df, "a"), get_array(df2, "a"))


# 定义测试函数，用于测试替换列表中的None操作是否正确（原地版本）
def test_replace_list_none_inplace_refs():
    # 创建包含一个字符串列的数据框
    df = DataFrame({"a": ["a", "b", "c"]})
    # 获取列'a'对应的底层数组
    arr = get_array(df, "a")
    # 备份原始数据框
    df_orig = df.copy()
    # 备份数据框视图
    view = df[:]
    # 使用原地方式将所有值为'a'的元素替换
    # 复制 DataFrame `df` 到 `df_orig`
    df_orig = df.copy()
    
    # 在 DataFrame `df` 中将所有 "a" 列中的值为 10 的项替换为 100
    df.replace({"a": 10}, 100, inplace=True)
    
    # 使用 `get_array` 函数获取视图 `view` 和 DataFrame `df` 中 "a" 列的数组，并断言它们共享内存
    assert np.shares_memory(get_array(view, "a"), get_array(df, "a"))
    
    # 修改 DataFrame `df` 的第一行第一列的值为 100
    df.iloc[0, 0] = 100
    
    # 使用 `tm.assert_frame_equal` 函数断言 `view` 和 `df_orig` 是否相等
    tm.assert_frame_equal(view, df_orig)
# 定义一个测试函数，用于测试不进行任何操作时的列替换功能
def test_replace_columnwise_no_op():
    # 创建一个包含两列的数据框
    df = DataFrame({"a": [1, 2, 3], "b": [1, 2, 3]})
    # 复制原始数据框
    df_orig = df.copy()
    # 对数据框进行替换操作，将"a"列中的10替换为100
    df2 = df.replace({"a": 10}, 100)
    # 断言替换后的"a"列与原始数据框的"a"列共享内存
    assert np.shares_memory(get_array(df2, "a"), get_array(df, "a"))
    # 修改替换后数据框的元素，验证未共享内存的影响
    df2.iloc[0, 0] = 100
    # 断言未共享内存的数据框与原始数据框相等
    tm.assert_frame_equal(df, df_orig)


# 定义一个测试函数，用于测试链式赋值时的替换操作
def test_replace_chained_assignment():
    # 创建一个数据框，包含NaN值
    df = DataFrame({"a": [1, np.nan, 2], "b": 1})
    # 复制原始数据框
    df_orig = df.copy()
    # 使用上下文管理器断言链式赋值错误被抛出
    with tm.raises_chained_assignment_error():
        df["a"].replace(1, 100, inplace=True)
    # 断言未共享内存的数据框与原始数据框相等
    tm.assert_frame_equal(df, df_orig)

    # 使用上下文管理器断言链式赋值错误被抛出
    with tm.raises_chained_assignment_error():
        df[["a"]].replace(1, 100, inplace=True)
    # 断言未共享内存的数据框与原始数据框相等
    tm.assert_frame_equal(df, df_orig)


# 定义一个测试函数，用于测试列表形式的替换操作
def test_replace_listlike():
    # 创建一个包含两列的数据框
    df = DataFrame({"a": [1, 2, 3], "b": [1, 2, 3]})
    # 复制原始数据框
    df_orig = df.copy()

    # 对数据框进行替换操作，将列表中的200和201替换为11
    result = df.replace([200, 201], [11, 11])
    # 断言替换后的"a"列与原始数据框的"a"列共享内存
    assert np.shares_memory(get_array(result, "a"), get_array(df, "a"))

    # 修改替换后数据框的元素，验证未共享内存的影响
    result.iloc[0, 0] = 100
    # 断言未共享内存的数据框与原始数据框相等
    tm.assert_frame_equal(df, df)

    # 对数据框进行替换操作，将列表中的200和2替换为10
    result = df.replace([200, 2], [10, 10])
    # 断言替换后的"a"列与原始数据框的"a"列未共享内存
    assert not np.shares_memory(get_array(df, "a"), get_array(result, "a"))
    # 断言未共享内存的数据框与原始数据框相等
    tm.assert_frame_equal(df, df_orig)


# 定义一个测试函数，用于测试列表形式的替换操作（原地替换）
def test_replace_listlike_inplace():
    # 创建一个包含两列的数据框
    df = DataFrame({"a": [1, 2, 3], "b": [1, 2, 3]})
    # 获取"a"列的数据数组
    arr = get_array(df, "a")
    # 在原地进行替换操作，将列表中的200和2替换为10和11
    df.replace([200, 2], [10, 11], inplace=True)
    # 断言替换后的"a"列与原始数据数组共享内存
    assert np.shares_memory(get_array(df, "a"), arr)

    # 复制一份数据框的视图
    view = df[:]
    # 复制原始数据框
    df_orig = df.copy()
    # 在原地进行替换操作，将列表中的200和3替换为10和11
    df.replace([200, 3], [10, 11], inplace=True)
    # 断言替换后的"a"列与原始数据数组未共享内存
    assert not np.shares_memory(get_array(df, "a"), arr)
    # 断言视图数据框与原始数据框相等
    tm.assert_frame_equal(view, df_orig)
```