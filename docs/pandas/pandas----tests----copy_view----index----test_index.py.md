# `D:\src\scipysrc\pandas\pandas\tests\copy_view\index\test_index.py`

```
# 导入必要的库和模块
import numpy as np
import pytest

# 从 pandas 库中导入特定的类和函数
from pandas import (
    DataFrame,
    Index,
    Series,
)
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array


# 定义一个函数，创建一个 DataFrame，并返回索引和视图
def index_view(index_data):
    df = DataFrame({"a": index_data, "b": 1.5})  # 创建包含索引数据和常数列的 DataFrame
    view = df[:]  # 创建 DataFrame 的视图
    df = df.set_index("a", drop=True)  # 将 DataFrame 设置为以列 'a' 作为索引，丢弃原索引
    idx = df.index  # 获取新的索引对象
    # df = None  # 此行被注释掉，没有实际作用
    return idx, view  # 返回索引对象和原始视图


# 测试函数，验证设置索引后更新列的行为
def test_set_index_update_column():
    df = DataFrame({"a": [1, 2], "b": 1})  # 创建 DataFrame
    df = df.set_index("a", drop=False)  # 设置索引为列 'a'，保留列 'a'
    expected = df.index.copy(deep=True)  # 深拷贝当前索引作为期望值
    df.iloc[0, 0] = 100  # 修改 DataFrame 的值
    tm.assert_index_equal(df.index, expected)  # 断言索引相等


# 测试函数，验证设置索引并丢弃原索引后更新列的行为
def test_set_index_drop_update_column():
    df = DataFrame({"a": [1, 2], "b": 1.5})  # 创建 DataFrame
    view = df[:]  # 创建 DataFrame 的视图
    df = df.set_index("a", drop=True)  # 设置索引为列 'a'，丢弃原索引
    expected = df.index.copy(deep=True)  # 深拷贝当前索引作为期望值
    view.iloc[0, 0] = 100  # 修改视图的值
    tm.assert_index_equal(df.index, expected)  # 断言索引相等


# 测试函数，验证以 Series 设置索引后更新列的行为
def test_set_index_series():
    df = DataFrame({"a": [1, 2], "b": 1.5})  # 创建 DataFrame
    ser = Series([10, 11])  # 创建 Series
    df = df.set_index(ser)  # 使用 Series 设置索引
    expected = df.index.copy(deep=True)  # 深拷贝当前索引作为期望值
    ser.iloc[0] = 100  # 修改 Series 的值
    tm.assert_index_equal(df.index, expected)  # 断言索引相等


# 测试函数，验证将 Series 分配为索引的行为
def test_assign_index_as_series():
    df = DataFrame({"a": [1, 2], "b": 1.5})  # 创建 DataFrame
    ser = Series([10, 11])  # 创建 Series
    df.index = ser  # 将 Series 分配为 DataFrame 的索引
    expected = df.index.copy(deep=True)  # 深拷贝当前索引作为期望值
    ser.iloc[0] = 100  # 修改 Series 的值
    tm.assert_index_equal(df.index, expected)  # 断言索引相等


# 测试函数，验证将 Index 对象分配为索引的行为
def test_assign_index_as_index():
    df = DataFrame({"a": [1, 2], "b": 1.5})  # 创建 DataFrame
    ser = Series([10, 11])  # 创建 Series
    rhs_index = Index(ser)  # 创建 Index 对象
    df.index = rhs_index  # 将 Index 对象分配为 DataFrame 的索引
    rhs_index = None  # 清除引用，以便释放资源
    expected = df.index.copy(deep=True)  # 深拷贝当前索引作为期望值
    ser.iloc[0] = 100  # 修改 Series 的值
    tm.assert_index_equal(df.index, expected)  # 断言索引相等


# 测试函数，验证从 Series 创建 Index 的行为
def test_index_from_series():
    ser = Series([1, 2])  # 创建 Series
    idx = Index(ser)  # 使用 Series 创建 Index
    expected = idx.copy(deep=True)  # 深拷贝当前索引作为期望值
    ser.iloc[0] = 100  # 修改 Series 的值
    tm.assert_index_equal(idx, expected)  # 断言索引相等


# 测试函数，验证从 Series 创建 Index（使用 copy=True）的行为
def test_index_from_series_copy():
    ser = Series([1, 2])  # 创建 Series
    idx = Index(ser, copy=True)  # 使用 Series 创建 Index（使用 copy=True）
    arr = get_array(ser)  # 获取 Series 的数组表示
    ser.iloc[0] = 100  # 修改 Series 的值
    assert np.shares_memory(get_array(ser), arr)  # 断言两个数组共享内存


# 测试函数，验证从 Index 创建 Index 的行为
def test_index_from_index():
    ser = Series([1, 2])  # 创建 Series
    idx = Index(ser)  # 使用 Series 创建 Index
    idx = Index(idx)  # 使用现有的 Index 创建新的 Index
    expected = idx.copy(deep=True)  # 深拷贝当前索引作为期望值
    ser.iloc[0] = 100  # 修改 Series 的值
    tm.assert_index_equal(idx, expected)  # 断言索引相等


# 测试函数，参数化测试不同的 Index 操作函数
@pytest.mark.parametrize(
    "func",
    [
        lambda x: x._shallow_copy(x._values),
        lambda x: x.view(),
        lambda x: x.take([0, 1]),
        lambda x: x.repeat([1, 1]),
        lambda x: x[slice(0, 2)],
        lambda x: x[[0, 1]],
        lambda x: x._getitem_slice(slice(0, 2)),
        lambda x: x.delete([]),
        lambda x: x.rename("b"),
        lambda x: x.astype("Int64", copy=False),
    ],
    ids=[
        "_shallow_copy",
        "view",
        "take",
        "repeat",
        "getitem_slice",
        "getitem_list",
        "_getitem_slice",
        "delete",
        "rename",
        "astype",
    ],
)
def test_index_ops(func, request):
    idx, view_ = index_view([1, 2])  # 调用 index_view 函数获取索引和视图
    expected = idx.copy(deep=True)  # 深拷贝当前索引作为期望值
    # 检查字符串 "astype" 是否在请求的节点规范 ID 中
    if "astype" in request.node.callspec.id:
        # 如果存在，则将期望值 expected 转换为 "Int64" 类型
        expected = expected.astype("Int64")
    
    # 调用函数 func 处理 idx，并更新 idx 的值
    idx = func(idx)
    
    # 将 view_ 的第一行第一列的元素设置为 100
    view_.iloc[0, 0] = 100
    
    # 使用测试工具（tm）来比较 idx 和 expected 的索引，不检查名称
    tm.assert_index_equal(idx, expected, check_names=False)
# 测试函数：用于测试 infer_objects 方法的功能
def test_infer_objects():
    # 调用 index_view 函数，获取索引和视图对象
    idx, view_ = index_view(["a", "b"])
    # 复制 idx 对象，用于后续比较
    expected = idx.copy(deep=True)
    # 调用 infer_objects 方法，尝试推断 idx 对象的数据类型，不进行复制
    idx = idx.infer_objects(copy=False)
    # 修改 view_ 的第一个元素为字符串 "aaaa"
    view_.iloc[0, 0] = "aaaa"
    # 使用 assert_index_equal 方法断言 idx 和 expected 对象相等，忽略名称检查
    tm.assert_index_equal(idx, expected, check_names=False)


# 测试函数：用于测试 index 对象转换为 DataFrame 的功能
def test_index_to_frame():
    # 创建 Index 对象，包含元素 [1, 2, 3]，名称为 "a"
    idx = Index([1, 2, 3], name="a")
    # 复制 idx 对象，用于后续比较
    expected = idx.copy(deep=True)
    # 调用 to_frame 方法，将 idx 转换为 DataFrame 对象 df
    df = idx.to_frame()
    # 使用 get_array 函数获取 df 中名称为 "a" 的数据数组，并与 idx 的值数组共享内存
    assert np.shares_memory(get_array(df, "a"), idx._values)
    # 断言 df 的管理器 _mgr 不含无引用的块
    assert not df._mgr._has_no_reference(0)

    # 修改 df 的第一个元素为 100
    df.iloc[0, 0] = 100
    # 使用 assert_index_equal 方法断言 idx 和 expected 对象相等
    tm.assert_index_equal(idx, expected)


# 测试函数：用于测试 index 对象的 values 属性
def test_index_values():
    # 创建 Index 对象，包含元素 [1, 2, 3]
    idx = Index([1, 2, 3])
    # 获取 idx 的 values 属性
    result = idx.values
    # 断言 result 的可写标志为 False
    assert result.flags.writeable is False
```