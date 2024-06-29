# `D:\src\scipysrc\pandas\pandas\tests\series\accessors\test_list_accessor.py`

```
import re  # 导入正则表达式模块

import pytest  # 导入 pytest 测试框架

from pandas import (  # 从 pandas 库中导入 ArrowDtype 和 Series 类
    ArrowDtype,
    Series,
)
import pandas._testing as tm  # 导入 pandas 内部测试模块 tm

pa = pytest.importorskip("pyarrow")  # 导入并检查是否存在 pyarrow 库

from pandas.compat import pa_version_under11p0  # 从 pandas 兼容模块中导入 pyarrow 版本比较函数


@pytest.mark.parametrize(  # 使用 pytest 的参数化装饰器定义参数化测试
    "list_dtype",
    (
        pa.list_(pa.int64()),  # 定义包含 int64 元素的列表数据类型
        pa.list_(pa.int64(), list_size=3),  # 定义包含固定大小的 int64 元素列表数据类型
        pa.large_list(pa.int64()),  # 定义包含大列表的 int64 元素数据类型
    ),
)
def test_list_getitem(list_dtype):
    ser = Series(  # 创建 Series 对象
        [[1, 2, 3], [4, None, 5], None],  # 设置 Series 的初始数据
        dtype=ArrowDtype(list_dtype),  # 指定 Series 的数据类型为 ArrowDtype
    )
    actual = ser.list[1]  # 获取 Series 的 list 属性的第二个元素
    expected = Series([2, None, None], dtype="int64[pyarrow]")  # 预期的结果 Series 对象
    tm.assert_series_equal(actual, expected)  # 使用 pandas._testing 模块的 assert_series_equal 函数比较结果


def test_list_getitem_index():
    # GH 58425
    ser = Series(  # 创建 Series 对象
        [[1, 2, 3], [4, None, 5], None],  # 设置 Series 的初始数据
        dtype=ArrowDtype(pa.list_(pa.int64())),  # 指定 Series 的数据类型为包含 int64 元素的列表
        index=[1, 3, 7],  # 设置 Series 的索引
    )
    actual = ser.list[1]  # 获取 Series 的 list 属性的第二个元素
    expected = Series([2, None, None], dtype="int64[pyarrow]", index=[1, 3, 7])  # 预期的结果 Series 对象
    tm.assert_series_equal(actual, expected)  # 使用 pandas._testing 模块的 assert_series_equal 函数比较结果


def test_list_getitem_slice():
    ser = Series(  # 创建 Series 对象
        [[1, 2, 3], [4, None, 5], None],  # 设置 Series 的初始数据
        dtype=ArrowDtype(pa.list_(pa.int64())),  # 指定 Series 的数据类型为包含 int64 元素的列表
        index=[1, 3, 7],  # 设置 Series 的索引
    )
    if pa_version_under11p0:  # 如果 pyarrow 的版本低于 11.0
        with pytest.raises(  # 抛出 pytest 异常
            NotImplementedError, match="List slice not supported by pyarrow "
        ):
            ser.list[1:None:None]  # 尝试使用列表切片操作
    else:
        actual = ser.list[1:None:None]  # 获取 Series 的 list 属性的切片结果
        expected = Series(  # 预期的结果 Series 对象
            [[2, 3], [None, 5], None],  # 切片后的数据内容
            dtype=ArrowDtype(pa.list_(pa.int64())),  # 数据类型为包含 int64 元素的列表
            index=[1, 3, 7],  # 设置 Series 的索引
        )
        tm.assert_series_equal(actual, expected)  # 使用 pandas._testing 模块的 assert_series_equal 函数比较结果


def test_list_len():
    ser = Series(  # 创建 Series 对象
        [[1, 2, 3], [4, None], None],  # 设置 Series 的初始数据
        dtype=ArrowDtype(pa.list_(pa.int64())),  # 指定 Series 的数据类型为包含 int64 元素的列表
    )
    actual = ser.list.len()  # 获取 Series 的 list 属性的长度
    expected = Series([3, 2, None], dtype=ArrowDtype(pa.int32()))  # 预期的结果 Series 对象
    tm.assert_series_equal(actual, expected)  # 使用 pandas._testing 模块的 assert_series_equal 函数比较结果


def test_list_flatten():
    ser = Series(  # 创建 Series 对象
        [[1, 2, 3], None, [4, None], [], [7, 8]],  # 设置 Series 的初始数据
        dtype=ArrowDtype(pa.list_(pa.int64())),  # 指定 Series 的数据类型为包含 int64 元素的列表
    )
    actual = ser.list.flatten()  # 获取 Series 的 list 属性的扁平化结果
    expected = Series(  # 预期的结果 Series 对象
        [1, 2, 3, 4, None, 7, 8],  # 扁平化后的数据内容
        dtype=ArrowDtype(pa.int64()),  # 数据类型为 int64
        index=[0, 0, 0, 2, 2, 4, 4],  # 设置 Series 的索引
    )
    tm.assert_series_equal(actual, expected)  # 使用 pandas._testing 模块的 assert_series_equal 函数比较结果


def test_list_getitem_slice_invalid():
    ser = Series(  # 创建 Series 对象
        [[1, 2, 3], [4, None, 5], None],  # 设置 Series 的初始数据
        dtype=ArrowDtype(pa.list_(pa.int64())),  # 指定 Series 的数据类型为包含 int64 元素的列表
    )
    if pa_version_under11p0:  # 如果 pyarrow 的版本低于 11.0
        with pytest.raises(  # 抛出 pytest 异常
            NotImplementedError, match="List slice not supported by pyarrow "
        ):
            ser.list[1:None:0]  # 尝试使用无效的列表切片操作
    else:
        with pytest.raises(  # 抛出 pytest 异常
            pa.lib.ArrowInvalid, match=re.escape("`step` must be >= 1")
        ):
            ser.list[1:None:0]  # 尝试使用无效的列表切片操作


def test_list_accessor_non_list_dtype():
    ser = Series(  # 创建 Series 对象
        [1, 2, 4],  # 设置 Series 的初始数据
        dtype=ArrowDtype(pa.int64()),  # 指定 Series 的数据类型为 int64
    )
    with pytest.raises(  # 抛出 pytest 异常
        AttributeError,  # 属性错误异常
        match=re.escape(  # 匹配错误消息的正则表达式
            "Can only use the '.list' accessor with 'list[pyarrow]' dtype, "
            "not int64[pyarrow]."
        ),
    ):
        # 循环：对于序列 `ser` 中的每一个元素
        ser.list[1:None:0]
# 使用 pytest.mark.parametrize 装饰器来参数化测试函数，测试不同的 list 数据类型
@pytest.mark.parametrize(
    "list_dtype",
    (
        pa.list_(pa.int64()),                     # 定义包含 int64 的列表数据类型
        pa.list_(pa.int64(), list_size=3),        # 定义包含 int64 且大小为 3 的列表数据类型
        pa.large_list(pa.int64()),                # 定义大列表数据类型，包含 int64
    ),
)
def test_list_getitem_invalid_index(list_dtype):
    # 创建包含不同数据类型的 Series 对象
    ser = Series(
        [[1, 2, 3], [4, None, 5], None],
        dtype=ArrowDtype(list_dtype),
    )
    # 使用 pytest.raises 捕获 ArrowInvalid 异常，检查是否正确处理越界索引
    with pytest.raises(pa.lib.ArrowInvalid, match="Index -1 is out of bounds"):
        ser.list[-1]                            # 尝试访问列表的负索引，应抛出异常
    with pytest.raises(pa.lib.ArrowInvalid, match="Index 5 is out of bounds"):
        ser.list[5]                             # 尝试访问列表的超出范围的正索引，应抛出异常
    with pytest.raises(ValueError, match="key must be an int or slice, got str"):
        ser.list["abc"]                         # 尝试使用字符串作为键访问列表，应抛出异常


def test_list_accessor_not_iterable():
    # 创建包含列表数据类型的 Series 对象
    ser = Series(
        [[1, 2, 3], [4, None], None],
        dtype=ArrowDtype(pa.list_(pa.int64())),
    )
    # 使用 pytest.raises 捕获 TypeError 异常，检查是否正确处理非可迭代的情况
    with pytest.raises(TypeError, match="'ListAccessor' object is not iterable"):
        iter(ser.list)                          # 尝试迭代列表访问器，应抛出异常
```