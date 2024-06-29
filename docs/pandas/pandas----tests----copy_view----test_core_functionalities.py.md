# `D:\src\scipysrc\pandas\pandas\tests\copy_view\test_core_functionalities.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算
import pytest  # 导入 pytest 库，用于单元测试

from pandas import DataFrame  # 从 pandas 库中导入 DataFrame 类
import pandas._testing as tm  # 导入 pandas 内部测试工具
from pandas.tests.copy_view.util import get_array  # 从指定路径导入 get_array 函数


def test_assigning_to_same_variable_removes_references():
    df = DataFrame({"a": [1, 2, 3]})
    df = df.reset_index()  # 重置 DataFrame 的索引
    assert df._mgr._has_no_reference(1)  # 断言检查是否有引用编号为1的引用
    arr = get_array(df, "a")  # 调用 get_array 函数获取 "a" 列的数组
    df.iloc[0, 1] = 100  # 修改 DataFrame 中的值

    assert np.shares_memory(arr, get_array(df, "a"))  # 断言检查是否共享内存


def test_setitem_dont_track_unnecessary_references():
    df = DataFrame({"a": [1, 2, 3], "b": 1, "c": 1})

    df["b"] = 100  # 修改 "b" 列的值
    arr = get_array(df, "a")  # 获取 "a" 列的数组
    # 在 setitem 中拆分块，如果不小心，新块将相互引用并触发复制
    df.iloc[0, 0] = 100  # 修改 DataFrame 中的值
    assert np.shares_memory(arr, get_array(df, "a"))  # 断言检查是否共享内存


def test_setitem_with_view_copies():
    df = DataFrame({"a": [1, 2, 3], "b": 1, "c": 1})
    view = df[:]  # 创建 DataFrame 的视图
    expected = df.copy()  # 复制 DataFrame

    df["b"] = 100  # 修改 "b" 列的值
    arr = get_array(df, "a")  # 获取 "a" 列的数组
    df.iloc[0, 0] = 100  # 检查是否正确跟踪引用
    assert not np.shares_memory(arr, get_array(df, "a"))  # 断言检查是否不共享内存
    tm.assert_frame_equal(view, expected)  # 使用测试工具断言视图与预期是否相等


def test_setitem_with_view_invalidated_does_not_copy(request):
    df = DataFrame({"a": [1, 2, 3], "b": 1, "c": 1})
    view = df[:]  # 创建 DataFrame 的视图

    df["b"] = 100  # 修改 "b" 列的值
    arr = get_array(df, "a")  # 获取 "a" 列的数组
    view = None  # 标记视图为 None，表示其超出范围
    # TODO(CoW) block gets split because of `df["b"] = 100`
    # which introduces additional refs, even when those of `view` go out of scopes
    df.iloc[0, 0] = 100  # 设置项拆分块。旧块与视图共享数据，新块引用视图和彼此
    # 当视图超出范围时，它们不再与任何其他块共享数据，因此不应触发复制
    mark = pytest.mark.xfail(reason="blk.delete does not track references correctly")  # 使用 pytest 的标记
    request.applymarker(mark)  # 应用标记
    assert np.shares_memory(arr, get_array(df, "a"))  # 断言检查是否共享内存


def test_out_of_scope():
    def func():
        df = DataFrame({"a": [1, 2], "b": 1.5, "c": 1})
        # 创建一些子集
        result = df[["a", "b"]]
        return result

    result = func()
    assert not result._mgr.blocks[0].refs.has_reference()  # 断言检查是否没有引用
    assert not result._mgr.blocks[1].refs.has_reference()  # 断言检查是否没有引用


def test_delete():
    df = DataFrame(
        np.random.default_rng(2).standard_normal((4, 3)), columns=["a", "b", "c"]
    )
    del df["b"]  # 删除 DataFrame 中的 "b" 列
    assert not df._mgr.blocks[0].refs.has_reference()  # 断言检查是否没有引用
    assert not df._mgr.blocks[1].refs.has_reference()  # 断言检查是否没有引用

    df = df[["a"]]  # 重新赋值 DataFrame，仅包含 "a" 列
    assert not df._mgr.blocks[0].refs.has_reference()  # 断言检查是否没有引用


def test_delete_reference():
    df = DataFrame(
        np.random.default_rng(2).standard_normal((4, 3)), columns=["a", "b", "c"]
    )
    x = df[:]
    del df["b"]  # 删除 DataFrame 中的 "b" 列
    assert df._mgr.blocks[0].refs.has_reference()  # 断言检查是否有引用
    assert df._mgr.blocks[1].refs.has_reference()  # 断言检查是否有引用
    assert x._mgr.blocks[0].refs.has_reference()  # 断言检查是否有引用
```