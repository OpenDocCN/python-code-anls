# `D:\src\scipysrc\pandas\pandas\tests\indexes\multi\test_copy.py`

```
# 导入需要的模块和函数
from copy import (
    copy,  # 导入 copy 函数用于浅复制
    deepcopy,  # 导入 deepcopy 函数用于深复制
)

import pytest  # 导入 pytest 测试框架

from pandas import MultiIndex  # 导入 MultiIndex 类
import pandas._testing as tm  # 导入 pandas 测试模块

# 定义函数，验证 MultiIndex 对象的复制行为是否符合预期
def assert_multiindex_copied(copy, original):
    # 验证 levels 属性至少是浅复制的
    tm.assert_copy(copy.levels, original.levels)
    # 验证 codes 属性的值几乎相等
    tm.assert_almost_equal(copy.codes, original.codes)

    # labels 属性不关心复制方式，只需确保值相等，但对象不同
    tm.assert_almost_equal(copy.codes, original.codes)
    assert copy.codes is not original.codes

    # names 属性不关心复制方式，只需确保值相等，但对象不同
    assert copy.names == original.names
    assert copy.names is not original.names

    # sortorder 属性应当完全复制
    assert copy.sortorder == original.sortorder


# 测试函数，验证 copy 方法的行为
def test_copy(idx):
    i_copy = idx.copy()

    assert_multiindex_copied(i_copy, idx)


# 测试函数，验证 _view 方法的行为（浅复制）
def test_shallow_copy(idx):
    i_copy = idx._view()

    assert_multiindex_copied(i_copy, idx)


# 测试函数，验证 view 方法的行为
def test_view(idx):
    i_view = idx.view()
    assert_multiindex_copied(i_view, idx)


# 参数化测试，验证 copy 和 deepcopy 方法的行为
@pytest.mark.parametrize("func", [copy, deepcopy])
def test_copy_and_deepcopy(func):
    idx = MultiIndex(
        levels=[["foo", "bar"], ["fizz", "buzz"]],
        codes=[[0, 0, 0, 1], [0, 0, 1, 1]],
        names=["first", "second"],
    )
    idx_copy = func(idx)
    assert idx_copy is not idx  # 确保对象不同
    assert idx_copy.equals(idx)  # 确保内容相同


# 参数化测试，验证 copy 方法的 kwargs 参数行为
@pytest.mark.parametrize("deep", [True, False])
@pytest.mark.parametrize(
    "kwarg, value",
    [
        ("names", ["third", "fourth"]),  # 测试 names 参数
    ],
)
def test_copy_method_kwargs(deep, kwarg, value):
    # gh-12309: 检查 "names" 参数及其他 kwargs 是否被正确应用
    idx = MultiIndex(
        levels=[["foo", "bar"], ["fizz", "buzz"]],
        codes=[[0, 0, 0, 1], [0, 0, 1, 1]],
        names=["first", "second"],
    )
    idx_copy = idx.copy(**{kwarg: value, "deep": deep})
    assert getattr(idx_copy, kwarg) == value  # 验证参数被正确设置


# 测试函数，验证 copy 方法的 deep=False 参数保留对象 ID 的行为
def test_copy_deep_false_retains_id():
    # GH#47878
    idx = MultiIndex(
        levels=[["foo", "bar"], ["fizz", "buzz"]],
        codes=[[0, 0, 0, 1], [0, 0, 1, 1]],
        names=["first", "second"],
    )

    res = idx.copy(deep=False)
    assert res._id is idx._id  # 确保对象 ID 被保留
```