# `D:\src\scipysrc\pandas\pandas\tests\indexes\multi\test_drop.py`

```
import numpy as np  # 导入 NumPy 库，用于处理数值数据
import pytest  # 导入 pytest 库，用于编写和运行测试用例

import pandas as pd  # 导入 Pandas 库，并用 pd 别名表示
from pandas import (  # 从 Pandas 库中导入 Index 和 MultiIndex 类
    Index,
    MultiIndex,
)
import pandas._testing as tm  # 导入 Pandas 内部的测试工具模块


def test_drop(idx):
    # 删除指定的索引元组，返回新的索引对象 dropped
    dropped = idx.drop([("foo", "two"), ("qux", "one")])

    # 创建 MultiIndex 对象，包含指定的元组列表 index
    index = MultiIndex.from_tuples([("foo", "two"), ("qux", "one")])
    # 删除指定的索引对象 index，返回新的索引对象 dropped2
    dropped2 = idx.drop(index)

    # 预期的索引对象，根据索引位置选择特定行
    expected = idx[[0, 2, 3, 5]]
    # 断言 dropped 和 expected 相等
    tm.assert_index_equal(dropped, expected)
    tm.assert_index_equal(dropped2, expected)

    # 删除指定的索引标签 ["bar"]，返回新的索引对象 dropped
    dropped = idx.drop(["bar"])
    # 预期的索引对象，根据索引位置选择特定行
    expected = idx[[0, 1, 3, 4, 5]]
    tm.assert_index_equal(dropped, expected)

    # 删除指定的索引标签 "foo"，返回新的索引对象 dropped
    dropped = idx.drop("foo")
    # 预期的索引对象，根据索引位置选择特定行
    expected = idx[[2, 3, 4, 5]]
    tm.assert_index_equal(dropped, expected)

    # 创建包含单个元组的 MultiIndex 对象 index
    index = MultiIndex.from_tuples([("bar", "two")])
    # 使用 pytest 检查是否引发 KeyError 异常，匹配特定的错误消息模式
    with pytest.raises(KeyError, match=r"^\('bar', 'two'\)$"):
        idx.drop([("bar", "two")])
    # 使用 pytest 检查是否引发 KeyError 异常，匹配特定的错误消息模式
    with pytest.raises(KeyError, match=r"^\('bar', 'two'\)$"):
        idx.drop(index)
    # 使用 pytest 检查是否引发 KeyError 异常，匹配特定的错误消息模式
    with pytest.raises(KeyError, match=r"^'two'$"):
        idx.drop(["foo", "two"])

    # 创建包含多个元组的 MultiIndex 对象 mixed_index
    mixed_index = MultiIndex.from_tuples([("qux", "one"), ("bar", "two")])
    # 使用 pytest 检查是否引发 KeyError 异常，匹配特定的错误消息模式
    with pytest.raises(KeyError, match=r"^\('bar', 'two'\)$"):
        idx.drop(mixed_index)

    # 删除指定的索引对象 index，使用错误处理模式 errors="ignore"
    dropped = idx.drop(index, errors="ignore")
    # 预期的索引对象，根据索引位置选择特定行
    expected = idx[[0, 1, 2, 3, 4, 5]]
    tm.assert_index_equal(dropped, expected)

    # 删除指定的索引对象 mixed_index，使用错误处理模式 errors="ignore"
    dropped = idx.drop(mixed_index, errors="ignore")
    # 预期的索引对象，根据索引位置选择特定行
    expected = idx[[0, 1, 2, 3, 5]]
    tm.assert_index_equal(dropped, expected)

    # 删除指定的索引标签 ["foo", "two"]，使用错误处理模式 errors="ignore"
    dropped = idx.drop(["foo", "two"], errors="ignore")
    # 预期的索引对象，根据索引位置选择特定行
    expected = idx[[2, 3, 4, 5]]
    tm.assert_index_equal(dropped, expected)

    # 删除混合部分和完整索引删除列表，返回新的索引对象 dropped
    dropped = idx.drop(["foo", ("qux", "one")])
    # 预期的索引对象，根据索引位置选择特定行
    expected = idx[[2, 3, 5]]
    tm.assert_index_equal(dropped, expected)

    # 删除混合部分和完整索引删除列表，使用错误处理模式 errors="ignore"
    mixed_index = ["foo", ("qux", "one"), "two"]
    # 使用 pytest 检查是否引发 KeyError 异常，匹配特定的错误消息模式
    with pytest.raises(KeyError, match=r"^'two'$"):
        idx.drop(mixed_index)
    # 删除混合部分和完整索引删除列表 mixed_index，使用错误处理模式 errors="ignore"
    dropped = idx.drop(mixed_index, errors="ignore")
    # 预期的索引对象，根据索引位置选择特定行
    expected = idx[[2, 3, 5]]
    tm.assert_index_equal(dropped, expected)


def test_droplevel_with_names(idx):
    # 从 idx 中选择包含 "foo" 的索引对象 index
    index = idx[idx.get_loc("foo")]
    # 删除第一级别的索引，返回新的索引对象 dropped
    dropped = index.droplevel(0)
    # 断言 dropped 的名称为 "second"
    assert dropped.name == "second"

    # 创建 MultiIndex 对象 index，包含指定的 levels 和 codes
    index = MultiIndex(
        levels=[Index(range(4)), Index(range(4)), Index(range(4))],
        codes=[
            np.array([0, 0, 1, 2, 2, 2, 3, 3]),
            np.array([0, 1, 0, 0, 0, 1, 0, 1]),
            np.array([1, 0, 1, 1, 0, 0, 1, 0]),
        ],
        names=["one", "two", "three"],
    )
    # 删除第一级别的索引，返回新的索引对象 dropped
    dropped = index.droplevel(0)
    # 断言 dropped 的名称为 ("two", "three")
    assert dropped.names == ("two", "three")

    # 删除指定名称的索引级别 "two"，返回新的索引对象 dropped
    dropped = index.droplevel("two")
    # 预期的索引对象，删除第二级别的索引
    expected = index.droplevel(1)
    assert dropped.equals(expected)


def test_droplevel_list():
    # 创建一个多层索引对象 `index`，包含三个层级，每个层级都是一个 Index 对象，每个 Index 对象包含 4 个元素
    index = MultiIndex(
        levels=[Index(range(4)), Index(range(4)), Index(range(4))],
        codes=[
            np.array([0, 0, 1, 2, 2, 2, 3, 3]),
            np.array([0, 1, 0, 0, 0, 1, 0, 1]),
            np.array([1, 0, 1, 1, 0, 0, 1, 0]),
        ],
        names=["one", "two", "three"],
    )
    
    # 从 `index` 中选择前两行，并移除层级 "three" 和 "one"，得到新的 MultiIndex 对象 `dropped`
    dropped = index[:2].droplevel(["three", "one"])
    # 创建预期的 MultiIndex 对象 `expected`，与 `dropped` 相同
    expected = index[:2].droplevel(2).droplevel(0)
    # 断言 `dropped` 和 `expected` 是否相等
    assert dropped.equals(expected)
    
    # 从 `index` 中选择前两行，并尝试移除空的层级列表，预期结果应与 `index[:2]` 相同
    dropped = index[:2].droplevel([])
    # 创建预期的 MultiIndex 对象 `expected`，与 `index[:2]` 相同
    expected = index[:2]
    # 断言 `dropped` 和 `expected` 是否相等
    assert dropped.equals(expected)
    
    # 定义一个错误消息，指示试图从具有 3 层级的索引中移除 3 层级是不允许的
    msg = (
        "Cannot remove 3 levels from an index with 3 levels: "
        "at least one level must be left"
    )
    # 使用 pytest 检查是否会引发 ValueError，并检查错误消息是否匹配 `msg`
    with pytest.raises(ValueError, match=msg):
        index[:2].droplevel(["one", "two", "three"])
    
    # 使用 pytest 检查是否会引发 KeyError，并检查错误消息是否包含 "'Level four not found'"
    with pytest.raises(KeyError, match="'Level four not found'"):
        index[:2].droplevel(["one", "four"])
# 定义一个测试函数，用于验证在非字典序排序的情况下执行`drop`操作
def test_drop_not_lexsorted(performance_warning):
    # GH 12078

    # 定义一个多级索引的字典序排序版本
    tuples = [("a", ""), ("b1", "c1"), ("b2", "c2")]
    lexsorted_mi = MultiIndex.from_tuples(tuples, names=["b", "c"])
    assert lexsorted_mi._is_lexsorted()  # 断言确保字典序排序

    # 定义一个非字典序排序的版本
    df = pd.DataFrame(
        columns=["a", "b", "c", "d"], data=[[1, "b1", "c1", 3], [1, "b2", "c2", 4]]
    )
    df = df.pivot_table(index="a", columns=["b", "c"], values="d")
    df = df.reset_index()
    not_lexsorted_mi = df.columns  # 获取列索引
    assert not not_lexsorted_mi._is_lexsorted()  # 断言确保非字典序排序

    # 比较两个版本的结果
    tm.assert_index_equal(lexsorted_mi, not_lexsorted_mi)

    # 断言在性能警告下执行`drop`操作后的结果
    with tm.assert_produces_warning(performance_warning):
        tm.assert_index_equal(lexsorted_mi.drop("a"), not_lexsorted_mi.drop("a"))


# 测试删除具有 NaN 的索引元素
def test_drop_with_nan_in_index(nulls_fixture):
    # GH#18853
    mi = MultiIndex.from_tuples([("blah", nulls_fixture)], names=["name", "date"])
    msg = r"labels \[Timestamp\('2001-01-01 00:00:00'\)\] not found in level"
    with pytest.raises(KeyError, match=msg):
        mi.drop(pd.Timestamp("2001"), level="date")


# 忽略非单调重复警告的情况下测试`drop`操作
@pytest.mark.filterwarnings("ignore::pandas.errors.PerformanceWarning")
def test_drop_with_non_monotonic_duplicates():
    # GH#33494
    mi = MultiIndex.from_tuples([(1, 2), (2, 3), (1, 2)])
    result = mi.drop((1, 2))
    expected = MultiIndex.from_tuples([(2, 3)])
    tm.assert_index_equal(result, expected)


# 测试在多级索引中删除部分缺失元素的情况
def test_single_level_drop_partially_missing_elements():
    # GH 37820

    mi = MultiIndex.from_tuples([(1, 2), (2, 2), (3, 2)])
    msg = r"labels \[4\] not found in level"
    with pytest.raises(KeyError, match=msg):
        mi.drop(4, level=0)
    with pytest.raises(KeyError, match=msg):
        mi.drop([1, 4], level=0)
    msg = r"labels \[nan\] not found in level"
    with pytest.raises(KeyError, match=msg):
        mi.drop([np.nan], level=0)
    with pytest.raises(KeyError, match=msg):
        mi.drop([np.nan, 1, 2, 3], level=0)

    mi = MultiIndex.from_tuples([(np.nan, 1), (1, 2)])
    msg = r"labels \['a'\] not found in level"
    with pytest.raises(KeyError, match=msg):
        mi.drop([np.nan, 1, "a"], level=0)


# 测试删除多级索引中的一个级别
def test_droplevel_multiindex_one_level():
    # GH#37208
    index = MultiIndex.from_tuples([(2,)], names=("b",))
    result = index.droplevel([])
    expected = Index([2], name="b")
    tm.assert_index_equal(result, expected)
```