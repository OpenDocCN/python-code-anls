# `D:\src\scipysrc\pandas\pandas\tests\indexes\multi\test_names.py`

```
import pytest  # 导入 pytest 库

import pandas as pd  # 导入 pandas 库，并命名为 pd
from pandas import MultiIndex  # 从 pandas 中导入 MultiIndex 类
import pandas._testing as tm  # 导入 pandas 内部测试模块


def check_level_names(index, names):
    assert [level.name for level in index.levels] == list(names)  # 断言检查索引中每个层级的名称是否与给定列表 names 相符


def test_slice_keep_name():
    x = MultiIndex.from_tuples([("a", "b"), (1, 2), ("c", "d")], names=["x", "y"])  # 创建 MultiIndex 对象 x，指定名称为 "x" 和 "y"
    assert x[1:].names == x.names  # 断言检查切片后的 x 对象的名称与原始 x 对象是否相同


def test_index_name_retained():
    # GH9857
    result = pd.DataFrame({"x": [1, 2, 6], "y": [2, 2, 8], "z": [-5, 0, 5]})
    result = result.set_index("z")  # 将结果 DataFrame 按列 "z" 设置为索引
    result.loc[10] = [9, 10]  # 向结果 DataFrame 添加新行
    df_expected = pd.DataFrame(
        {"x": [1, 2, 6, 9], "y": [2, 2, 8, 10], "z": [-5, 0, 5, 10]}
    )
    df_expected = df_expected.set_index("z")  # 将期望的 DataFrame 按列 "z" 设置为索引
    tm.assert_frame_equal(result, df_expected)  # 使用测试模块中的函数比较 result 和 df_expected 的内容是否相同


def test_changing_names(idx):
    assert [level.name for level in idx.levels] == ["first", "second"]  # 断言检查 idx 对象的每个层级名称是否为 ["first", "second"]

    view = idx.view()  # 创建 idx 的视图对象 view
    copy = idx.copy()  # 复制 idx 对象为 copy
    shallow_copy = idx._view()  # 创建 idx 的浅复制对象 shallow_copy

    # 修改名称不应该改变对象上的层级名称
    new_names = [name + "a" for name in idx.names]  # 创建新的名称列表，每个名称加上后缀 "a"
    idx.names = new_names  # 将 idx 对象的名称设置为新的名称列表
    check_level_names(idx, ["firsta", "seconda"])  # 断言检查 idx 对象的层级名称是否为 ["firsta", "seconda"]

    # 并且不应该影响到复制的对象
    check_level_names(view, ["first", "second"])  # 断言检查视图对象 view 的层级名称是否为 ["first", "second"]
    check_level_names(copy, ["first", "second"])  # 断言检查复制对象 copy 的层级名称是否为 ["first", "second"]
    check_level_names(shallow_copy, ["first", "second"])  # 断言检查浅复制对象 shallow_copy 的层级名称是否为 ["first", "second"]

    # 复制的对象也不应该改变原始对象
    shallow_copy.names = [name + "c" for name in shallow_copy.names]  # 将浅复制对象 shallow_copy 的名称列表中每个名称加上后缀 "c"
    check_level_names(idx, ["firsta", "seconda"])  # 断言检查 idx 对象的层级名称是否为 ["firsta", "seconda"]


def test_take_preserve_name(idx):
    taken = idx.take([3, 0, 1])  # 获取 idx 对象中指定索引的子集，返回新的 MultiIndex 对象 taken
    assert taken.names == idx.names  # 断言检查 taken 对象的名称是否与 idx 对象的名称相同


def test_copy_names():
    # 检查将 "names" 参数添加到复制过程中是否被遵循
    # GH14302
    multi_idx = MultiIndex.from_tuples([(1, 2), (3, 4)], names=["MyName1", "MyName2"])  # 创建 MultiIndex 对象 multi_idx，并指定名称为 "MyName1" 和 "MyName2"
    multi_idx1 = multi_idx.copy()  # 复制 multi_idx 对象为 multi_idx1

    assert multi_idx.equals(multi_idx1)  # 断言检查 multi_idx 和 multi_idx1 是否相等
    assert multi_idx.names == ["MyName1", "MyName2"]  # 断言检查 multi_idx 的名称是否为 ["MyName1", "MyName2"]
    assert multi_idx1.names == ["MyName1", "MyName2"]  # 断言检查 multi_idx1 的名称是否为 ["MyName1", "MyName2"]

    multi_idx2 = multi_idx.copy(names=["NewName1", "NewName2"])  # 以新的名称 ["NewName1", "NewName2"] 复制 multi_idx 对象为 multi_idx2

    assert multi_idx.equals(multi_idx2)  # 断言检查 multi_idx 和 multi_idx2 是否相等
    assert multi_idx.names == ["MyName1", "MyName2"]  # 断言检查 multi_idx 的名称是否为 ["MyName1", "MyName2"]
    assert multi_idx2.names == ["NewName1", "NewName2"]  # 断言检查 multi_idx2 的名称是否为 ["NewName1", "NewName2"]

    multi_idx3 = multi_idx.copy(name=["NewName1", "NewName2"])  # 尝试以新的名称 ["NewName1", "NewName2"] 复制 multi_idx 对象为 multi_idx3

    assert multi_idx.equals(multi_idx3)  # 断言检查 multi_idx 和 multi_idx3 是否相等
    assert multi_idx.names == ["MyName1", "MyName2"]  # 断言检查 multi_idx 的名称是否为 ["MyName1", "MyName2"]
    assert multi_idx3.names == ["NewName1", "NewName2"]  # 断言检查 multi_idx3 的名称是否为 ["NewName1", "NewName2"]

    # gh-35592
    with pytest.raises(ValueError, match="Length of new names must be 2, got 1"):  # 断言检查是否引发 ValueError 异常，异常信息需包含 "Length of new names must be 2, got 1"
        multi_idx.copy(names=["mario"])  # 尝试以长度为 1 的名称列表 ["mario"] 复制 multi_idx 对象

    with pytest.raises(TypeError, match="MultiIndex.name must be a hashable type"):  # 断言检查是否引发 TypeError 异常，异常信息需包含 "MultiIndex.name must be a hashable type"
        multi_idx.copy(names=[["mario"], ["luigi"]])  # 尝试以非可哈希类型的名称列表复制 multi_idx 对象


def test_names(idx):
    # 在设置中分配了名称
    assert idx.names == ["first", "second"]  # 断言检查 idx 对象的名称是否为 ["first", "second"]
    level_names = [level.name for level in idx.levels]  # 获取 idx 对象每个层级的名称列表
    assert level_names == idx.names  # 断言检查 idx 对象每个层级的名称列表是否与 idx 对象的名称列表相同

    # 尝试在现有名称上设置错误的名称
    index = idx
    with pytest.raises(ValueError, match="^Length of names"):  # 断言检查是否引发 ValueError 异常，异常信息需以 "Length of names" 开头
        setattr(index, "names", list(index.names) + ["third"])  # 尝试将现有名称列表加上 "third" 设置为 index 对象的名称
    # 使用 pytest 的断言来验证是否会引发 ValueError 异常，并检查异常消息是否以 "Length of names" 开头
    with pytest.raises(ValueError, match="^Length of names"):
        # 设置 index 对象的 "names" 属性为一个空列表，长度不符合预期，应该引发异常
        setattr(index, "names", [])

    # 使用错误的 names 初始化 MultiIndex，预期应该引发异常
    major_axis, minor_axis = idx.levels
    major_codes, minor_codes = idx.codes
    with pytest.raises(ValueError, match="^Length of names"):
        # 使用不匹配的 names 长度初始化 MultiIndex 对象
        MultiIndex(
            levels=[major_axis, minor_axis],
            codes=[major_codes, minor_codes],
            names=["first"],
        )
    with pytest.raises(ValueError, match="^Length of names"):
        # 使用不匹配的 names 长度初始化 MultiIndex 对象
        MultiIndex(
            levels=[major_axis, minor_axis],
            codes=[major_codes, minor_codes],
            names=["first", "second", "third"],
        )

    # 虽然给 index 分配了 names，但并未将它们传递给 levels
    index.names = ["a", "b"]
    # 检查 index 的 levels 是否包含了正确的 names
    level_names = [level.name for level in index.levels]
    # 断言 index 的 levels 中的 names 是否符合预期
    assert level_names == ["a", "b"]


这段代码主要进行了以下操作：

1. 使用 `pytest.raises` 断言捕获了设置 `index.names` 时可能引发的 `ValueError` 异常，并验证异常消息。
2. 初始化 `MultiIndex` 对象时，使用了不匹配长度的 `names` 参数，期望引发异常。
3. 将 `index.names` 设置为指定的 names，并检查是否正确传递给了 `index.levels`。
4. 使用 `assert` 断言验证 `index.levels` 中的 `names` 是否符合预期。
def test_duplicate_level_names_access_raises(idx):
    # 定义一个测试函数，用于测试重复级别名称的访问引发异常的情况
    # 设置索引对象的名称列表为 ["foo", "foo"]
    idx.names = ["foo", "foo"]
    # 使用 pytest 检查是否引发 ValueError 异常，并匹配错误信息 "name foo occurs multiple times"
    with pytest.raises(ValueError, match="name foo occurs multiple times"):
        # 调用索引对象的内部方法 _get_level_number("foo")
        idx._get_level_number("foo")


def test_get_names_from_levels():
    # 定义一个测试函数，用于测试从级别中获取名称的功能
    # 创建一个 MultiIndex 对象，包含两个级别的产品组合：[["a"], [1, 2]]，并设置级别名称为 ["a", "b"]
    idx = MultiIndex.from_product([["a"], [1, 2]], names=["a", "b"])

    # 断言第一个级别的名称是否为 "a"
    assert idx.levels[0].name == "a"
    # 断言第二个级别的名称是否为 "b"
    assert idx.levels[1].name == "b"


def test_setting_names_from_levels_raises():
    # 定义一个测试函数，用于测试设置级别名称时引发异常的情况
    # 创建一个 MultiIndex 对象，包含两个级别的产品组合：[["a"], [1, 2]]，并设置级别名称为 ["a", "b"]
    idx = MultiIndex.from_product([["a"], [1, 2]], names=["a", "b"])
    # 使用 pytest 检查是否引发 RuntimeError 异常，并匹配错误信息 "set_names"
    with pytest.raises(RuntimeError, match="set_names"):
        # 尝试修改第一个级别的名称为 "foo"
        idx.levels[0].name = "foo"

    with pytest.raises(RuntimeError, match="set_names"):
        # 尝试修改第二个级别的名称为 "foo"
        idx.levels[1].name = "foo"

    # 创建一个新的 Series 对象，索引使用第一个级别的索引
    new = pd.Series(1, index=idx.levels[0])
    with pytest.raises(RuntimeError, match="set_names"):
        # 尝试修改新 Series 的索引名称为 "bar"
        new.index.name = "bar"

    # 断言 pd.Index._no_setting_name 和 pd.RangeIndex._no_setting_name 都为 False
    assert pd.Index._no_setting_name is False
    assert pd.RangeIndex._no_setting_name is False


@pytest.mark.parametrize("func", ["rename", "set_names"])
@pytest.mark.parametrize(
    "rename_dict, exp_names",
    [
        ({"x": "z"}, ["z", "y", "z"]),
        ({"x": "z", "y": "x"}, ["z", "x", "z"]),
        ({"y": "z"}, ["x", "z", "x"]),
        ({}, ["x", "y", "x"]),
        ({"z": "a"}, ["x", "y", "x"]),
        ({"y": "z", "a": "b"}, ["x", "z", "x"]),
    ],
)
def test_name_mi_with_dict_like_duplicate_names(func, rename_dict, exp_names):
    # GH#20421
    # 定义一个测试函数，用于测试使用类似字典的方式重命名 MultiIndex 对象时处理重复名称的情况
    # 创建一个 MultiIndex 对象，包含三个级别的数组 [[1, 2], [3, 4], [5, 6]]，并设置级别名称为 ["x", "y", "x"]
    mi = MultiIndex.from_arrays([[1, 2], [3, 4], [5, 6]], names=["x", "y", "x"])
    # 调用 mi 对象的 rename 或 set_names 方法，根据参数 func 指定的操作
    result = getattr(mi, func)(rename_dict)
    # 创建一个期望的 MultiIndex 对象，设置期望的级别名称为 exp_names
    expected = MultiIndex.from_arrays([[1, 2], [3, 4], [5, 6]], names=exp_names)
    # 使用 tm.assert_index_equal 检查 result 和 expected 是否相等
    tm.assert_index_equal(result, expected)


@pytest.mark.parametrize("func", ["rename", "set_names"])
@pytest.mark.parametrize(
    "rename_dict, exp_names",
    [
        ({"x": "z"}, ["z", "y"]),
        ({"x": "z", "y": "x"}, ["z", "x"]),
        ({"a": "z"}, ["x", "y"]),
        ({}, ["x", "y"]),
    ],
)
def test_name_mi_with_dict_like(func, rename_dict, exp_names):
    # GH#20421
    # 定义一个测试函数，用于测试使用类似字典的方式重命名 MultiIndex 对象的级别名称
    # 创建一个 MultiIndex 对象，包含两个级别的数组 [[1, 2], [3, 4]]，并设置级别名称为 ["x", "y"]
    mi = MultiIndex.from_arrays([[1, 2], [3, 4]], names=["x", "y"])
    # 调用 mi 对象的 rename 或 set_names 方法，根据参数 func 指定的操作
    result = getattr(mi, func)(rename_dict)
    # 创建一个期望的 MultiIndex 对象，设置期望的级别名称为 exp_names
    expected = MultiIndex.from_arrays([[1, 2], [3, 4]], names=exp_names)
    # 使用 tm.assert_index_equal 检查 result 和 expected 是否相等
    tm.assert_index_equal(result, expected)


def test_index_name_with_dict_like_raising():
    # GH#20421
    # 定义一个测试函数，用于测试在单级索引对象上设置名称时引发异常的情况
    ix = pd.Index([1, 2])
    msg = "Can only pass dict-like as `names` for MultiIndex."
    # 使用 pytest 检查是否引发 TypeError 异常，并匹配错误信息 "Can only pass dict-like as `names` for MultiIndex."
    with pytest.raises(TypeError, match=msg):
        # 尝试使用 set_names 方法设置单级索引对象的名称为 {"x": "z"}
        ix.set_names({"x": "z"})


def test_multiindex_name_and_level_raising():
    # GH#20421
    # 定义一个测试函数，用于测试在设置 MultiIndex 对象的名称和级别时引发异常的情况
    # 创建一个 MultiIndex 对象，包含两个级别的数组 [[1, 2], [3, 4]]，并设置级别名称为 ["x", "y"]
    mi = MultiIndex.from_arrays([[1, 2], [3, 4]], names=["x", "y"])
    # 使用 pytest 检查是否引发 TypeError 异常，并匹配错误信息 "Can not pass level for dictlike `names`."
    with pytest.raises(TypeError, match="Can not pass level for dictlike `names`."):
        # 尝试使用 set_names 方法设置 MultiIndex 对象的名称和级别
        mi.set_names(names={"x": "z"}, level={"x": "z"})
```