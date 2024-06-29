# `D:\src\scipysrc\pandas\pandas\tests\indexes\multi\test_get_set.py`

```
# 导入所需的库
import numpy as np
import pytest  # 导入 pytest 库

from pandas.compat import PY311  # 从 pandas.compat 中导入 PY311

from pandas.core.dtypes.dtypes import DatetimeTZDtype  # 从 pandas.core.dtypes.dtypes 中导入 DatetimeTZDtype

import pandas as pd  # 导入 pandas 库
from pandas import (  # 从 pandas 中导入 CategoricalIndex 和 MultiIndex
    CategoricalIndex,
    MultiIndex,
)
import pandas._testing as tm  # 导入 pandas._testing 库


def assert_matching(actual, expected, check_dtype=False):
    # 避免尽可能指定内部表示
    assert len(actual) == len(expected)  # 断言 actual 和 expected 的长度相等
    for act, exp in zip(actual, expected):
        act = np.asarray(act)  # 将 act 转换为 numpy 数组
        exp = np.asarray(exp)  # 将 exp 转换为 numpy 数组
        tm.assert_numpy_array_equal(act, exp, check_dtype=check_dtype)  # 使用 tm.assert_numpy_array_equal 断言 act 和 exp 相等


def test_get_level_number_integer(idx):
    idx.names = [1, 0]  # 设置 idx 的名称为 [1, 0]
    assert idx._get_level_number(1) == 0  # 断言 idx 的第 1 级编号为 0
    assert idx._get_level_number(0) == 1  # 断言 idx 的第 0 级编号为 1
    msg = "Too many levels: Index has only 2 levels, not 3"
    with pytest.raises(IndexError, match=msg):  # 使用 pytest 断言抛出 IndexError 异常，并匹配错误消息 msg
        idx._get_level_number(2)
    with pytest.raises(KeyError, match="Level fourth not found"):  # 使用 pytest 断言抛出 KeyError 异常，并匹配错误消息 "Level fourth not found"
        idx._get_level_number("fourth")


def test_get_dtypes(using_infer_string):
    # 测试 MultiIndex.dtypes (# Gh37062)
    idx_multitype = MultiIndex.from_product(
        [[1, 2, 3], ["a", "b", "c"], pd.date_range("20200101", periods=2, tz="UTC")],
        names=["int", "string", "dt"],  # 设置 MultiIndex 的名称为 ["int", "string", "dt"]
    )

    exp = "object" if not using_infer_string else "string"  # 根据 using_infer_string 决定 exp 的值
    expected = pd.Series(
        {
            "int": np.dtype("int64"),  # 设置 "int" 的数据类型为 np.int64
            "string": exp,  # 设置 "string" 的数据类型为 exp
            "dt": DatetimeTZDtype(tz="utc"),  # 设置 "dt" 的数据类型为 DatetimeTZDtype，时区为 UTC
        }
    )
    tm.assert_series_equal(expected, idx_multitype.dtypes)  # 使用 tm.assert_series_equal 断言 expected 和 idx_multitype.dtypes 相等


def test_get_dtypes_no_level_name(using_infer_string):
    # 测试 MultiIndex.dtypes (# GH38580 )
    idx_multitype = MultiIndex.from_product(
        [
            [1, 2, 3],
            ["a", "b", "c"],
            pd.date_range("20200101", periods=2, tz="UTC"),
        ],
    )
    exp = "object" if not using_infer_string else "string"  # 根据 using_infer_string 决定 exp 的值
    expected = pd.Series(
        {
            "level_0": np.dtype("int64"),  # 设置 "level_0" 的数据类型为 np.int64
            "level_1": exp,  # 设置 "level_1" 的数据类型为 exp
            "level_2": DatetimeTZDtype(tz="utc"),  # 设置 "level_2" 的数据类型为 DatetimeTZDtype，时区为 UTC
        }
    )
    tm.assert_series_equal(expected, idx_multitype.dtypes)  # 使用 tm.assert_series_equal 断言 expected 和 idx_multitype.dtypes 相等


def test_get_dtypes_duplicate_level_names(using_infer_string):
    # 测试 MultiIndex.dtypes with non-unique level names (# GH45174)
    result = MultiIndex.from_product(
        [
            [1, 2, 3],
            ["a", "b", "c"],
            pd.date_range("20200101", periods=2, tz="UTC"),
        ],
        names=["A", "A", "A"],  # 设置 MultiIndex 的名称为 ["A", "A", "A"]，非唯一级别名称
    ).dtypes
    exp = "object" if not using_infer_string else "string"  # 根据 using_infer_string 决定 exp 的值
    expected = pd.Series(
        [np.dtype("int64"), exp, DatetimeTZDtype(tz="utc")],  # 设置预期结果的数据类型列表
        index=["A", "A", "A"],  # 设置预期结果的索引为 ["A", "A", "A"]
    )
    tm.assert_series_equal(result, expected)  # 使用 tm.assert_series_equal 断言 result 和 expected 相等


def test_get_level_number_out_of_bounds(multiindex_dataframe_random_data):
    frame = multiindex_dataframe_random_data

    with pytest.raises(IndexError, match="Too many levels"):  # 使用 pytest 断言抛出 IndexError 异常，并匹配错误消息 "Too many levels"
        frame.index._get_level_number(2)  # 访问 frame 的索引对象，并调用 _get_level_number 方法
    # 使用 pytest 的上下文管理器来测试特定的异常情况
    with pytest.raises(IndexError, match="not a valid level number"):
        # 调用 DataFrame 的 index 对象的 _get_level_number 方法，并传入参数 -3
        frame.index._get_level_number(-3)
# 测试设置名称的方法
def test_set_name_methods(idx):
    # 只要这些是同义词，就不需要测试 set_names
    index_names = ["first", "second"]
    # 断言 rename 方法和 set_names 方法是相同的
    assert idx.rename == idx.set_names
    # 创建新的名称列表，为每个名称添加后缀
    new_names = [name + "SUFFIX" for name in index_names]
    # 使用新的名称列表设置索引的名称
    ind = idx.set_names(new_names)
    # 断言原始索引的名称未被修改
    assert idx.names == index_names
    # 断言设置后的索引的名称为新的名称列表
    assert ind.names == new_names
    # 定义错误消息，用于测试名称数量与 MultiIndex 层级数量是否匹配
    msg = "Length of names must match number of levels in MultiIndex"
    # 测试设置名称时超出名称数量的情况是否会引发 ValueError 异常
    with pytest.raises(ValueError, match=msg):
        ind.set_names(new_names + new_names)
    # 创建另一个带有后缀的新名称列表
    new_names2 = [name + "SUFFIX2" for name in new_names]
    # 使用 inplace=True 设置新的名称列表，并返回结果
    res = ind.set_names(new_names2, inplace=True)
    # 断言返回结果为 None
    assert res is None
    # 断言设置后的索引的名称为新的名称列表
    assert ind.names == new_names2

    # 为特定层级设置名称 (# GH7792)
    # 使用新名称列表中的第一个名称为第 0 层级设置名称
    ind = idx.set_names(new_names[0], level=0)
    # 断言原始索引的名称未被修改
    assert idx.names == index_names
    # 断言设置后的索引的名称，第 0 层级为新名称列表的第一个名称，第 1 层级为原始索引的第二个名称
    assert ind.names == [new_names[0], index_names[1]]

    # 使用 inplace=True 为第 0 层级设置新名称，并返回结果
    res = ind.set_names(new_names2[0], level=0, inplace=True)
    # 断言返回结果为 None
    assert res is None
    # 断言设置后的索引的名称，第 0 层级为新名称列表的第一个名称，第 1 层级为原始索引的第二个名称
    assert ind.names == [new_names2[0], index_names[1]]

    # 为多个层级设置名称
    # 使用新的名称列表为第 [0, 1] 层级设置名称
    ind = idx.set_names(new_names, level=[0, 1])
    # 断言原始索引的名称未被修改
    assert idx.names == index_names
    # 断言设置后的索引的名称与新的名称列表相同
    assert ind.names == new_names

    # 使用 inplace=True 为第 [0, 1] 层级设置新名称，并返回结果
    res = ind.set_names(new_names2, level=[0, 1], inplace=True)
    # 断言返回结果为 None
    assert res is None
    # 断言设置后的索引的名称与新的名称列表（带后缀）相同
    assert ind.names == new_names2


# 测试直接设置 levels 和 codes 的方法
def test_set_levels_codes_directly(idx):
    # 直接设置 levels/codes 将引发 AttributeError 异常

    # 获取当前的 levels
    levels = idx.levels
    # 为每个 level 中的元素添加后缀，创建新的 levels 列表
    new_levels = [[lev + "a" for lev in level] for level in levels]

    # 获取当前的 codes
    codes = idx.codes
    major_codes, minor_codes = codes
    # 将 major_codes 中的每个元素加 1 取余数（模 3）
    major_codes = [(x + 1) % 3 for x in major_codes]
    # 将 minor_codes 中的每个元素加 1 取余数（模 1，即不变）
    minor_codes = [(x + 1) % 1 for x in minor_codes]
    # 创建新的 codes 列表
    new_codes = [major_codes, minor_codes]

    # 定义错误消息，用于测试设置 levels 是否会引发 AttributeError 异常
    msg = "Can't set attribute"
    # 测试设置 levels 属性时是否会引发 AttributeError 异常，并匹配错误消息
    with pytest.raises(AttributeError, match=msg):
        idx.levels = new_levels

    # 定义错误消息，用于测试设置 codes 是否会引发 AttributeError 异常
    msg = (
        "property 'codes' of 'MultiIndex' object has no setter"
        if PY311
        else "can't set attribute"
    )
    # 测试设置 codes 属性时是否会引发 AttributeError 异常，并匹配错误消息
    with pytest.raises(AttributeError, match=msg):
        idx.codes = new_codes


# 测试设置 levels 的方法
def test_set_levels(idx):
    # 注意：通常不建议直接像这样直接使用 levels 和 codes
    # 获取当前的 levels
    levels = idx.levels
    # 为每个 level 中的元素添加后缀，创建新的 levels 列表
    new_levels = [[lev + "a" for lev in level] for level in levels]

    # 更改 level，不改变原索引
    # 使用新的 levels 列表设置索引的 levels，返回一个新的索引对象
    ind2 = idx.set_levels(new_levels)
    # 断言新的索引对象的 levels 与新的 levels 列表相匹配
    assert_matching(ind2.levels, new_levels)
    # 断言原始索引的 levels 未被修改
    assert_matching(idx.levels, levels)

    # 更改特定 level，不改变原索引
    # 使用新的 levels[0] 列表设置索引的第 0 层级的 levels，返回一个新的索引对象
    ind2 = idx.set_levels(new_levels[0], level=0)
    # 断言新的索引对象的 levels[0] 与新的 levels[0] 列表相匹配
    assert_matching(ind2.levels, [new_levels[0], levels[1]])
    # 断言原始索引的 levels 未被修改
    assert_matching(idx.levels, levels)

    # 使用新的 levels[1] 列表设置索引的第 1 层级的 levels，返回一个新的索引对象
    ind2 = idx.set_levels(new_levels[1], level=1)
    # 断言新的索引对象的 levels[1] 与新的 levels[1] 列表相匹配
    assert_matching(ind2.levels, [levels[0], new_levels[1]])
    # 断言原始索引的 levels 未被修改
    assert_matching(idx.levels, levels)

    # 更改多个 levels，不改变原索引
    # 使用新的 levels 列表设置索引的 levels[0] 和 levels[1]，返回一个新的索引对象
    ind2 = idx.set_levels(new_levels, level=[0, 1])
    # 断言新的索引对象的 levels 与新的 levels 列表相匹配
    assert_matching(ind2.levels, new_levels)
    # 断言原始索引的 levels 未被修改
    assert_matching(idx.levels, levels)
    # 复制索引的原始状态，以便后续断言比较
    original_index = idx.copy()

    # 使用 pytest 检查设置 levels 方法是否引发 ValueError 异常，匹配错误信息 "^On"
    with pytest.raises(ValueError, match="^On"):
        idx.set_levels(["c"], level=0)
    
    # 断言索引的 levels 属性与原始索引的 levels 属性相匹配，包括检查数据类型
    assert_matching(idx.levels, original_index.levels, check_dtype=True)

    # 使用 pytest 检查设置 codes 方法是否引发 ValueError 异常，匹配错误信息 "^On"
    with pytest.raises(ValueError, match="^On"):
        idx.set_codes([0, 1, 2, 3, 4, 5], level=0)
    
    # 断言索引的 codes 属性与原始索引的 codes 属性相匹配，包括检查数据类型
    assert_matching(idx.codes, original_index.codes, check_dtype=True)

    # 使用 pytest 检查设置 levels 方法是否引发 TypeError 异常，匹配错误信息 "Levels"
    with pytest.raises(TypeError, match="^Levels"):
        idx.set_levels("c", level=0)
    
    # 再次断言索引的 levels 属性与原始索引的 levels 属性相匹配，包括检查数据类型
    assert_matching(idx.levels, original_index.levels, check_dtype=True)

    # 使用 pytest 检查设置 codes 方法是否引发 TypeError 异常，匹配错误信息 "Codes"
    with pytest.raises(TypeError, match="^Codes"):
        idx.set_codes(1, level=0)
    
    # 最后断言索引的 codes 属性与原始索引的 codes 属性相匹配，包括检查数据类型
    assert_matching(idx.codes, original_index.codes, check_dtype=True)
# 定义一个测试函数，用于测试设置索引对象的编码（codes）
def test_set_codes(idx):
    # 获取索引对象的编码（codes）
    codes = idx.codes
    # 将编码（codes）分为主要编码和次要编码
    major_codes, minor_codes = codes
    # 主要编码中的每个元素加1并取模3，生成新的主要编码
    major_codes = [(x + 1) % 3 for x in major_codes]
    # 次要编码中的每个元素加1并取模1，生成新的次要编码（这里模1的结果仍然是0）
    minor_codes = [(x + 1) % 1 for x in minor_codes]
    # 将新的主要编码和次要编码组成一个列表
    new_codes = [major_codes, minor_codes]

    # 使用新的编码（codes）设置索引对象，返回新的索引对象（ind2）
    ind2 = idx.set_codes(new_codes)
    # 断言新的索引对象（ind2）的编码（codes）与设置的新编码（new_codes）相匹配
    assert_matching(ind2.codes, new_codes)
    # 断言原始索引对象（idx）的编码（codes）未被改变
    assert_matching(idx.codes, codes)

    # 在不改变原索引对象的情况下，针对特定级别更改编码（codes）
    ind2 = idx.set_codes(new_codes[0], level=0)
    assert_matching(ind2.codes, [new_codes[0], codes[1]])
    assert_matching(idx.codes, codes)

    ind2 = idx.set_codes(new_codes[1], level=1)
    assert_matching(ind2.codes, [codes[0], new_codes[1]])
    assert_matching(idx.codes, codes)

    # 在不改变原索引对象的情况下，同时更改多个级别的编码（codes）
    ind2 = idx.set_codes(new_codes, level=[0, 1])
    assert_matching(ind2.codes, new_codes)
    assert_matching(idx.codes, codes)

    # 改变不同类别级别的标签而不改变索引对象（ind）本身
    ind = MultiIndex.from_tuples([(0, i) for i in range(130)])
    new_codes = range(129, -1, -1)
    expected = MultiIndex.from_tuples([(0, i) for i in new_codes])

    # [w/o mutation]
    # 设置索引对象的编码（codes），只改变特定级别（level=1）的编码
    result = ind.set_codes(codes=new_codes, level=1)
    # 断言设置后的结果与预期的索引对象（expected）相等
    assert result.equals(expected)


# 测试设置索引对象的级别（levels）、编码（codes）、名称（names）时的错误输入情况
def test_set_levels_codes_names_bad_input(idx):
    # 获取索引对象的级别（levels）、编码（codes）和名称（names）
    levels, codes = idx.levels, idx.codes
    names = idx.names

    # 断言设置级别（levels）时长度错误会引发 ValueError 异常
    with pytest.raises(ValueError, match="Length of levels"):
        idx.set_levels([levels[0]])

    # 断言设置编码（codes）时长度错误会引发 ValueError 异常
    with pytest.raises(ValueError, match="Length of codes"):
        idx.set_codes([codes[0]])

    # 断言设置名称（names）时长度错误会引发 ValueError 异常
    with pytest.raises(ValueError, match="Length of names"):
        idx.set_names([names[0]])

    # 断言应传入列表形式的级别（levels），而非标量数据，会引发 TypeError 异常
    with pytest.raises(TypeError, match="list of lists-like"):
        idx.set_levels(levels[0])

    # 断言应传入列表形式的编码（codes），而非标量数据，会引发 TypeError 异常
    with pytest.raises(TypeError, match="list of lists-like"):
        idx.set_codes(codes[0])

    # 断言应传入列表形式的名称（names），而非标量数据，会引发 TypeError 异常
    with pytest.raises(TypeError, match="list-like"):
        idx.set_names(names[0])

    # 断言设置级别（levels）时应传入相应的级别索引列表，会引发 TypeError 异常
    with pytest.raises(TypeError, match="list of lists-like"):
        idx.set_levels(levels[0], level=[0, 1])

    # 断言设置级别（levels）时应传入列表形式的级别索引，而非标量数据，会引发 TypeError 异常
    with pytest.raises(TypeError, match="list-like"):
        idx.set_levels(levels, level=0)

    # 断言设置编码（codes）时应传入相应的级别编码列表，会引发 TypeError 异常
    with pytest.raises(TypeError, match="list of lists-like"):
        idx.set_codes(codes[0], level=[0, 1])

    # 断言设置编码（codes）时应传入列表形式的级别编码，而非标量数据，会引发 TypeError 异常
    with pytest.raises(TypeError, match="list-like"):
        idx.set_codes(codes, level=0)

    # 断言设置名称（names）时长度不匹配会引发 ValueError 异常
    with pytest.raises(ValueError, match="Length of names"):
        idx.set_names(names[0], level=[0, 1])

    # 断言设置名称（names）时应传入相应的级别名称列表，会引发 TypeError 异常
    with pytest.raises(TypeError, match="Names must be a"):
        idx.set_names(names, level=0)
@pytest.mark.parametrize("inplace", [True, False])
# 使用 pytest 的 parametrize 装饰器，定义测试函数 test_set_names_with_nlevel_1 的参数化测试，参数为 inplace，取值为 True 和 False
def test_set_names_with_nlevel_1(inplace):
    # GH 21149
    # 确保针对 nlevels == 1 的 MultiIndex，调用 .set_names 不会引发任何错误
    expected = MultiIndex(levels=[[0, 1]], codes=[[0, 1]], names=["first"])
    # 创建一个预期的 MultiIndex 对象，包含一个 level，两个 code，名称为 "first"
    m = MultiIndex.from_product([[0, 1]])
    # 从 product 创建 MultiIndex 对象 m，包含值 [0, 1]
    result = m.set_names("first", level=0, inplace=inplace)
    # 调用 m 的 set_names 方法，设置 level 0 的名称为 "first"，根据 inplace 参数决定是否在原对象上修改

    if inplace:
        result = m
    # 如果 inplace 为 True，则将 result 设为 m 本身

    tm.assert_index_equal(result, expected)
    # 使用 tm.assert_index_equal 断言 result 和 expected 相等


@pytest.mark.parametrize("ordered", [True, False])
# 使用 pytest 的 parametrize 装饰器，定义测试函数 test_set_levels_categorical 的参数化测试，参数为 ordered，取值为 True 和 False
def test_set_levels_categorical(ordered):
    # GH13854
    # 创建一个 MultiIndex 对象 index，由两个数组构成，第一个数组包含字符 "xyzx"，第二个数组为 [0, 1, 2, 3]
    index = MultiIndex.from_arrays([list("xyzx"), [0, 1, 2, 3]])

    cidx = CategoricalIndex(list("bac"), ordered=ordered)
    # 创建一个 CategoricalIndex 对象 cidx，其 categories 为 ['b', 'a', 'c']，ordered 参数根据测试函数的参数 ordered 决定
    result = index.set_levels(cidx, level=0)
    # 调用 index 的 set_levels 方法，将 cidx 设置为 level 0 的 levels

    expected = MultiIndex(levels=[cidx, [0, 1, 2, 3]], codes=index.codes)
    # 创建一个预期的 MultiIndex 对象 expected，其 levels 的第一个元素为 cidx，第二个元素与 index 的 codes 属性相同
    tm.assert_index_equal(result, expected)
    # 使用 tm.assert_index_equal 断言 result 和 expected 相等

    result_lvl = result.get_level_values(0)
    # 获取 result 的第一个 level 的值
    expected_lvl = CategoricalIndex(
        list("bacb"), categories=cidx.categories, ordered=cidx.ordered
    )
    # 创建一个预期的 CategoricalIndex 对象 expected_lvl，其值为 ['b', 'a', 'c', 'b']，categories 和 ordered 与 cidx 相同
    tm.assert_index_equal(result_lvl, expected_lvl)
    # 使用 tm.assert_index_equal 断言 result_lvl 和 expected_lvl 相等


def test_set_value_keeps_names():
    # motivating example from #3742
    # 创建一个 MultiIndex 对象 idx，包含两个 level，分别为 ["hans", "hans", "hans", "grethe", "grethe", "grethe"] 和 ["1", "2", "3", "1", "2", "3"]
    lev1 = ["hans", "hans", "hans", "grethe", "grethe", "grethe"]
    lev2 = ["1", "2", "3"] * 2
    idx = MultiIndex.from_arrays([lev1, lev2], names=["Name", "Number"])
    # 创建一个 DataFrame 对象 df，包含随机数据，索引为 idx，列名为 ["one", "two", "three", "four"]
    df = pd.DataFrame(
        np.random.default_rng(2).standard_normal((6, 4)),
        columns=["one", "two", "three", "four"],
        index=idx,
    )
    # 对 df 按索引进行排序
    df = df.sort_index()
    # 断言 df 的索引名称为 ("Name", "Number")
    assert df.index.names == ("Name", "Number")
    # 在 df 中更新指定位置 ("grethe", "4") 的 "one" 列为 99.34
    df.at[("grethe", "4"), "one"] = 99.34
    # 断言 df 的索引名称仍为 ("Name", "Number")
    assert df.index.names == ("Name", "Number")


def test_set_levels_with_iterable():
    # GH23273
    # 创建一个 MultiIndex 对象 index，包含两个 level，分别为 ["size", "size", "size"] 和 ["black", "black", "black"]
    sizes = [1, 2, 3]
    colors = ["black"] * 3
    index = MultiIndex.from_arrays([sizes, colors], names=["size", "color"])

    result = index.set_levels(map(int, ["3", "2", "1"]), level="size")
    # 调用 index 的 set_levels 方法，使用 map 转换为整数后的 ["3", "2", "1"] 设置为 size level 的 levels

    expected_sizes = [3, 2, 1]
    expected = MultiIndex.from_arrays([expected_sizes, colors], names=["size", "color"])
    # 创建一个预期的 MultiIndex 对象 expected，其 levels 的第一个元素为 expected_sizes，第二个元素为 colors
    tm.assert_index_equal(result, expected)
    # 使用 tm.assert_index_equal 断言 result 和 expected 相等


def test_set_empty_level():
    # GH#48636
    # 创建一个空的 MultiIndex 对象 midx，包含一个 level，名称为 "A"
    midx = MultiIndex.from_arrays([[]], names=["A"])
    result = midx.set_levels(pd.DatetimeIndex([]), level=0)
    # 调用 midx 的 set_levels 方法，使用空的 DatetimeIndex 设置为 level 0 的 levels

    expected = MultiIndex.from_arrays([pd.DatetimeIndex([])], names=["A"])
    # 创建一个预期的 MultiIndex 对象 expected，其 levels 的第一个元素为空的 DatetimeIndex，名称为 "A"
    tm.assert_index_equal(result, expected)
    # 使用 tm.assert_index_equal 断言 result 和 expected 相等


def test_set_levels_pos_args_removal():
    # https://github.com/pandas-dev/pandas/issues/41485
    # 创建一个 MultiIndex 对象 idx，包含两个 level，名称分别为 "foo" 和 "bar"
    idx = MultiIndex.from_tuples(
        [
            (1, "one"),
            (3, "one"),
        ],
        names=["foo", "bar"],
    )
    # 使用 pytest.raises 断言调用 idx 的 set_levels 方法时，传入位置参数会引发 TypeError 异常，异常信息匹配 "positional arguments"
    with pytest.raises(TypeError, match="positional arguments"):
        idx.set_levels(["a", "b", "c"], 0)

    with pytest.raises(TypeError, match="positional arguments"):
        idx.set_codes([[0, 1], [1, 0]], 0)


def test_set_levels_categorical_keep_dtype():
    # GH#52125
    # 创建一个 MultiIndex 对象 midx，包含一个 level，第一个数组为 [5, 6]
    midx = MultiIndex.from_arrays([[5, 6]])
    result = midx.set_levels(levels=pd.Categorical([1, 2]), level=0)
    # 调用 midx 的 set_levels 方法，使用 pd.Categorical([1, 2]) 设置为 level 0 的 levels
    # 创建一个 MultiIndex 对象，使用 from_arrays 方法，传入一个包含一个类别型数据的列表
    expected = MultiIndex.from_arrays([pd.Categorical([1, 2])])
    # 使用测试工具包中的 assert_index_equal 方法，比较 result 和 expected 是否相等
    tm.assert_index_equal(result, expected)
```