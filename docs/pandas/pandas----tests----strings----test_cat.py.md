# `D:\src\scipysrc\pandas\pandas\tests\strings\test_cat.py`

```
import re  # 导入正则表达式模块

import numpy as np  # 导入 NumPy 库
import pytest  # 导入 pytest 测试框架

import pandas.util._test_decorators as td  # 导入 pandas 测试装饰器模块

from pandas import (  # 从 pandas 库导入多个类和函数
    DataFrame,
    Index,
    MultiIndex,
    Series,
    _testing as tm,
    concat,
    option_context,
)


@pytest.fixture
def index_or_series2(index_or_series):
    return index_or_series


@pytest.mark.parametrize("other", [None, Series, Index])
def test_str_cat_name(index_or_series, other):
    # GH 21053
    box = index_or_series  # 将 index_or_series 赋值给 box
    values = ["a", "b"]  # 创建字符串列表 values
    if other:
        other = other(values)  # 如果 other 不为空，则调用 other(values)，可能是 Series 或 Index
    else:
        other = values  # 否则，使用 values 本身
    result = box(values, name="name").str.cat(other, sep=",")  # 对 box 应用 str.cat 方法，用逗号连接 other
    assert result.name == "name"  # 断言结果的名称为 "name"


@pytest.mark.parametrize(
    "infer_string", [False, pytest.param(True, marks=td.skip_if_no("pyarrow"))]
)
def test_str_cat(index_or_series, infer_string):
    with option_context("future.infer_string", infer_string):
        box = index_or_series  # 将 index_or_series 赋值给 box
        # test_cat above tests "str_cat" from ndarray;
        # here testing "str.cat" from Series/Index to ndarray/list
        s = box(["a", "a", "b", "b", "c", np.nan])  # 创建 Series s 包含多个元素

        # single array
        result = s.str.cat()  # 使用 str.cat() 方法连接字符串
        expected = "aabbc"  # 预期结果为 "aabbc"
        assert result == expected  # 断言结果与预期相符

        result = s.str.cat(na_rep="-")  # 使用 str.cat() 方法连接字符串，指定缺失值替换为 "-"
        expected = "aabbc-"  # 预期结果为 "aabbc-"
        assert result == expected  # 断言结果与预期相符

        result = s.str.cat(sep="_", na_rep="NA")  # 使用 str.cat() 方法连接字符串，指定分隔符为 "_"，缺失值替换为 "NA"
        expected = "a_a_b_b_c_NA"  # 预期结果为 "a_a_b_b_c_NA"
        assert result == expected  # 断言结果与预期相符

        t = np.array(["a", np.nan, "b", "d", "foo", np.nan], dtype=object)  # 创建包含多个元素的 NumPy 数组 t
        expected = box(["aa", "a-", "bb", "bd", "cfoo", "--"])  # 使用 box 创建预期结果的 Series/Index

        # Series/Index with array
        result = s.str.cat(t, na_rep="-")  # 使用 str.cat() 方法连接 s 和 t，指定缺失值替换为 "-"
        tm.assert_equal(result, expected)  # 使用测试框架 tm 断言结果与预期相等

        # Series/Index with list
        result = s.str.cat(list(t), na_rep="-")  # 使用 str.cat() 方法连接 s 和 t 的列表形式，指定缺失值替换为 "-"
        tm.assert_equal(result, expected)  # 使用测试框架 tm 断言结果与预期相等

        # errors for incorrect lengths
        rgx = r"If `others` contains arrays or lists \(or other list-likes.*"
        z = Series(["1", "2", "3"])  # 创建包含多个元素的 Series z

        with pytest.raises(ValueError, match=rgx):  # 断言 ValueError 异常，匹配指定的正则表达式信息
            s.str.cat(z.values)

        with pytest.raises(ValueError, match=rgx):  # 断言 ValueError 异常，匹配指定的正则表达式信息
            s.str.cat(list(z))


def test_str_cat_raises_intuitive_error(index_or_series):
    # GH 11334
    box = index_or_series  # 将 index_or_series 赋值给 box
    s = box(["a", "b", "c", "d"])  # 创建包含多个元素的 Series s
    message = "Did you mean to supply a `sep` keyword?"  # 错误信息提示
    with pytest.raises(ValueError, match=message):  # 断言 ValueError 异常，匹配指定的错误信息
        s.str.cat("|")
    with pytest.raises(ValueError, match=message):  # 断言 ValueError 异常，匹配指定的错误信息
        s.str.cat("    ")


@pytest.mark.parametrize(
    "infer_string", [False, pytest.param(True, marks=td.skip_if_no("pyarrow"))]
)
@pytest.mark.parametrize("sep", ["", None])
@pytest.mark.parametrize("dtype_target", ["object", "category"])
@pytest.mark.parametrize("dtype_caller", ["object", "category"])
def test_str_cat_categorical(
    index_or_series, dtype_caller, dtype_target, sep, infer_string
):
    box = index_or_series  # 将 index_or_series 赋值给 box
    # 设置上下文选项，配置 future.infer_string 参数为 infer_string
    with option_context("future.infer_string", infer_string):
        # 创建 Index 对象 s，包含指定的数据和数据类型 dtype_caller
        s = Index(["a", "a", "b", "a"], dtype=dtype_caller)
        # 如果 box 等于 Index，则 s 不变；否则将 s 封装成 Series 对象，使用 s 作为索引和数据，数据类型为 s 的数据类型
        s = s if box == Index else Series(s, index=s, dtype=s.dtype)
        # 创建 Index 对象 t，包含指定的数据和数据类型 dtype_target
        t = Index(["b", "a", "b", "c"], dtype=dtype_target)

        # 创建期望的 Index 对象 expected，包含指定的数据和数据类型
        expected = Index(
            ["ab", "aa", "bb", "ac"], dtype=object if dtype_caller == "object" else None
        )
        # 如果 box 等于 Index，则 expected 不变；否则将 expected 封装成 Series 对象，使用 s 作为索引和数据类型为 dtype_caller 的数据类型
        expected = (
            expected
            if box == Index
            else Series(
                expected, index=Index(s, dtype=dtype_caller), dtype=expected.dtype
            )
        )

        # 将 Series/Index s 中的字符串与 t.values 中的字符串按照指定的分隔符 sep 连接，返回结果给 result
        result = s.str.cat(t.values, sep=sep)
        # 断言 result 与期望的结果 expected 相等
        tm.assert_equal(result, expected)

        # 使用具有匹配索引的 Series 对象 t 与 Series/Index s 进行字符串连接
        t = Series(t.values, index=Index(s, dtype=dtype_caller))
        result = s.str.cat(t, sep=sep)
        tm.assert_equal(result, expected)

        # 将 Series/Index s 中的字符串与 Series t 中的值按照指定的分隔符 sep 连接，返回结果给 result
        result = s.str.cat(t.values, sep=sep)
        # 断言 result 与期望的结果 expected 相等
        tm.assert_equal(result, expected)

        # 使用具有不同索引的 Series 对象 t 与 Series/Index s 进行字符串连接
        t = Series(t.values, index=t.values)
        # 创建期望的 Index 对象 expected，包含指定的数据和数据类型
        expected = Index(
            ["aa", "aa", "bb", "bb", "aa"],
            dtype=object if dtype_caller == "object" else None,
        )
        # 确定期望的数据类型 dtype
        dtype = object if dtype_caller == "object" else s.dtype.categories.dtype
        # 如果 box 等于 Index，则 expected 不变；否则将 expected 封装成 Series 对象，使用 expected.str[:1] 作为索引和数据类型为 dtype 的数据类型
        expected = (
            expected
            if box == Index
            else Series(
                expected,
                index=Index(expected.str[:1], dtype=dtype),
                dtype=expected.dtype,
            )
        )

        # 将 Series/Index s 中的字符串与 Series t 中的值按照指定的分隔符 sep 连接，返回结果给 result
        result = s.str.cat(t, sep=sep)
        # 断言 result 与期望的结果 expected 相等
        tm.assert_equal(result, expected)
@pytest.mark.parametrize(
    "data",
    [[1, 2, 3], [0.1, 0.2, 0.3], [1, 2, "b"]],
    ids=["integers", "floats", "mixed"],
)
# 使用 pytest 的 parametrize 装饰器，定义多组测试数据和对应的标识符
@pytest.mark.parametrize(
    "box",
    [Series, Index, list, lambda x: np.array(x, dtype=object)],
    ids=["Series", "Index", "list", "np.array"],
)
# 使用 pytest 的 parametrize 装饰器，定义多个测试参数（Series、Index、list、np.array）及其标识符
def test_str_cat_wrong_dtype_raises(box, data):
    # GH 22722
    # 为了解决 GitHub 问题编号 22722
    s = Series(["a", "b", "c"])
    # 创建一个包含字符串的 Series 对象 s

    t = box(data)
    # 使用给定的 box 参数创建对象 t，可以是 Series、Index、list 或带有特定 dtype 的 np.array

    msg = "Concatenation requires list-likes containing only strings.*"
    # 设置错误消息的正则表达式模式，用于匹配 TypeError 异常信息

    with pytest.raises(TypeError, match=msg):
        # 使用 pytest 的 raises 方法检查是否会引发 TypeError 异常，并匹配预期的错误消息
        # 需要使用 join="outer" 和 na_rep，否则 Index 对象不会引发异常
        s.str.cat(t, join="outer", na_rep="-")


def test_str_cat_mixed_inputs(index_or_series):
    # 测试函数，接受 index_or_series 参数
    box = index_or_series
    # 将参数值赋给变量 box

    s = Index(["a", "b", "c", "d"])
    # 创建一个包含字符串的 Index 对象 s

    s = s if box == Index else Series(s, index=s)
    # 如果 box 是 Index，则保持 s 不变，否则将 s 转换为 Series 对象并设置索引为 s 的值

    t = Series(["A", "B", "C", "D"], index=s.values)
    # 创建一个 Series 对象 t，包含字符串和与 s 相同的索引值，并将其与 s 合并

    d = concat([t, Series(s, index=s)], axis=1)
    # 使用 concat 函数将 t 和包含 s 的 Series 对象在 axis=1 方向上合并为 DataFrame 对象 d

    expected = Index(["aAa", "bBb", "cCc", "dDd"])
    # 创建预期的 Index 对象 expected，包含合并字符串的结果

    expected = expected if box == Index else Series(expected.values, index=s.values)
    # 如果 box 是 Index，则保持 expected 不变，否则将其转换为 Series 对象并设置索引为 s 的值

    # Series/Index with DataFrame
    # 使用 DataFrame 的 Series/Index
    result = s.str.cat(d)
    # 使用 Series 对象 s 的 str.cat 方法与 DataFrame 对象 d 进行连接
    tm.assert_equal(result, expected)

    # Series/Index with two-dimensional ndarray
    # 使用二维 ndarray 的 Series/Index
    result = s.str.cat(d.values)
    # 使用 Series 对象 s 的 str.cat 方法与 d.values（DataFrame 的值数组）进行连接
    tm.assert_equal(result, expected)

    # Series/Index with list of Series
    # 使用 Series 的 Series/Index
    result = s.str.cat([t, s])
    # 使用 Series 对象 s 的 str.cat 方法与包含 t 和 s 的列表进行连接
    tm.assert_equal(result, expected)

    # Series/Index with mixed list of Series/array
    # 使用混合列表的 Series/Index
    result = s.str.cat([t, s.values])
    # 使用 Series 对象 s 的 str.cat 方法与包含 t 和 s 的值数组的列表进行连接
    tm.assert_equal(result, expected)

    # Series/Index with list of Series; different indexes
    # 使用具有不同索引的 Series/Index
    t.index = ["b", "c", "d", "a"]
    # 修改 Series 对象 t 的索引

    expected = box(["aDa", "bAb", "cBc", "dCd"])
    # 创建预期的对象 expected，根据 box 的类型创建不同的对象（Index 或 Series）

    expected = expected if box == Index else Series(expected.values, index=s.values)
    # 如果 box 是 Index，则保持 expected 不变，否则将其转换为 Series 对象并设置索引为 s 的值

    result = s.str.cat([t, s])
    # 使用 Series 对象 s 的 str.cat 方法与包含 t 和 s 的列表进行连接
    tm.assert_equal(result, expected)

    # Series/Index with mixed list; different index
    # 使用混合列表的 Series/Index，具有不同的索引
    result = s.str.cat([t, s.values])
    # 使用 Series 对象 s 的 str.cat 方法与包含 t 和 s 的值数组的列表进行连接
    tm.assert_equal(result, expected)

    # Series/Index with DataFrame; different indexes
    # 使用具有不同索引的 DataFrame 的 Series/Index
    d.index = ["b", "c", "d", "a"]
    # 修改 DataFrame 对象 d 的索引

    expected = box(["aDd", "bAa", "cBb", "dCc"])
    # 创建预期的对象 expected，根据 box 的类型创建不同的对象（Index 或 Series）

    expected = expected if box == Index else Series(expected.values, index=s.values)
    # 如果 box 是 Index，则保持 expected 不变，否则将其转换为 Series 对象并设置索引为 s 的值

    result = s.str.cat(d)
    # 使用 Series 对象 s 的 str.cat 方法与 DataFrame 对象 d 进行连接
    tm.assert_equal(result, expected)

    # errors for incorrect lengths
    # 长度不正确的错误处理
    rgx = r"If `others` contains arrays or lists \(or other list-likes.*"
    # 错误消息的正则表达式模式，用于匹配 ValueError 异常信息

    z = Series(["1", "2", "3"])
    # 创建一个包含字符串的 Series 对象 z

    e = concat([z, z], axis=1)
    # 使用 concat 函数将 z 和 z 在 axis=1 方向上合并为 DataFrame 对象 e

    # two-dimensional ndarray
    # 二维 ndarray
    with pytest.raises(ValueError, match=rgx):
        # 使用 pytest 的 raises 方法检查是否会引发 ValueError 异常，并匹配预期的错误消息
        s.str.cat(e.values)

    # list of list-likes
    # 列表形式的列表类对象
    with pytest.raises(ValueError, match=rgx):
        # 使用 pytest 的 raises 方法检查是否会引发 ValueError 异常，并匹配预期的错误消息
        s.str.cat([z.values, s.values])

    # mixed list of Series/list-like
    # 混合的 Series 和列表类对象
    with pytest.raises(ValueError, match=rgx):
        # 使用 pytest 的 raises 方法检查是否会引发 ValueError 异常，并匹配预期的错误消息
        s.str.cat([z.values, s])

    # errors for incorrect arguments in list-like
    # 列表类对象中错误参数的错误处理
    rgx = "others must be Series, Index, DataFrame,.*"
    # 错误消息的正则表达式模式

    # make sure None/NaN do not crash checks in _get_series_list
    # 确保 None/NaN 在 _get_series_list 中不会导致崩溃检查
    # 创建一个包含字符串和缺失值的 Series 对象
    u = Series(["a", np.nan, "c", None])

    # 测试字符串和 Series 混合输入的情况
    with pytest.raises(TypeError, match=rgx):
        s.str.cat([u, "u"])

    # 测试包含 DataFrame 的列表作为输入的情况
    with pytest.raises(TypeError, match=rgx):
        s.str.cat([u, d])

    # 测试包含二维 ndarray 的列表作为输入的情况
    with pytest.raises(TypeError, match=rgx):
        s.str.cat([u, d.values])

    # 测试包含嵌套列表的输入情况
    with pytest.raises(TypeError, match=rgx):
        s.str.cat([u, [u, d]])

    # 测试禁止输入类型为集合 (set) 的情况，GH 23009
    with pytest.raises(TypeError, match=rgx):
        s.str.cat(set(u))

    # 测试列表中包含禁止输入类型为集合 (set) 的情况，GH 23009
    with pytest.raises(TypeError, match=rgx):
        s.str.cat([u, set(u)])

    # 测试其他禁止的输入类型，例如整数
    with pytest.raises(TypeError, match=rgx):
        s.str.cat(1)

    # 测试嵌套类似列表的情况
    with pytest.raises(TypeError, match=rgx):
        s.str.cat(iter([t.values, list(s)]))
# 定义函数，测试字符串连接并对齐索引或系列对象
def test_str_cat_align_indexed(index_or_series, join_type):
    # GitHub 上的问题链接
    box = index_or_series

    # 创建 Series 对象 s 和 t，每个对象都有自定义的索引
    s = Series(["a", "b", "c", "d"], index=["a", "b", "c", "d"])
    t = Series(["D", "A", "E", "B"], index=["d", "a", "e", "b"])
    # 对 s 和 t 进行对齐操作，根据指定的连接方式
    sa, ta = s.align(t, join=join_type)
    # 手动对齐输入后的结果
    expected = sa.str.cat(ta, na_rep="-")

    # 如果 box 是 Index 类型，则将 s、sa、expected 转换为 Index 对象
    if box == Index:
        s = Index(s)
        sa = Index(sa)
        expected = Index(expected)

    # 使用字符串连接方法，根据指定的连接方式对 s 和 t 进行连接，并使用 '-' 作为缺失值的表示
    result = s.str.cat(t, join=join_type, na_rep="-")
    # 断言结果与预期相等
    tm.assert_equal(result, expected)


# 定义函数，测试混合输入情况下的字符串连接和对齐
def test_str_cat_align_mixed_inputs(join_type):
    # 创建 Series 对象 s 和 t，t 带有自定义的索引
    s = Series(["a", "b", "c", "d"])
    t = Series(["d", "a", "e", "b"], index=[3, 0, 4, 1])
    # 创建 t 的复制，并沿轴 1 连接形成 DataFrame 对象 d
    d = concat([t, t], axis=1)

    # 创建预期的结果 Series 对象 expected_outer
    expected_outer = Series(["aaa", "bbb", "c--", "ddd", "-ee"])
    # 根据索引的交集或并集获得预期结果
    expected = expected_outer.loc[s.index.join(t.index, how=join_type)]

    # 使用字符串连接方法，根据指定的连接方式对 s 和 [t, t] 进行连接，并使用 '-' 作为缺失值的表示
    result = s.str.cat([t, t], join=join_type, na_rep="-")
    # 断言结果与预期相等
    tm.assert_series_equal(result, expected)

    # 使用字符串连接方法，根据指定的连接方式对 s 和 d 进行连接，并使用 '-' 作为缺失值的表示
    result = s.str.cat(d, join=join_type, na_rep="-")
    # 断言结果与预期相等
    tm.assert_series_equal(result, expected)

    # 创建 np.array 对象 u 和预期的结果 Series 对象 expected_outer
    u = np.array(["A", "B", "C", "D"])
    expected_outer = Series(["aaA", "bbB", "c-C", "ddD", "-e-"])
    # 获取 rhs_idx，即 t.index 与 s.index 的交集或并集
    rhs_idx = (
        t.index.intersection(s.index)
        if join_type == "inner"
        else t.index.union(s.index)
        if join_type == "outer"
        else t.index.append(s.index.difference(t.index))
    )
    # 根据索引的交集或并集获得预期结果
    expected = expected_outer.loc[s.index.join(rhs_idx, how=join_type)]
    # 使用字符串连接方法，根据指定的连接方式对 s 和 [t, u] 进行连接，并使用 '-' 作为缺失值的表示
    result = s.str.cat([t, u], join=join_type, na_rep="-")
    # 断言结果与预期相等
    tm.assert_series_equal(result, expected)

    # 使用 pytest 引发 TypeError 异常，测试禁止嵌套列表的情况
    with pytest.raises(TypeError, match="others must be Series,.*"):
        s.str.cat([t, list(u)], join=join_type)

    # 使用 pytest 引发 ValueError 异常，测试长度不匹配的情况
    rgx = r"If `others` contains arrays or lists \(or other list-likes.*"
    z = Series(["1", "2", "3"]).values

    # 测试长度不匹配的非索引对象的情况
    with pytest.raises(ValueError, match=rgx):
        s.str.cat(z, join=join_type)

    # 测试长度不匹配的非索引对象列表的情况
    with pytest.raises(ValueError, match=rgx):
        s.str.cat([t, z], join=join_type)


# 定义函数，测试所有缺失值情况下的字符串连接
def test_str_cat_all_na(index_or_series, index_or_series2):
    # GH 24044
    box = index_or_series
    other = index_or_series2

    # 创建 Index 对象 s，如果 box 是 Series 类型，则将其转换为 Series 对象
    s = Index(["a", "b", "c", "d"])
    s = s if box == Index else Series(s, index=s)
    # 创建 other 对象 t，其值全部为 NaN
    t = other([np.nan] * 4, dtype=object)
    # 如果 other 是 Series 类型，则将 t 转换为 Series 对象并使用 s 的索引对齐
    t = t if other == Index else Series(t, index=s)

    # 如果 box 是 Series 类型，则创建所有值为 NaN 的 Series 对象 expected
    if box == Series:
        expected = Series([np.nan] * 4, index=s.index, dtype=s.dtype)
    else:  # box == Index
        # TODO: Strimg option, this should return string dtype
        expected = Index([np.nan] * 4, dtype=object)
    # 使用 Series 对象 s 的 str.cat 方法，将其与对象 t 连接，并指定连接方式为左连接
    result = s.str.cat(t, join="left")
    # 使用测试框架中的 assert_equal 方法，验证 result 是否等于 expected
    tm.assert_equal(result, expected)

    # 如果 other 等于 Series，则执行以下逻辑（仅适用于 Series 对象）
    if other == Series:
        # 创建一个值全为 NaN 的 Series 对象，长度为 4，数据类型为 object，索引与 t 的索引相同
        expected = Series([np.nan] * 4, dtype=object, index=t.index)
        # 使用 Series 对象 t 的 str.cat 方法，将其与对象 s 连接，并指定连接方式为左连接
        result = t.str.cat(s, join="left")
        # 使用测试框架中的 assert_series_equal 方法，验证 result 是否与 expected 的 Series 对象相等
        tm.assert_series_equal(result, expected)
# 定义一个测试函数，用于测试字符串串联的特殊情况
def test_str_cat_special_cases():
    # 创建一个包含字符串元素的 Series 对象
    s = Series(["a", "b", "c", "d"])
    # 创建另一个 Series 对象，指定索引
    t = Series(["d", "a", "e", "b"], index=[3, 0, 4, 1])

    # 期望的串联结果，包含不同类型元素的迭代器
    expected = Series(["aaa", "bbb", "c-c", "ddd", "-e-"])
    # 使用 s 的字符串方法串联 t 和 s.values，以外部连接方式，空值表示为 "-"
    result = s.str.cat(iter([t, s.values]), join="outer", na_rep="-")
    # 断言结果与期望相等
    tm.assert_series_equal(result, expected)

    # 右对齐，使用不同索引的其他 Series 对象
    expected = Series(["aa-", "d-d"], index=[0, 3])
    # 使用 s 的字符串方法串联 t.loc[[0]] 和 t.loc[[3]]，以右连接方式，空值表示为 "-"
    result = s.str.cat([t.loc[[0]], t.loc[[3]]], join="right", na_rep="-")
    # 断言结果与期望相等
    tm.assert_series_equal(result, expected)


# 定义测试函数，测试在筛选后的索引上进行串联操作
def test_cat_on_filtered_index():
    # 创建一个带有多级索引的 DataFrame 对象
    df = DataFrame(
        index=MultiIndex.from_product(
            [[2011, 2012], [1, 2, 3]], names=["year", "month"]
        )
    )

    # 重置索引
    df = df.reset_index()
    # 筛选出月份大于1的行
    df = df[df.month > 1]

    # 将年份转换为字符串
    str_year = df.year.astype("str")
    # 将月份转换为字符串
    str_month = df.month.astype("str")
    # 使用字符串方法串联年份和月份，以空格分隔
    str_both = str_year.str.cat(str_month, sep=" ")

    # 断言串联结果在指定位置等于期望值
    assert str_both.loc[1] == "2011 2"

    # 使用字符串方法串联年份和多个月份，以空格分隔
    str_multiple = str_year.str.cat([str_month, str_month], sep=" ")

    # 断言串联结果在指定位置等于期望值
    assert str_multiple.loc[1] == "2011 2 2"


# 使用 pytest.mark.parametrize 标记，测试不同类型的串联操作
@pytest.mark.parametrize("klass", [tuple, list, np.array, Series, Index])
def test_cat_different_classes(klass):
    # 创建一个包含字符串元素的 Series 对象
    s = Series(["a", "b", "c"])
    # 使用字符串方法串联 s 和给定类的对象，例如 tuple, list, np.array 等
    result = s.str.cat(klass(["x", "y", "z"]))
    # 期望的串联结果
    expected = Series(["ax", "by", "cz"])
    # 断言结果与期望相等
    tm.assert_series_equal(result, expected)


# 定义测试函数，测试在 Series 对象上使用字符串方法进行串联操作
def test_cat_on_series_dot_str():
    # 创建一个包含字符串元素的 Series 对象
    ps = Series(["AbC", "de", "FGHI", "j", "kLLLm"])

    # 设置期望抛出的异常消息
    message = re.escape(
        "others must be Series, Index, DataFrame, np.ndarray "
        "or list-like (either containing only strings or "
        "containing only objects of type Series/Index/"
        "np.ndarray[1-dim])"
    )
    # 断言在串联操作中抛出特定类型的 TypeError 异常，异常消息与期望相匹配
    with pytest.raises(TypeError, match=message):
        ps.str.cat(others=ps.str)
```