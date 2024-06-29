# `D:\src\scipysrc\pandas\pandas\tests\util\test_assert_index_equal.py`

```
# 导入必要的库：numpy用于数值计算，pytest用于测试
import numpy as np
import pytest

# 从pandas库中导入多个类和函数
from pandas import (
    NA,                 # 未知或缺失的数据
    Categorical,        # 分类数据类型
    CategoricalIndex,   # 分类索引类型
    Index,              # 单级索引类型
    MultiIndex,         # 多级索引类型
    NaT,                # 表示缺失的时间数据
    RangeIndex,         # 等差数列索引类型
)

# 导入pandas测试工具模块
import pandas._testing as tm


# 定义一个测试函数，用于检验索引不相等时抛出异常
def test_index_equal_levels_mismatch():
    # 定义异常消息，指示索引级别不同
    msg = """Index are different

Index levels are different
\[left\]:  1, Index\(\[1, 2, 3\], dtype='int64'\)
\[right\]: 2, MultiIndex\(\[\('A', 1\),
            \('A', 2\),
            \('B', 3\),
            \('B', 4\)\],
           \)"""
    
    # 创建两个不同类型的索引对象
    idx1 = Index([1, 2, 3])
    idx2 = MultiIndex.from_tuples([("A", 1), ("A", 2), ("B", 3), ("B", 4)])

    # 使用pytest断言检查索引是否相等，如不相等则抛出异常并匹配消息
    with pytest.raises(AssertionError, match=msg):
        tm.assert_index_equal(idx1, idx2, exact=False)


# 定义一个测试函数，用于检验索引值不相等时抛出异常
def test_index_equal_values_mismatch(check_exact):
    # 定义异常消息，指示多级索引值不同
    msg = """MultiIndex level \[1\] are different

MultiIndex level \[1\] values are different \(25\.0 %\)
\[left\]:  Index\(\[2, 2, 3, 4\], dtype='int64'\)
\[right\]: Index\(\[1, 2, 3, 4\], dtype='int64'\)"""

    # 创建两个不同值的多级索引对象
    idx1 = MultiIndex.from_tuples([("A", 2), ("A", 2), ("B", 3), ("B", 4)])
    idx2 = MultiIndex.from_tuples([("A", 1), ("A", 2), ("B", 3), ("B", 4)])

    # 使用pytest断言检查索引是否相等，如不相等则抛出异常并匹配消息
    with pytest.raises(AssertionError, match=msg):
        tm.assert_index_equal(idx1, idx2, check_exact=check_exact)


# 定义一个测试函数，用于检验索引长度不相等时抛出异常
def test_index_equal_length_mismatch(check_exact):
    # 定义异常消息，指示索引长度不同
    msg = """Index are different

Index length are different
\[left\]:  3, Index\(\[1, 2, 3\], dtype='int64'\)
\[right\]: 4, Index\(\[1, 2, 3, 4\], dtype='int64'\)"""

    # 创建两个不同长度的索引对象
    idx1 = Index([1, 2, 3])
    idx2 = Index([1, 2, 3, 4])

    # 使用pytest断言检查索引是否相等，如不相等则抛出异常并匹配消息
    with pytest.raises(AssertionError, match=msg):
        tm.assert_index_equal(idx1, idx2, check_exact=check_exact)


# 使用pytest参数化装饰器，测试索引类不同但相等情况
@pytest.mark.parametrize("exact", [False, "equiv"])
def test_index_equal_class(exact):
    # 创建两个不同类型的索引对象
    idx1 = Index([0, 1, 2])
    idx2 = RangeIndex(3)

    # 使用pandas测试工具断言检查索引是否相等
    tm.assert_index_equal(idx1, idx2, exact=exact)


# 定义一个测试函数，用于检验索引数据类型不同但相等情况
def test_int_float_index_equal_class_mismatch(check_exact):
    # 定义异常消息，指示索引数据类型不同
    msg = """Index are different

Attribute "inferred_type" are different
\[left\]:  integer
\[right\]: floating"""

    # 创建一个整数类型的索引和一个浮点类型的索引
    idx1 = Index([1, 2, 3])
    idx2 = Index([1, 2, 3], dtype=np.float64)

    # 使用pytest断言检查索引是否相等，如不相等则抛出异常并匹配消息
    with pytest.raises(AssertionError, match=msg):
        tm.assert_index_equal(idx1, idx2, exact=True, check_exact=check_exact)


# 定义一个测试函数，用于检验索引类别不同但相等情况
def test_range_index_equal_class_mismatch(check_exact):
    # 定义异常消息，指示索引类别不同
    msg = """Index are different

Index classes are different
\[left\]:  Index\(\[1, 2, 3\], dtype='int64'\)
\[right\]: """

    # 创建一个普通索引和一个范围索引
    idx1 = Index([1, 2, 3])
    idx2 = RangeIndex(range(3))

    # 使用pytest断言检查索引是否相等，如不相等则抛出异常并匹配消息
    with pytest.raises(AssertionError, match=msg):
        tm.assert_index_equal(idx1, idx2, exact=True, check_exact=check_exact)


# 定义一个测试函数，用于检验索引值接近但不相等时抛出异常
def test_index_equal_values_close(check_exact):
    # 创建两个接近但不相等的浮点数索引
    idx1 = Index([1, 2, 3.0])
    idx2 = Index([1, 2, 3.0000000001])

    # 如果要求精确比较，定义异常消息，指示索引值不同
    if check_exact:
        msg = """Index are different

Index values are different \(33\.33333 %\)
\[left\]:  Index\(\[1.0, 2.0, 3.0], dtype='float64'\)
# 测试函数：检查索引对象是否相等，带有不同的测试情况和参数化设置

def test_index_equal_values_less_close(check_exact, rtol):
    # 创建第一个索引对象 idx1，包含整数和浮点数
    idx1 = Index([1, 2, 3.0])
    # 创建第二个索引对象 idx2，与 idx1 类似，但其中一个值稍有不同
    idx2 = Index([1, 2, 3.0001])
    kwargs = {"check_exact": check_exact, "rtol": rtol}

    # 如果要求精确匹配或者相对容差小于 0.5e-3，则设置错误消息和断言
    if check_exact or rtol < 0.5e-3:
        # 准备错误消息，显示不匹配的索引值和详情
        msg = """Index are different

Index values are different \\(33\\.33333 %\\)
\\[left\\]:  Index\\(\\[1.0, 2.0, 3.0], dtype='float64'\\)
\\[right\\]: Index\\(\\[1.0, 2.0, 3.0001\\], dtype='float64'\\)"""

        # 使用 pytest 断言，检查 assert_index_equal 是否会引发 AssertionError，并匹配指定的错误消息
        with pytest.raises(AssertionError, match=msg):
            tm.assert_index_equal(idx1, idx2, **kwargs)
    else:
        # 否则，直接比较 idx1 和 idx2 的内容，不会引发异常
        tm.assert_index_equal(idx1, idx2, **kwargs)


def test_index_equal_values_too_far(check_exact, rtol):
    # 创建第一个索引对象 idx1，包含整数
    idx1 = Index([1, 2, 3])
    # 创建第二个索引对象 idx2，与 idx1 类似，但其中一个值明显不同
    idx2 = Index([1, 2, 4])
    kwargs = {"check_exact": check_exact, "rtol": rtol}

    # 准备错误消息，显示不匹配的索引值和详情
    msg = """Index are different

Index values are different \\(33\\.33333 %\\)
\\[left\\]:  Index\\(\\[1, 2, 3\\], dtype='int64'\\)
\\[right\\]: Index\\(\\[1, 2, 4\\], dtype='int64'\\)"""

    # 使用 pytest 断言，检查 assert_index_equal 是否会引发 AssertionError，并匹配指定的错误消息
    with pytest.raises(AssertionError, match=msg):
        tm.assert_index_equal(idx1, idx2, **kwargs)


@pytest.mark.parametrize("check_order", [True, False])
def test_index_equal_value_order_mismatch(check_exact, rtol, check_order):
    # 创建第一个索引对象 idx1，包含整数
    idx1 = Index([1, 2, 3])
    # 创建第二个索引对象 idx2，与 idx1 类似，但顺序颠倒
    idx2 = Index([3, 2, 1])

    # 准备错误消息，显示不匹配的索引值和详情
    msg = """Index are different

Index values are different \\(66\\.66667 %\\)
\\[left\\]:  Index\\(\\[1, 2, 3\\], dtype='int64'\\)
\\[right\\]: Index\\(\\[3, 2, 1\\], dtype='int64'\\)"""

    # 如果需要检查顺序，使用 pytest 断言，检查 assert_index_equal 是否会引发 AssertionError，并匹配指定的错误消息
    if check_order:
        with pytest.raises(AssertionError, match=msg):
            tm.assert_index_equal(
                idx1, idx2, check_exact=check_exact, rtol=rtol, check_order=True
            )
    else:
        # 否则，直接比较 idx1 和 idx2 的内容，不会引发异常
        tm.assert_index_equal(
            idx1, idx2, check_exact=check_exact, rtol=rtol, check_order=False
        )


def test_index_equal_level_values_mismatch(check_exact, rtol):
    # 创建第一个多级索引对象 idx1，包含元组列表
    idx1 = MultiIndex.from_tuples([("A", 2), ("A", 2), ("B", 3), ("B", 4)])
    # 创建第二个多级索引对象 idx2，与 idx1 类似，但其中一个值不同
    idx2 = MultiIndex.from_tuples([("A", 1), ("A", 2), ("B", 3), ("B", 4)])
    kwargs = {"check_exact": check_exact, "rtol": rtol}

    # 准备错误消息，显示不匹配的索引级别值和详情
    msg = """MultiIndex level \\[1\\] are different

MultiIndex level \\[1\\] values are different \\(25\\.0 %\\)
\\[left\\]:  Index\\(\\[2, 2, 3, 4\\], dtype='int64'\\)
\\[right\\]: Index\\(\\[1, 2, 3, 4\\], dtype='int64'\\)"""

    # 使用 pytest 断言，检查 assert_index_equal 是否会引发 AssertionError，并匹配指定的错误消息
    with pytest.raises(AssertionError, match=msg):
        tm.assert_index_equal(idx1, idx2, **kwargs)


@pytest.mark.parametrize(
    "name1,name2",
    [(None, "x"), ("x", "x"), (np.nan, np.nan), (NaT, NaT), (np.nan, NaT)],
)
def test_index_equal_names(name1, name2):
    # 创建第一个索引对象 idx1，具有指定的名称 name1
    idx1 = Index([1, 2, 3], name=name1)
    # 创建第二个索引对象 idx2，具有指定的名称 name2
    idx2 = Index([1, 2, 3], name=name2)

    # 如果名称相同或均为 None 或 NaN，则断言这两个索引相等
    if name1 == name2 or name1 is name2:
        tm.assert_index_equal(idx1, idx2)
    else:
        # 如果name1为"x"，则将name1赋值为"'x'"，否则保持不变
        name1 = "'x'" if name1 == "x" else name1
        # 如果name2为"x"，则将name2赋值为"'x'"，否则保持不变
        name2 = "'x'" if name2 == "x" else name2
        # 构建包含多行字符串的消息，指示索引不同的情况
        msg = f"""Index are different
# 测试函数：测试在索引对象不相等时是否会引发断言错误，检查名称时不考虑顺序
def test_assert_index_equal_different_names_check_order_false():
    # 创建具有不同名称的两个索引对象
    idx1 = Index([1, 3], name="a")
    idx2 = Index([3, 1], name="b")
    # 断言测试函数会引发断言错误，错误消息中会显示名称不同
    with pytest.raises(AssertionError, match='"names" are different'):
        tm.assert_index_equal(idx1, idx2, check_order=False, check_names=True)


# 测试函数：测试在混合数据类型的索引对象相等时是否不会引发断言错误
def test_assert_index_equal_mixed_dtype():
    # 创建包含混合数据类型的索引对象
    idx = Index(["foo", "bar", 42])
    # 断言两个相等的索引对象不会引发断言错误
    tm.assert_index_equal(idx, idx, check_order=False)


# 测试函数：测试在具有任意数值型EA数据类型的索引对象相等时是否不会引发断言错误
def test_assert_index_equal_ea_dtype_order_false(any_numeric_ea_dtype):
    # 创建具有相同EA数据类型的两个索引对象
    idx1 = Index([1, 3], dtype=any_numeric_ea_dtype)
    idx2 = Index([3, 1], dtype=any_numeric_ea_dtype)
    # 断言两个相等的索引对象不会引发断言错误
    tm.assert_index_equal(idx1, idx2, check_order=False)


# 测试函数：测试在对象型整数索引对象相等时是否不会引发断言错误
def test_assert_index_equal_object_ints_order_false():
    # 创建对象型整数的两个索引对象
    idx1 = Index([1, 3], dtype="object")
    idx2 = Index([3, 1], dtype="object")
    # 断言两个相等的索引对象不会引发断言错误
    tm.assert_index_equal(idx1, idx2, check_order=False)


# 测试函数：测试在索引对象的推断类型不同时是否会引发断言错误
def test_assert_index_equal_different_inferred_types():
    # 创建推断类型不同的两个索引对象
    msg = """\
Index are different

Attribute "inferred_type" are different
\\[left\\]:  mixed
\\[right\\]: datetime"""
    idx1 = Index([NA, np.datetime64("nat")])
    idx2 = Index([NA, NaT])
    # 断言测试函数会引发断言错误，错误消息中会显示推断类型不同
    with pytest.raises(AssertionError, match=msg):
        tm.assert_index_equal(idx1, idx2)


# 测试函数：测试在范围类别索引对象的不精确匹配时是否会引发断言错误
@pytest.mark.parametrize("exact", [False, True])
def test_index_equal_range_categories(check_categorical, exact):
    # GH41263
    msg = """\
Index are different

Index classes are different
\\[left\\]:  RangeIndex\\(start=0, stop=10, step=1\\)
\\[right\\]: Index\\(\\[0, 1, 2, 3, 4, 5, 6, 7, 8, 9\\], dtype='int64'\\)"""
    rcat = CategoricalIndex(RangeIndex(10))
    icat = CategoricalIndex(list(range(10)))
    if check_categorical and exact:
        # 断言测试函数会引发断言错误，错误消息中会显示索引类别不同
        with pytest.raises(AssertionError, match=msg):
            tm.assert_index_equal(rcat, icat, check_categorical=True, exact=True)
    else:
        # 断言两个相等的索引对象不会引发断言错误
        tm.assert_index_equal(rcat, icat, check_categorical=check_categorical, exact=exact)
# 使用 pytest 的 parametrize 装饰器，分别对 check_categorical 和 check_names 参数进行多组参数化测试
@pytest.mark.parametrize("check_categorical", [True, False])
@pytest.mark.parametrize("check_names", [True, False])
def test_assert_ea_index_equal_non_matching_na(check_names, check_categorical):
    # GH#48608: 测试用例标识号
    # 创建包含整数和 NA 值的 Index 对象 idx1 和 idx2，指定数据类型为 "Int64"
    idx1 = Index([1, 2], dtype="Int64")
    idx2 = Index([1, NA], dtype="Int64")
    # 使用 pytest.raises 检查是否会引发 AssertionError 异常，并匹配错误信息中包含 "50.0 %"
    with pytest.raises(AssertionError, match="50.0 %"):
        # 调用 tm.assert_index_equal 方法，对比 idx1 和 idx2 的相等性，根据 check_names 和 check_categorical 参数选择性检查
        tm.assert_index_equal(
            idx1, idx2, check_names=check_names, check_categorical=check_categorical
        )


# 使用 pytest 的 parametrize 装饰器，对 check_categorical 参数进行多组参数化测试
@pytest.mark.parametrize("check_categorical", [True, False])
def test_assert_multi_index_dtype_check_categorical(check_categorical):
    # GH#52126: 测试用例标识号
    # 创建包含分类数据的 MultiIndex 对象 idx1 和 idx2，分别使用不同的数据类型（np.uint64 和 np.int64）
    idx1 = MultiIndex.from_arrays([Categorical(np.array([1, 2], dtype=np.uint64))])
    idx2 = MultiIndex.from_arrays([Categorical(np.array([1, 2], dtype=np.int64))])
    if check_categorical:
        # 如果 check_categorical 为 True，则使用 pytest.raises 检查是否会引发 AssertionError 异常，
        # 并匹配错误信息以 "^MultiIndex level \[0\] are different" 开头
        with pytest.raises(
            AssertionError, match=r"^MultiIndex level \[0\] are different"
        ):
            # 调用 tm.assert_index_equal 方法，对比 idx1 和 idx2 的相等性，检查分类数据
            tm.assert_index_equal(idx1, idx2, check_categorical=check_categorical)
    else:
        # 如果 check_categorical 为 False，则直接调用 tm.assert_index_equal 方法对比 idx1 和 idx2 的相等性，不检查分类数据
        tm.assert_index_equal(idx1, idx2, check_categorical=check_categorical)
```