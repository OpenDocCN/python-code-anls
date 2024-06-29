# `D:\src\scipysrc\pandas\pandas\tests\arrays\categorical\test_algos.py`

```
# 导入 NumPy 库，用于数值计算
import numpy as np
# 导入 Pytest 库，用于单元测试
import pytest

# 导入 Pandas 库，用于数据处理和分析
import pandas as pd
# 导入 Pandas 内部测试工具
import pandas._testing as tm


# 使用 pytest 的参数化装饰器，定义测试函数 test_factorize，传入参数 categories
@pytest.mark.parametrize("categories", [["b", "a", "c"], ["a", "b", "c", "d"]])
def test_factorize(categories, ordered):
    # 创建一个分类数据 cat，指定数据和分类列表 categories，是否有序由参数 ordered 决定
    cat = pd.Categorical(
        ["b", "b", "a", "c", None], categories=categories, ordered=ordered
    )
    # 使用 Pandas 的 factorize 函数进行因子化处理，返回因子化后的代码和唯一值
    codes, uniques = pd.factorize(cat)
    # 期望的因子化后的代码数组
    expected_codes = np.array([0, 0, 1, 2, -1], dtype=np.intp)
    # 期望的唯一值数组，使用相同的 categories 和 ordered 参数
    expected_uniques = pd.Categorical(
        ["b", "a", "c"], categories=categories, ordered=ordered
    )

    # 断言 codes 数组与期望的代码数组相等
    tm.assert_numpy_array_equal(codes, expected_codes)
    # 断言 uniques 对象与期望的唯一值对象相等
    tm.assert_categorical_equal(uniques, expected_uniques)


# 定义测试函数 test_factorized_sort
def test_factorized_sort():
    # 创建一个分类数据 cat，指定数据并不指定 categories 或 ordered 参数
    cat = pd.Categorical(["b", "b", None, "a"])
    # 使用 Pandas 的 factorize 函数进行因子化处理，排序参数为 True
    codes, uniques = pd.factorize(cat, sort=True)
    # 期望的因子化后的代码数组，按字母序排序
    expected_codes = np.array([1, 1, -1, 0], dtype=np.intp)
    # 期望的唯一值数组，按字母序排序
    expected_uniques = pd.Categorical(["a", "b"])

    # 断言 codes 数组与期望的代码数组相等
    tm.assert_numpy_array_equal(codes, expected_codes)
    # 断言 uniques 对象与期望的唯一值对象相等
    tm.assert_categorical_equal(uniques, expected_uniques)


# 定义测试函数 test_factorized_sort_ordered
def test_factorized_sort_ordered():
    # 创建一个分类数据 cat，指定数据、categories 和 ordered 参数为 True
    cat = pd.Categorical(
        ["b", "b", None, "a"], categories=["c", "b", "a"], ordered=True
    )

    # 使用 Pandas 的 factorize 函数进行因子化处理，排序参数为 True
    codes, uniques = pd.factorize(cat, sort=True)
    # 期望的因子化后的代码数组，按字母序排序
    expected_codes = np.array([0, 0, -1, 1], dtype=np.intp)
    # 期望的唯一值数组，按字母序排序
    expected_uniques = pd.Categorical(
        ["b", "a"], categories=["c", "b", "a"], ordered=True
    )

    # 断言 codes 数组与期望的代码数组相等
    tm.assert_numpy_array_equal(codes, expected_codes)
    # 断言 uniques 对象与期望的唯一值对象相等
    tm.assert_categorical_equal(uniques, expected_uniques)


# 定义测试函数 test_isin_cats
def test_isin_cats():
    # GH2003
    # 创建一个分类数据 cat，包含字符串 "a"、"b" 和 NaN
    cat = pd.Categorical(["a", "b", np.nan])

    # 调用 isin 方法，判断是否包含 "a" 和 NaN，返回布尔数组
    result = cat.isin(["a", np.nan])
    # 期望的布尔数组
    expected = np.array([True, False, True], dtype=bool)
    # 断言结果数组与期望数组相等
    tm.assert_numpy_array_equal(expected, result)

    # 调用 isin 方法，判断是否包含 "a" 和 "c"，返回布尔数组
    result = cat.isin(["a", "c"])
    # 期望的布尔数组
    expected = np.array([True, False, False], dtype=bool)
    # 断言结果数组与期望数组相等
    tm.assert_numpy_array_equal(expected, result)


# 使用 pytest 的参数化装饰器，定义测试函数 test_isin_cats_corner_cases，传入参数 value
@pytest.mark.parametrize("value", [[""], [None, ""], [pd.NaT, ""]])
def test_isin_cats_corner_cases(value):
    # GH36550
    # 创建一个分类数据 cat，包含一个空字符串 ""
    cat = pd.Categorical([""])
    # 调用 isin 方法，判断是否包含 value 参数中的值，返回布尔数组
    result = cat.isin(value)
    # 期望的布尔数组
    expected = np.array([True], dtype=bool)
    # 断言结果数组与期望数组相等
    tm.assert_numpy_array_equal(expected, result)


# 使用 pytest 的参数化装饰器，定义测试函数 test_isin_empty，传入参数 empty
@pytest.mark.parametrize("empty", [[], pd.Series(dtype=object), np.array([])])
def test_isin_empty(empty):
    # 创建一个分类数据 s，包含字符串 "a" 和 "b"
    s = pd.Categorical(["a", "b"])
    # 期望的布尔数组，判断 s 中的元素是否在 empty 参数中
    expected = np.array([False, False], dtype=bool)

    # 调用 isin 方法，判断 s 中的元素是否在 empty 参数中，返回布尔数组
    result = s.isin(empty)
    # 断言结果数组与期望数组相等
    tm.assert_numpy_array_equal(expected, result)


# 定义测试函数 test_diff
def test_diff():
    # 创建一个分类数据 ser，包含整数 1、2、3
    ser = pd.Series([1, 2, 3], dtype="category")

    # 错误消息字符串
    msg = "Convert to a suitable dtype"
    # 使用 pytest 的上下文管理器，断言调用 ser 的 diff 方法抛出 TypeError 异常，并匹配错误消息
    with pytest.raises(TypeError, match=msg):
        ser.diff()

    # 将 ser 转换为数据框 df，列名为 "A"
    df = ser.to_frame(name="A")
    # 使用 pytest 的上下文管理器，断言调用 df 的 diff 方法抛出 TypeError 异常，并匹配错误消息
    with pytest.raises(TypeError, match=msg):
        df.diff()


# 定义测试函数 test_hash_read_only_categorical
def test_hash_read_only_categorical():
    # GH#58481
    # 创建一个索引 idx，使用对象类型的索引，包含字符串 "a"、"b"、"c"
    idx = pd.Index(pd.Index(["a", "b", "c"], dtype="object").values)
    # 创建一个分类数据类型 cat，基于 idx
    cat = pd.CategoricalDtype(idx)
    # 创建一个系列 arr，包含字符串 "a"、"b"，数据类型为 cat
    arr = pd.Series(["a", "b"], dtype=cat).values
    # 断言 arr 的数据类型的哈希值相等
    assert hash(arr.dtype) == hash(arr.dtype)
```