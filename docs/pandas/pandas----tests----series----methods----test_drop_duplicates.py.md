# `D:\src\scipysrc\pandas\pandas\tests\series\methods\test_drop_duplicates.py`

```
# 导入必要的库
import numpy as np
import pytest

# 导入 pandas 库，并从中导入 Categorical 和 Series 类
import pandas as pd
from pandas import (
    Categorical,
    Series,
)
# 导入 pandas 测试模块
import pandas._testing as tm

# 使用 pytest 的 parametrize 装饰器定义参数化测试
@pytest.mark.parametrize(
    "keep, expected",
    [
        ("first", Series([False, False, False, False, True, True, False])),  # 预期 keep="first" 时的结果
        ("last", Series([False, True, True, False, False, False, False])),  # 预期 keep="last" 时的结果
        (False, Series([False, True, True, False, True, True, False])),     # 预期 keep=False 时的结果
    ],
)
# 定义测试函数 test_drop_duplicates，接受参数 any_numpy_dtype, keep, expected
def test_drop_duplicates(any_numpy_dtype, keep, expected):
    # 创建 Series 对象 tc，包含指定数据和数据类型
    tc = Series([1, 0, 3, 5, 3, 0, 4], dtype=np.dtype(any_numpy_dtype))

    # 如果数据类型为布尔型，跳过此测试
    if tc.dtype == "bool":
        pytest.skip("tested separately in test_drop_duplicates_bool")

    # 断言 Series 对象的重复项结果与预期结果一致
    tm.assert_series_equal(tc.duplicated(keep=keep), expected)
    # 断言 Series 对象去重后的结果与预期结果一致
    tm.assert_series_equal(tc.drop_duplicates(keep=keep), tc[~expected])
    # 创建 Series 对象的浅拷贝 sc
    sc = tc.copy()
    # 调用 inplace 模式下的 drop_duplicates 方法
    return_value = sc.drop_duplicates(keep=keep, inplace=True)
    # 断言 inplace 操作返回值为 None
    assert return_value is None
    # 断言修改后的 Series 对象 sc 与预期结果一致
    tm.assert_series_equal(sc, tc[~expected])


@pytest.mark.parametrize(
    "keep, expected",
    [
        ("first", [False, False, True, True]),   # 预期 keep="first" 时的结果
        ("last", [True, True, False, False]),    # 预期 keep="last" 时的结果
        (False, [True, True, True, True]),       # 预期 keep=False 时的结果
    ],
)
# 定义测试函数 test_drop_duplicates_bool，接受参数 keep, expected
def test_drop_duplicates_bool(keep, expected):
    # 创建布尔型 Series 对象 tc
    tc = Series([True, False, True, False])
    # 创建预期结果的 Series 对象
    expected = Series(expected)
    # 断言 Series 对象的重复项结果与预期结果一致
    tm.assert_series_equal(tc.duplicated(keep=keep), expected)
    # 断言 Series 对象去重后的结果与预期结果一致
    tm.assert_series_equal(tc.drop_duplicates(keep=keep), tc[~expected])
    # 创建 Series 对象的浅拷贝 sc
    sc = tc.copy()
    # 调用 inplace 模式下的 drop_duplicates 方法
    return_value = sc.drop_duplicates(keep=keep, inplace=True)
    # 断言修改后的 Series 对象 sc 与预期结果一致
    tm.assert_series_equal(sc, tc[~expected])
    # 断言 inplace 操作返回值为 None
    assert return_value is None


@pytest.mark.parametrize("values", [[], list(range(5))])
# 定义测试函数 test_drop_duplicates_no_duplicates，接受参数 any_numpy_dtype, keep, values
def test_drop_duplicates_no_duplicates(any_numpy_dtype, keep, values):
    # 创建指定数据和数据类型的 Series 对象 tc
    tc = Series(values, dtype=np.dtype(any_numpy_dtype))
    # 创建全为 False 的预期结果 Series 对象
    expected = Series([False] * len(tc), dtype="bool")

    # 如果数据类型为布尔型，只保留前两个元素
    if tc.dtype == "bool":
        # 0 -> False and 1-> True
        # any other value would be duplicated
        tc = tc[:2]
        expected = expected[:2]

    # 断言 Series 对象的重复项结果与预期结果一致
    tm.assert_series_equal(tc.duplicated(keep=keep), expected)

    # 调用 drop_duplicates 方法获取去重后的结果
    result_dropped = tc.drop_duplicates(keep=keep)
    # 断言去重后的结果与原始 Series 对象一致
    tm.assert_series_equal(result_dropped, tc)

    # 验证浅拷贝
    assert result_dropped is not tc


# 定义测试类 TestSeriesDropDuplicates
class TestSeriesDropDuplicates:
    # 定义 pytest fixture，参数化测试中的数据类型
    @pytest.fixture(
        params=["int_", "uint", "float64", "str_", "timedelta64[h]", "datetime64[D]"]
    )
    def dtype(self, request):
        return request.param

    # 定义 pytest fixture，创建不使用的分类 Series
    @pytest.fixture
    def cat_series_unused_category(self, dtype, ordered):
        # 创建分类数组 cat_array
        cat_array = np.array([1, 2, 3, 4, 5], dtype=np.dtype(dtype))

        # 创建输入数组 input1
        input1 = np.array([1, 2, 3, 3], dtype=np.dtype(dtype))
        # 创建分类对象 cat
        cat = Categorical(input1, categories=cat_array, ordered=ordered)
        # 创建 Series 对象 tc1
        tc1 = Series(cat)
        return tc1
    # 测试用例：对非布尔型的分类序列进行去重操作，使用 `cat_series_unused_category` 作为输入数据
    def test_drop_duplicates_categorical_non_bool(self, cat_series_unused_category):
        # 从参数中获取分类序列 `cat_series_unused_category`
        tc1 = cat_series_unused_category

        # 预期的去重后的序列，是一个包含布尔值的 `Series`
        expected = Series([False, False, False, True])

        # 执行去重操作，返回去重后的结果序列
        result = tc1.duplicated()
        # 使用测试工具比较结果序列和预期序列是否相等
        tm.assert_series_equal(result, expected)

        # 执行去重操作，并返回去重后的序列
        result = tc1.drop_duplicates()
        # 使用测试工具比较结果序列和预期的非重复部分序列是否相等
        tm.assert_series_equal(result, tc1[~expected])

        # 复制分类序列 `tc1` 到 `sc`
        sc = tc1.copy()
        # 在原地进行去重操作，期望返回值为 `None`
        return_value = sc.drop_duplicates(inplace=True)
        # 断言返回值是否为 `None`
        assert return_value is None
        # 使用测试工具比较原地去重后的序列 `sc` 和预期的非重复部分序列是否相等
        tm.assert_series_equal(sc, tc1[~expected])

    # 测试用例：对非布尔型的分类序列进行保留最后一个重复值的去重操作，使用 `cat_series_unused_category` 作为输入数据
    def test_drop_duplicates_categorical_non_bool_keeplast(
        self, cat_series_unused_category
    ):
        # 从参数中获取分类序列 `cat_series_unused_category`
        tc1 = cat_series_unused_category

        # 预期的保留最后一个重复值的去重后的序列，是一个包含布尔值的 `Series`
        expected = Series([False, False, True, False])

        # 执行保留最后一个重复值的去重操作，返回结果序列
        result = tc1.duplicated(keep="last")
        # 使用测试工具比较结果序列和预期序列是否相等
        tm.assert_series_equal(result, expected)

        # 执行保留最后一个重复值的去重操作，并返回去重后的序列
        result = tc1.drop_duplicates(keep="last")
        # 使用测试工具比较结果序列和预期的非重复部分序列是否相等
        tm.assert_series_equal(result, tc1[~expected])

        # 复制分类序列 `tc1` 到 `sc`
        sc = tc1.copy()
        # 在原地进行保留最后一个重复值的去重操作，期望返回值为 `None`
        return_value = sc.drop_duplicates(keep="last", inplace=True)
        # 断言返回值是否为 `None`
        assert return_value is None
        # 使用测试工具比较原地去重后的序列 `sc` 和预期的非重复部分序列是否相等
        tm.assert_series_equal(sc, tc1[~expected])

    # 测试用例：对非布尔型的分类序列进行保留第一个重复值的去重操作，使用 `cat_series_unused_category` 作为输入数据
    def test_drop_duplicates_categorical_non_bool_keepfalse(
        self, cat_series_unused_category
    ):
        # 从参数中获取分类序列 `cat_series_unused_category`
        tc1 = cat_series_unused_category

        # 预期的保留第一个重复值的去重后的序列，是一个包含布尔值的 `Series`
        expected = Series([False, False, True, True])

        # 执行保留第一个重复值的去重操作，返回结果序列
        result = tc1.duplicated(keep=False)
        # 使用测试工具比较结果序列和预期序列是否相等
        tm.assert_series_equal(result, expected)

        # 执行保留第一个重复值的去重操作，并返回去重后的序列
        result = tc1.drop_duplicates(keep=False)
        # 使用测试工具比较结果序列和预期的非重复部分序列是否相等
        tm.assert_series_equal(result, tc1[~expected])

        # 复制分类序列 `tc1` 到 `sc`
        sc = tc1.copy()
        # 在原地进行保留第一个重复值的去重操作，期望返回值为 `None`
        return_value = sc.drop_duplicates(keep=False, inplace=True)
        # 断言返回值是否为 `None`
        assert return_value is None
        # 使用测试工具比较原地去重后的序列 `sc` 和预期的非重复部分序列是否相等
        tm.assert_series_equal(sc, tc1[~expected])

    # Pytest fixture：生成一个分类序列 `cat_series`，根据 `dtype` 和 `ordered` 参数设定
    @pytest.fixture
    def cat_series(self, dtype, ordered):
        # 定义一个分类数组 `cat_array`，包含一组分类值
        cat_array = np.array([1, 2, 3, 4, 5], dtype=np.dtype(dtype))

        # 定义输入数组 `input2`，包含一组值，然后基于 `cat_array` 创建一个分类对象 `cat`
        input2 = np.array([1, 2, 3, 5, 3, 2, 4], dtype=np.dtype(dtype))
        cat = Categorical(input2, categories=cat_array, ordered=ordered)

        # 将分类对象 `cat` 转换为 `Series` 类型并返回
        tc2 = Series(cat)
        return tc2

    # 测试用例：对非布尔型的分类序列进行去重操作，使用 `cat_series` 作为输入数据
    def test_drop_duplicates_categorical_non_bool2(self, cat_series):
        # 从参数中获取分类序列 `cat_series`
        tc2 = cat_series

        # 预期的去重后的序列，是一个包含布尔值的 `Series`
        expected = Series([False, False, False, False, True, True, False])

        # 执行去重操作，返回去重后的结果序列
        result = tc2.duplicated()
        # 使用测试工具比较结果序列和预期序列是否相等
        tm.assert_series_equal(result, expected)

        # 执行去重操作，并返回去重后的序列
        result = tc2.drop_duplicates()
        # 使用测试工具比较结果序列和预期的非重复部分序列是否相等
        tm.assert_series_equal(result, tc2[~expected])

        # 复制分类序列 `tc2` 到 `sc`
        sc = tc2.copy()
        # 在原地进行去重操作，期望返回值为 `None`
        return_value = sc.drop_duplicates(inplace=True)
        # 断言返回值是否为 `None`
        assert return_value is None
        # 使用测试工具比较原地去重后的序列 `sc` 和预期的非重复部分序列是否相等
        tm.assert_series_equal(sc, tc2[~expected])
    # 测试删除重复的分类数据（非布尔类型），保留最后一个重复项
    def test_drop_duplicates_categorical_non_bool2_keeplast(self, cat_series):
        # 将传入的 cat_series 赋给 tc2
        tc2 = cat_series

        # 预期的结果是一个 Series，包含布尔值，表示是否为重复项的期望结果
        expected = Series([False, True, True, False, False, False, False])

        # 使用 keep="last" 参数计算出结果
        result = tc2.duplicated(keep="last")
        # 断言计算结果与预期结果相等
        tm.assert_series_equal(result, expected)

        # 使用 drop_duplicates 方法删除重复项，保留最后一个重复项
        result = tc2.drop_duplicates(keep="last")
        # 断言处理后的结果与预期的非重复项匹配
        tm.assert_series_equal(result, tc2[~expected])

        # 复制 tc2 到 sc
        sc = tc2.copy()
        # 在原地（inplace）删除重复项，保留最后一个重复项，并获取返回值
        return_value = sc.drop_duplicates(keep="last", inplace=True)
        # 断言返回值为 None
        assert return_value is None
        # 断言处理后的 sc 与预期的非重复项匹配
        tm.assert_series_equal(sc, tc2[~expected])

    # 测试删除重复的分类数据（非布尔类型），保留第一个重复项
    def test_drop_duplicates_categorical_non_bool2_keepfalse(self, cat_series):
        # 将传入的 cat_series 赋给 tc2
        tc2 = cat_series

        # 预期的结果是一个 Series，包含布尔值，表示是否为重复项的期望结果
        expected = Series([False, True, True, False, True, True, False])

        # 使用 keep=False 参数计算出结果
        result = tc2.duplicated(keep=False)
        # 断言计算结果与预期结果相等
        tm.assert_series_equal(result, expected)

        # 使用 drop_duplicates 方法删除所有重复项，保留第一个重复项
        result = tc2.drop_duplicates(keep=False)
        # 断言处理后的结果与预期的非重复项匹配
        tm.assert_series_equal(result, tc2[~expected])

        # 复制 tc2 到 sc
        sc = tc2.copy()
        # 在原地（inplace）删除所有重复项，保留第一个重复项，并获取返回值
        return_value = sc.drop_duplicates(keep=False, inplace=True)
        # 断言返回值为 None
        assert return_value is None
        # 断言处理后的 sc 与预期的非重复项匹配
        tm.assert_series_equal(sc, tc2[~expected])

    # 测试删除重复的布尔类型分类数据
    def test_drop_duplicates_categorical_bool(self, ordered):
        # 创建一个包含布尔类型数据的 Series，使用 ordered 参数指定顺序
        tc = Series(
            Categorical(
                [True, False, True, False], categories=[True, False], ordered=ordered
            )
        )

        # 预期的结果是一个 Series，包含布尔值，表示是否为重复项的期望结果
        expected = Series([False, False, True, True])

        # 计算所有重复项，并与预期结果进行断言
        tm.assert_series_equal(tc.duplicated(), expected)
        # 使用 drop_duplicates 方法删除所有重复项，并与预期的非重复项进行断言
        tm.assert_series_equal(tc.drop_duplicates(), tc[~expected])

        # 复制 tc 到 sc
        sc = tc.copy()
        # 在原地（inplace）删除所有重复项，并获取返回值
        return_value = sc.drop_duplicates(inplace=True)
        # 断言返回值为 None
        assert return_value is None
        # 断言处理后的 sc 与预期的非重复项匹配
        tm.assert_series_equal(sc, tc[~expected])

        # 计算保留最后一个重复项的结果，并与预期结果进行断言
        expected = Series([True, True, False, False])
        tm.assert_series_equal(tc.duplicated(keep="last"), expected)
        # 使用 drop_duplicates 方法删除重复项，保留最后一个重复项，并与预期的非重复项进行断言
        tm.assert_series_equal(tc.drop_duplicates(keep="last"), tc[~expected])

        # 复制 tc 到 sc
        sc = tc.copy()
        # 在原地（inplace）删除重复项，保留最后一个重复项，并获取返回值
        return_value = sc.drop_duplicates(keep="last", inplace=True)
        # 断言返回值为 None
        assert return_value is None
        # 断言处理后的 sc 与预期的非重复项匹配
        tm.assert_series_equal(sc, tc[~expected])

        # 计算删除所有重复项的结果，并与预期结果进行断言
        expected = Series([True, True, True, True])
        tm.assert_series_equal(tc.duplicated(keep=False), expected)
        # 使用 drop_duplicates 方法删除所有重复项，并与预期的非重复项进行断言
        tm.assert_series_equal(tc.drop_duplicates(keep=False), tc[~expected])

        # 复制 tc 到 sc
        sc = tc.copy()
        # 在原地（inplace）删除所有重复项，并获取返回值
        return_value = sc.drop_duplicates(keep=False, inplace=True)
        # 断言返回值为 None
        assert return_value is None
        # 断言处理后的 sc 与预期的非重复项匹配
        tm.assert_series_equal(sc, tc[~expected])

    # 测试删除包含 NA 值的布尔类型分类数据
    def test_drop_duplicates_categorical_bool_na(self, nulls_fixture):
        # 创建一个包含 NA 值的布尔类型分类数据的 Series
        ser = Series(
            Categorical(
                [True, False, True, False, nulls_fixture],
                categories=[True, False],
                ordered=True,
            )
        )

        # 使用 drop_duplicates 方法删除所有重复项，并与预期的结果进行断言
        result = ser.drop_duplicates()
        expected = Series(
            Categorical([True, False, np.nan], categories=[True, False], ordered=True),
            index=[0, 1, 4],
        )
        tm.assert_series_equal(result, expected)
    # 定义一个测试方法，用于测试忽略索引去重功能
    def test_drop_duplicates_ignore_index(self):
        # 标记此测试相关的 GitHub issue 编号为 #48304
        ser = Series([1, 2, 2, 3])
        # 对序列进行去重操作，忽略索引
        result = ser.drop_duplicates(ignore_index=True)
        # 预期的去重后的序列
        expected = Series([1, 2, 3])
        # 断言操作：比较 result 和 expected 是否相等
        tm.assert_series_equal(result, expected)

    # 定义一个测试方法，用于测试带有 Arrow 数据类型的重复值去重
    def test_duplicated_arrow_dtype(self):
        # 如果没有安装 pyarrow 库，则跳过此测试
        pytest.importorskip("pyarrow")
        # 创建一个布尔类型的序列，使用 pyarrow 指定的数据类型
        ser = Series([True, False, None, False], dtype="bool[pyarrow]")
        # 对序列进行去重操作
        result = ser.drop_duplicates()
        # 预期的去重后的序列，使用相同的 Arrow 数据类型
        expected = Series([True, False, None], dtype="bool[pyarrow]")
        # 断言操作：比较 result 和 expected 是否相等
        tm.assert_series_equal(result, expected)

    # 定义一个测试方法，用于测试带有 Arrow 字符串数据类型的重复值去重
    def test_drop_duplicates_arrow_strings(self):
        # 标记此测试相关的 GitHub issue 编号为 #54904
        # 导入 pyarrow 库，如果没有安装则跳过此测试
        pa = pytest.importorskip("pyarrow")
        # 创建一个字符串类型的序列，使用 pandas 支持的 Arrow 数据类型
        ser = Series(["a", "a"], dtype=pd.ArrowDtype(pa.string()))
        # 对序列进行去重操作
        result = ser.drop_duplicates()
        # 预期的去重后的序列，使用相同的 Arrow 字符串数据类型
        expecetd = Series(["a"], dtype=pd.ArrowDtype(pa.string()))
        # 断言操作：比较 result 和 expecetd 是否相等
        tm.assert_series_equal(result, expecetd)
```