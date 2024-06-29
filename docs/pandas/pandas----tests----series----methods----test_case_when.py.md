# `D:\src\scipysrc\pandas\pandas\tests\series\methods\test_case_when.py`

```
# 导入所需的库
import numpy as np
import pytest

# 从 pandas 库中导入 DataFrame, Series, pd_array, date_range
from pandas import (
    DataFrame,
    Series,
    array as pd_array,
    date_range,
)
# 导入 pandas._testing 模块
import pandas._testing as tm

# 定义一个 fixture 函数，返回用于测试的基础 DataFrame
@pytest.fixture
def df():
    """
    base dataframe for testing
    """
    return DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

# 测试函数：当 caselist 不是列表时引发 ValueError
def test_case_when_caselist_is_not_a_list(df):
    """
    Raise ValueError if caselist is not a list.
    """
    msg = "The caselist argument should be a list; "
    msg += "instead got.+"
    with pytest.raises(TypeError, match=msg):  # GH39154
        df["a"].case_when(caselist=())

# 测试函数：如果没有提供 caselist，则引发 ValueError
def test_case_when_no_caselist(df):
    """
    Raise ValueError if no caselist is provided.
    """
    msg = "provide at least one boolean condition, "
    msg += "with a corresponding replacement."
    with pytest.raises(ValueError, match=msg):  # GH39154
        df["a"].case_when([])

# 测试函数：如果 caselist 的数量是奇数，则引发 ValueError
def test_case_when_odd_caselist(df):
    """
    Raise ValueError if no of caselist is odd.
    """
    msg = "Argument 0 must have length 2; "
    msg += "a condition and replacement; instead got length 3."

    with pytest.raises(ValueError, match=msg):
        df["a"].case_when([(df["a"].eq(1), 1, df.a.gt(1)])

# 测试函数：从 Series.mask 中引发错误
def test_case_when_raise_error_from_mask(df):
    """
    Raise Error from within Series.mask
    """
    msg = "Failed to apply condition0 and replacement0."
    with pytest.raises(ValueError, match=msg):
        df["a"].case_when([(df["a"].eq(1), [1, 2])])

# 测试函数：测试单个条件的输出
def test_case_when_single_condition(df):
    """
    Test output on a single condition.
    """
    result = Series([np.nan, np.nan, np.nan]).case_when([(df.a.eq(1), 1)])
    expected = Series([1, np.nan, np.nan])
    tm.assert_series_equal(result, expected)

# 测试函数：测试多个条件的输出
def test_case_when_multiple_conditions(df):
    """
    Test output when booleans are derived from a computation
    """
    result = Series([np.nan, np.nan, np.nan]).case_when(
        [(df.a.eq(1), 1), (Series([False, True, False]), 2)]
    )
    expected = Series([1, 2, np.nan])
    tm.assert_series_equal(result, expected)

# 测试函数：测试多个条件且替换为列表时的输出
def test_case_when_multiple_conditions_replacement_list(df):
    """
    Test output when replacement is a list
    """
    result = Series([np.nan, np.nan, np.nan]).case_when(
        [([True, False, False], 1), (df["a"].gt(1) & df["b"].eq(5), [1, 2, 3])]
    )
    expected = Series([1, 2, np.nan])
    tm.assert_series_equal(result, expected)

# 测试函数：测试多个条件且替换为扩展数据类型时的输出
def test_case_when_multiple_conditions_replacement_extension_dtype(df):
    """
    Test output when replacement has an extension dtype
    """
    result = Series([np.nan, np.nan, np.nan]).case_when(
        [
            ([True, False, False], 1),
            (df["a"].gt(1) & df["b"].eq(5), pd_array([1, 2, 3], dtype="Int64")),
        ],
    )
    expected = Series([1, 2, np.nan], dtype="Float64")
    tm.assert_series_equal(result, expected)

# 测试函数：测试多个条件且替换为 Series 时的输出
def test_case_when_multiple_conditions_replacement_series(df):
    """
    Test output when replacement is a Series
    """
    # 创建一个 Series 对象，包含三个 NaN 值，并调用 case_when 方法进行条件赋值
    result = Series([np.nan, np.nan, np.nan]).case_when(
        [
            # 第一个条件：根据给定的布尔数组和数值 1 进行条件赋值
            (np.array([True, False, False]), 1),
            # 第二个条件：根据 DataFrame df 中列 "a" 大于 1 和列 "b" 等于 5 的条件，
            # 使用 Series([1, 2, 3]) 进行条件赋值
            (df["a"].gt(1) & df["b"].eq(5), Series([1, 2, 3])),
        ],
    )
    # 创建一个期望的 Series 对象，包含预期的数值 [1, 2, NaN]
    expected = Series([1, 2, np.nan])
    # 使用断言函数确保 result 和 expected 的内容相等
    tm.assert_series_equal(result, expected)
# 定义一个测试函数，用于测试当索引不是 RangeIndex 时的输出
def test_case_when_non_range_index():
    # 使用随机数生成器创建一个新的随机数生成器实例，设定种子为 123
    rng = np.random.default_rng(seed=123)
    # 创建一个日期范围，从 "1/1/2000" 开始，包含 8 个时间点
    dates = date_range("1/1/2000", periods=8)
    # 创建一个 DataFrame，包含大小为 (8, 4) 的正态分布随机数，使用 dates 作为索引，列名为 ["A", "B", "C", "D"]
    df = DataFrame(
        rng.standard_normal(size=(8, 4)), index=dates, columns=["A", "B", "C", "D"]
    )
    # 对 DataFrame 中的 "A" 列应用 case_when 函数，如果 df.A 大于 0，则使用 df.B，否则使用 5
    result = Series(5, index=df.index, name="A").case_when([(df.A.gt(0), df.B)])
    # 创建预期的 Series，使用 df.A.gt(0) 作为 mask 条件，如果为真则使用 df.B，否则使用 5
    expected = df.A.mask(df.A.gt(0), df.B).where(df.A.gt(0), 5)
    # 使用测试框架的函数来断言 result 与 expected 是否相等
    tm.assert_series_equal(result, expected)


def test_case_when_callable():
    # 定义一个测试函数，测试在可调用对象上的输出
    """
    Test output on a callable
    """
    # 创建一个从 -2.5 到 2.5 的等差数列，包含 6 个点
    x = np.linspace(-2.5, 2.5, 6)
    # 将数列转换为 Series 对象
    ser = Series(x)
    # 对 Series 应用 case_when 函数，根据 caselist 中的条件和函数对 x 进行分段处理
    result = ser.case_when(
        caselist=[
            (lambda df: df < 0, lambda df: -df),  # 如果小于 0，则取负数
            (lambda df: df >= 0, lambda df: df),  # 如果大于等于 0，则保持不变
        ]
    )
    # 创建预期的数组，使用 np.piecewise 来根据条件对 x 进行分段处理
    expected = np.piecewise(x, [x < 0, x >= 0], [lambda x: -x, lambda x: x])
    # 使用测试框架的函数来断言 result 与 expected 是否相等
    tm.assert_series_equal(result, Series(expected))
```