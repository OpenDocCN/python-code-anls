# `D:\src\scipysrc\pandas\pandas\tests\arrays\integer\test_reduction.py`

```
# 导入必要的库
import numpy as np  # 导入NumPy库，用于数值计算
import pytest  # 导入pytest库，用于单元测试

import pandas as pd  # 导入Pandas库，用于数据处理
from pandas import (  # 从Pandas中导入DataFrame、Series、array等
    DataFrame,
    Series,
    array,
)
import pandas._testing as tm  # 导入Pandas测试工具模块

# 定义测试函数，测试Series对象的不同归约操作
@pytest.mark.parametrize(
    "op, expected",
    [
        ["sum", np.int64(3)],  # 求和操作，期望结果为整数3
        ["prod", np.int64(2)],  # 求乘积操作，期望结果为整数2
        ["min", np.int64(1)],   # 求最小值操作，期望结果为整数1
        ["max", np.int64(2)],   # 求最大值操作，期望结果为整数2
        ["mean", np.float64(1.5)],  # 求平均值操作，期望结果为浮点数1.5
        ["median", np.float64(1.5)],  # 求中位数操作，期望结果为浮点数1.5
        ["var", np.float64(0.5)],  # 求方差操作，期望结果为浮点数0.5
        ["std", np.float64(0.5**0.5)],  # 求标准差操作，期望结果为浮点数0.5的平方根
        ["skew", pd.NA],  # 求偏度操作，期望结果为缺失值
        ["kurt", pd.NA],  # 求峰度操作，期望结果为缺失值
        ["any", True],   # 检查是否有任意非零元素，期望结果为True
        ["all", True],   # 检查是否所有元素都为True，期望结果为True
    ],
)
def test_series_reductions(op, expected):
    # 创建一个包含整数1和2的Series对象，数据类型为Int64
    ser = Series([1, 2], dtype="Int64")
    # 执行指定的归约操作
    result = getattr(ser, op)()
    # 使用测试工具模块验证结果是否与期望值相等
    tm.assert_equal(result, expected)


# 定义测试函数，测试DataFrame对象的不同归约操作
@pytest.mark.parametrize(
    "op, expected",
    [
        ["sum", Series([3], index=["a"], dtype="Int64")],  # 求和操作，期望结果为包含索引"a"的Series对象
        ["prod", Series([2], index=["a"], dtype="Int64")],  # 求乘积操作，期望结果为包含索引"a"的Series对象
        ["min", Series([1], index=["a"], dtype="Int64")],   # 求最小值操作，期望结果为包含索引"a"的Series对象
        ["max", Series([2], index=["a"], dtype="Int64")],   # 求最大值操作，期望结果为包含索引"a"的Series对象
        ["mean", Series([1.5], index=["a"], dtype="Float64")],  # 求平均值操作，期望结果为包含索引"a"的Series对象
        ["median", Series([1.5], index=["a"], dtype="Float64")],  # 求中位数操作，期望结果为包含索引"a"的Series对象
        ["var", Series([0.5], index=["a"], dtype="Float64")],  # 求方差操作，期望结果为包含索引"a"的Series对象
        ["std", Series([0.5**0.5], index=["a"], dtype="Float64")],  # 求标准差操作，期望结果为包含索引"a"的Series对象
        ["skew", Series([pd.NA], index=["a"], dtype="Float64")],  # 求偏度操作，期望结果为包含索引"a"的Series对象
        ["kurt", Series([pd.NA], index=["a"], dtype="Float64")],  # 求峰度操作，期望结果为包含索引"a"的Series对象
        ["any", Series([True], index=["a"], dtype="boolean")],  # 检查是否有任意非零元素，期望结果为包含索引"a"的Series对象
        ["all", Series([True], index=["a"], dtype="boolean")],  # 检查是否所有元素都为True，期望结果为包含索引"a"的Series对象
    ],
)
def test_dataframe_reductions(op, expected):
    # 创建一个包含列"a"的DataFrame对象，列数据为整数数组
    df = DataFrame({"a": array([1, 2], dtype="Int64")})
    # 执行指定的归约操作
    result = getattr(df, op)()
    # 使用测试工具模块验证结果是否与期望值相等
    tm.assert_series_equal(result, expected)


# 定义测试函数，测试GroupBy对象的不同归约操作
@pytest.mark.parametrize(
    "op, expected",
    [
        ["sum", array([1, 3], dtype="Int64")],  # 求和操作，期望结果为整数数组
        ["prod", array([1, 3], dtype="Int64")],  # 求乘积操作，期望结果为整数数组
        ["min", array([1, 3], dtype="Int64")],   # 求最小值操作，期望结果为整数数组
        ["max", array([1, 3], dtype="Int64")],   # 求最大值操作，期望结果为整数数组
        ["mean", array([1, 3], dtype="Float64")],  # 求平均值操作，期望结果为浮点数数组
        ["median", array([1, 3], dtype="Float64")],  # 求中位数操作，期望结果为浮点数数组
        ["var", array([pd.NA], dtype="Float64")],  # 求方差操作，期望结果为浮点数数组
        ["std", array([pd.NA], dtype="Float64")],  # 求标准差操作，期望结果为浮点数数组
        ["skew", array([pd.NA], dtype="Float64")],  # 求偏度操作，期望结果为浮点数数组
        ["any", array([True, True], dtype="boolean")],  # 检查是否有任意非零元素，期望结果为布尔值数组
        ["all", array([True, True], dtype="boolean")],  # 检查是否所有元素都为True，期望结果为布尔值数组
    ],
)
def test_groupby_reductions(op, expected):
    # 创建一个DataFrame对象，包含列"A"和"B"，B列包括整数和空值
    df = DataFrame(
        {
            "A": ["a", "b", "b"],
            "B": array([1, None, 3], dtype="Int64"),
        }
    )
    # 对"A"列进行分组，并执行指定的归约操作
    result = getattr(df.groupby("A"), op)()
    # 创建一个期望的DataFrame对象，用于验证结果
    expected = DataFrame(expected, index=pd.Index(["a", "b"], name="A"), columns=["B"])
    # 使用测试工具模块验证结果是否与期望值相等
    tm.assert_frame_equal(result, expected)
    # 创建包含不同统计函数结果的列表，每个元素是一个包含索引为["B", "C"]的Series对象
    [
        # 计算数组的总和并创建Series对象
        ["sum", Series([4, 4], index=["B", "C"], dtype="Float64")],
        # 计算数组的乘积并创建Series对象
        ["prod", Series([3, 3], index=["B", "C"], dtype="Float64")],
        # 计算数组的最小值并创建Series对象
        ["min", Series([1, 1], index=["B", "C"], dtype="Float64")],
        # 计算数组的最大值并创建Series对象
        ["max", Series([3, 3], index=["B", "C"], dtype="Float64")],
        # 计算数组的平均值并创建Series对象
        ["mean", Series([2, 2], index=["B", "C"], dtype="Float64")],
        # 计算数组的中位数并创建Series对象
        ["median", Series([2, 2], index=["B", "C"], dtype="Float64")],
        # 计算数组的方差并创建Series对象
        ["var", Series([2, 2], index=["B", "C"], dtype="Float64")],
        # 计算数组的标准差并创建Series对象
        ["std", Series([2**0.5, 2**0.5], index=["B", "C"], dtype="Float64")],
        # 创建包含缺失值的偏斜度（skewness）Series对象
        ["skew", Series([pd.NA, pd.NA], index=["B", "C"], dtype="Float64")],
        # 创建包含缺失值的峰度（kurtosis）Series对象
        ["kurt", Series([pd.NA, pd.NA], index=["B", "C"], dtype="Float64")],
        # 创建包含布尔值True的Series对象
        ["any", Series([True, True, True], index=["A", "B", "C"], dtype="boolean")],
        # 创建包含布尔值True的Series对象
        ["all", Series([True, True, True], index=["A", "B", "C"], dtype="boolean")],
    ],
)
def test_mixed_reductions(op, expected, using_infer_string):
    # 如果操作是 "any" 或 "all"，并且使用了推断字符串，则将期望结果转换为布尔类型
    if op in ["any", "all"] and using_infer_string:
        expected = expected.astype("bool")
    
    # 创建一个 DataFrame 对象 df，包含三列：A列是字符串列表，B列是整数与空值，C列是带有NaN的整数类型数组
    df = DataFrame(
        {
            "A": ["a", "b", "b"],
            "B": [1, None, 3],
            "C": array([1, None, 3], dtype="Int64"),
        }
    )

    # 对列 'C' 执行指定操作 op，结果存储在 result 中
    result = getattr(df.C, op)()
    # 使用测试工具比较 result 和预期结果中的 'C' 列
    tm.assert_equal(result, expected["C"])

    # 如果操作是 "any" 或 "all"，则对整个 DataFrame 执行操作，否则对数值列进行操作
    if op in ["any", "all"]:
        result = getattr(df, op)()
    else:
        result = getattr(df, op)(numeric_only=True)
    # 使用测试工具比较 result 和预期结果
    tm.assert_series_equal(result, expected)
```