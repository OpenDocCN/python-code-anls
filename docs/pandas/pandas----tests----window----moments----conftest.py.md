# `D:\src\scipysrc\pandas\pandas\tests\window\moments\conftest.py`

```
# 导入 itertools 模块，用于生成迭代器的工具函数
import itertools

# 导入 numpy 库，并使用 np 别名
import numpy as np

# 导入 pytest 测试框架
import pytest

# 从 pandas 库中导入 DataFrame 和 Series 类，以及 notna 函数
from pandas import (
    DataFrame,
    Series,
    notna,
)


# 定义函数 create_series，返回多个 Series 对象的列表
def create_series():
    return [
        Series(dtype=np.float64, name="a"),  # 创建一个指定数据类型和名称的 Series 对象
        Series([np.nan] * 5),  # 创建一个包含5个 NaN 值的 Series 对象
        Series([1.0] * 5),  # 创建一个包含5个 1.0 值的 Series 对象
        Series(range(5, 0, -1)),  # 创建一个倒序排列的 Series 对象
        Series(range(5)),  # 创建一个顺序排列的 Series 对象
        Series([np.nan, 1.0, np.nan, 1.0, 1.0]),  # 创建一个包含 NaN 和 1.0 的 Series 对象
        Series([np.nan, 1.0, np.nan, 2.0, 3.0]),  # 创建一个包含 NaN、1.0、2.0、3.0 的 Series 对象
        Series([np.nan, 1.0, np.nan, 3.0, 2.0]),  # 创建一个包含 NaN、1.0、3.0、2.0 的 Series 对象
    ]


# 定义函数 create_dataframes，返回多个 DataFrame 对象的列表
def create_dataframes():
    return [
        DataFrame(columns=["a", "a"]),  # 创建一个指定列名的空 DataFrame 对象
        DataFrame(np.arange(15).reshape((5, 3)), columns=["a", "a", 99]),  # 创建一个包含特定数据和列名的 DataFrame 对象
    ] + [DataFrame(s) for s in create_series()]  # 将多个 Series 对象转换为 DataFrame 并返回列表


# 定义函数 is_constant，判断输入的 Series 或 DataFrame 对象是否所有值相同
def is_constant(x):
    values = x.values.ravel("K")  # 将 Series 或 DataFrame 的值展平为一维数组
    return len(set(values[notna(values)])) == 1  # 检查非 NaN 值的集合长度是否为1，即所有值是否相同


# 定义 fixture consistent_data，返回满足 is_constant 条件的 Series 或 DataFrame 对象
@pytest.fixture(
    params=(
        obj
        for obj in itertools.chain(create_series(), create_dataframes())
        if is_constant(obj)
    ),
)
def consistent_data(request):
    return request.param


# 定义 fixture series_data，返回 create_series 函数生成的 Series 对象
@pytest.fixture(params=create_series())
def series_data(request):
    return request.param


# 定义 fixture all_data，返回 create_series 和 create_dataframes 生成的所有 Series 和 DataFrame 对象
@pytest.fixture(params=itertools.chain(create_series(), create_dataframes()))
def all_data(request):
    """
    测试:
        - 空的 Series / DataFrame
        - 全为 NaN 的 Series / DataFrame
        - 所有值相同的 Series / DataFrame
        - 递减排序的 Series
        - 递增排序的 Series
        - 包含 NaN 值且所有值相同的 Series
        - 包含 NaN 值且递增排序的 Series
        - 包含 NaN 值且递减排序的 Series
    """
    return request.param


# 定义 fixture min_periods，返回包含0和2的参数列表
@pytest.fixture(params=[0, 2])
def min_periods(request):
    return request.param
```