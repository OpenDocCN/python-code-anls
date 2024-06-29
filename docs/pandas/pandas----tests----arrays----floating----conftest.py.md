# `D:\src\scipysrc\pandas\pandas\tests\arrays\floating\conftest.py`

```
# 导入必要的库
import numpy as np  # 导入 NumPy 库
import pytest  # 导入 Pytest 库

import pandas as pd  # 导入 Pandas 库
from pandas.core.arrays.floating import (  # 从 Pandas 浮点数组模块中导入以下类型
    Float32Dtype,  # 32位浮点数类型
    Float64Dtype,  # 64位浮点数类型
)


@pytest.fixture(params=[Float32Dtype, Float64Dtype])
def dtype(request):
    """返回一个参数化的 fixture，根据 request 中的浮点 'dtype'"""
    return request.param()


@pytest.fixture
def data(dtype):
    """返回一个 'data' 数组的 fixture，根据参数化的浮点 'dtype'"""
    return pd.array(
        list(np.arange(0.1, 0.9, 0.1))  # 生成从0.1到0.8的步长为0.1的数组
        + [pd.NA]  # 添加一个缺失值
        + list(np.arange(1, 9.8, 0.1))  # 继续生成从1到9.7的步长为0.1的数组
        + [pd.NA]  # 再添加一个缺失值
        + [9.9, 10.0],  # 最后添加两个额外的数值
        dtype=dtype,  # 设置数据类型为参数化的浮点 'dtype'
    )


@pytest.fixture
def data_missing(dtype):
    """
    返回一个包含缺失数据的数组的 fixture，根据参数化的浮点 'dtype'。
    """
    return pd.array([np.nan, 0.1], dtype=dtype)  # 返回一个包含 NaN 和 0.1 的 Pandas 数组


@pytest.fixture(params=["data", "data_missing"])
def all_data(request, data, data_missing):
    """返回一个参数化的 fixture，根据 request 中的 'data' 或 'data_missing' 数组。

    用于测试带有和不带有缺失值的数据类型转换。
    """
    if request.param == "data":
        return data  # 如果请求的参数是 'data'，返回 data 数组
    elif request.param == "data_missing":
        return data_missing  # 如果请求的参数是 'data_missing'，返回 data_missing 数组
```