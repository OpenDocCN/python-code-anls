# `D:\src\scipysrc\pandas\pandas\tests\dtypes\cast\test_infer_datetimelike.py`

```
import numpy as np  # 导入 NumPy 库，通常用于数值计算
import pytest  # 导入 Pytest 库，用于编写和运行测试用例

from pandas import (  # 从 Pandas 库中导入以下几个模块
    DataFrame,  # DataFrame：用于处理二维表格数据
    NaT,  # NaT：表示不确定或缺失的时间戳
    Series,  # Series：用于处理一维标记数据结构
    Timestamp,  # Timestamp：表示时间戳的数据类型
)

@pytest.mark.parametrize(  # 使用 Pytest 的参数化装饰器来标记测试函数及其参数
    "data,exp_size",  # 参数化的参数名字和期望大小
    [  # 参数化的数据集合
        # see gh-16362.
        ([[NaT, "a", "b", 0], [NaT, "b", "c", 1]], 8),  # 第一个测试数据，期望大小为 8
        ([[NaT, "a", 0], [NaT, "b", 1]], 6),  # 第二个测试数据，期望大小为 6
    ],
)
def test_maybe_infer_to_datetimelike_df_construct(data, exp_size):
    result = DataFrame(np.array(data))  # 使用给定的数据创建 Pandas DataFrame 对象
    assert result.size == exp_size  # 断言 DataFrame 的大小是否等于期望的大小


def test_maybe_infer_to_datetimelike_ser_construct():
    # see gh-19671.
    result = Series(["M1701", Timestamp("20130101")])  # 使用指定数据创建 Pandas Series 对象
    assert result.dtype.kind == "O"  # 断言 Series 对象的数据类型的种类是否为对象型
```