# `D:\src\scipysrc\pandas\pandas\tests\indexes\conftest.py`

```
# 导入必要的库：numpy 和 pytest
import numpy as np
import pytest

# 从 pandas 库中导入 Series 和 array
from pandas import (
    Series,
    array,
)


# 定义一个 pytest fixture 'sort'，接受参数为 None 或 False
@pytest.fixture(params=[None, False])
def sort(request):
    """
    'sort' 参数在 Index 的集合操作方法（如交集、并集等）中的有效取值。

    注意:
        不要将这个 fixture 与用于 concat 的 'sort' fixture 混淆。
        后者的参数是 [True, False]。

        我们不能将它们合并，因为在 Index 的集合操作方法中不允许 sort=True。
    """
    return request.param


# 定义一个 pytest fixture 'freq_sample'，接受参数为一组频率字符串
@pytest.fixture(params=["D", "3D", "-3D", "h", "2h", "-2h", "min", "2min", "s", "-3s"])
def freq_sample(request):
    """
    'freq' 参数在创建 date_range 和 timedelta_range 中使用的有效取值。
    """
    return request.param


# 定义一个 pytest fixture 'listlike_box'，接受参数为 list、tuple、np.array、pandas 的 array 或 Series
@pytest.fixture(params=[list, tuple, np.array, array, Series])
def listlike_box(request):
    """
    'listlike_box' 参数可以作为 searchsorted 的索引器传递的类型。
    """
    return request.param
```