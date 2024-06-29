# `D:\src\scipysrc\pandas\pandas\tests\indexes\multi\conftest.py`

```
# 导入 numpy 库并重命名为 np
import numpy as np
# 导入 pytest 库，用于编写和运行测试
import pytest

# 从 pandas 库中导入 Index 和 MultiIndex 类
from pandas import (
    Index,
    MultiIndex,
)

# 注释 pytest fixture，用于创建一个名为 idx 的多级索引对象
# 注意：这里的注释是对 "multi" 在顶层 "index" fixture 中的相同条目的说明
@pytest.fixture
def idx():
    # 创建主轴 Index，用于测试对象的一般功能
    major_axis = Index(["foo", "bar", "baz", "qux"])
    # 创建次轴 Index
    minor_axis = Index(["one", "two"])

    # 定义主轴和次轴的编码
    major_codes = np.array([0, 0, 1, 2, 3, 3])
    minor_codes = np.array([0, 1, 0, 1, 0, 1])
    # 定义索引的名称列表
    index_names = ["first", "second"]
    
    # 创建 MultiIndex 对象 mi，使用给定的 levels、codes 和 names 参数
    # verify_integrity 参数设置为 False，允许创建不一致的 MultiIndex 对象
    mi = MultiIndex(
        levels=[major_axis, minor_axis],
        codes=[major_codes, minor_codes],
        names=index_names,
        verify_integrity=False,
    )
    
    # 返回创建的 MultiIndex 对象 mi 作为 fixture 的结果
    return mi
```