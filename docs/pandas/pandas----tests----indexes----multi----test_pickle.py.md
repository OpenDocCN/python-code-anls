# `D:\src\scipysrc\pandas\pandas\tests\indexes\multi\test_pickle.py`

```
import pytest
# 导入 pytest 模块，用于进行单元测试和断言

from pandas import MultiIndex
# 从 pandas 库中导入 MultiIndex 类，用于创建多层索引对象


def test_pickle_compat_construction():
    # 定义单元测试函数 test_pickle_compat_construction，用于测试 pickle 兼容性构造
    
    # 使用 pytest.raises 断言捕获 TypeError 异常，并检查异常消息是否为 "Must pass both levels and codes"
    with pytest.raises(TypeError, match="Must pass both levels and codes"):
        # 在此上下文中，期望调用 MultiIndex() 构造函数会抛出 TypeError 异常，并匹配指定的异常消息
        MultiIndex()
```