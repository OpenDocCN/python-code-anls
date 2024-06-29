# `D:\src\scipysrc\pandas\pandas\_libs\testing.pyi`

```
# 从 typing 模块导入 Mapping 类型
from typing import Mapping

# 定义一个函数 assert_dict_equal，用于比较两个映射类型对象是否相等
def assert_dict_equal(a: Mapping, b: Mapping, compare_keys: bool = ...) -> bool:
    ...

# 定义一个函数 assert_almost_equal，用于比较两个数值类型对象是否几乎相等
def assert_almost_equal(
    a,  # 第一个比较对象
    b,  # 第二个比较对象
    rtol: float = ...,  # 相对容差，用于比较浮点数
    atol: float = ...,  # 绝对容差，用于比较浮点数
    check_dtype: bool = ...,  # 是否检查数据类型
    obj=...,  # 用于错误消息的比较对象
    lobj=...,  # 用于错误消息的左侧比较对象
    robj=...,  # 用于错误消息的右侧比较对象
    index_values=...,  # 比较对象的索引值
) -> bool:  # 返回比较结果的布尔值
    ...
```