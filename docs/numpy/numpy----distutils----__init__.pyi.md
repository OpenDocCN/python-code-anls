# `D:\src\scipysrc\numpy\numpy\distutils\__init__.pyi`

```
# 从 typing 模块导入 Any 类型
from typing import Any

# TODO: 当完整的 numpy 命名空间被定义后移除此函数
# 定义一个特殊函数 __getattr__，用于动态获取对象的属性
def __getattr__(name: str) -> Any: ...
```