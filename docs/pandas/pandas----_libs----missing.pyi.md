# `D:\src\scipysrc\pandas\pandas\_libs\missing.pyi`

```
# 导入 numpy 库，并且导入 typing 模块中的 npt 别名
import numpy as np
from numpy import typing as npt

# 定义 NAType 类，重载 __new__ 方法，但方法体未实现
class NAType:
    def __new__(cls, *args, **kwargs): ...

# NA 变量，用于表示 NAType 类的实例
NA: NAType

# 检查两个对象是否匹配 NA 值
# left: 左侧对象
# right: 右侧对象
# nan_matches_none: 是否将 NaN 视为 None 的匹配项，默认未指定
# 返回布尔值，表示是否匹配
def is_matching_na(
    left: object, right: object, nan_matches_none: bool = ...
) -> bool: ...

# 检查标量是否为正无穷
# val: 要检查的标量值
# 返回布尔值，表示是否为正无穷
def isposinf_scalar(val: object) -> bool: ...

# 检查标量是否为负无穷
# val: 要检查的标量值
# 返回布尔值，表示是否为负无穷
def isneginf_scalar(val: object) -> bool: ...

# 检查对象是否为 null 值
# val: 要检查的对象
# 返回布尔值，表示是否为 null
def checknull(val: object) -> bool: ...

# 检查 numpy 数组中的对象是否为 NA 值
# arr: 要检查的 numpy 数组
# 返回布尔数组，表示每个位置是否为 NA 值
def isnaobj(arr: np.ndarray) -> npt.NDArray[np.bool_]: ...

# 检查 numpy 数组中的数值对象是否为 NA 值
# values: 要检查的数值 numpy 数组
# 返回布尔数组，表示每个位置的数值是否为 NA 值
def is_numeric_na(values: np.ndarray) -> npt.NDArray[np.bool_]: ...
```