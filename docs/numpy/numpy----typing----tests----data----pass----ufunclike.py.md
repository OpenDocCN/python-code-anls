# `.\numpy\numpy\typing\tests\data\pass\ufunclike.py`

```
from __future__ import annotations
from typing import Any, Optional
import numpy as np

class Object:
    # 定义 ceil 魔术方法，返回对象自身
    def __ceil__(self) -> Object:
        return self

    # 定义 floor 魔术方法，返回对象自身
    def __floor__(self) -> Object:
        return self

    # 定义 >= 比较魔术方法，始终返回 True
    def __ge__(self, value: object) -> bool:
        return True

    # 定义数组转换魔术方法，将对象转换为包含自身的 numpy 数组
    def __array__(self, dtype: Optional[np.typing.DTypeLike] = None,
                  copy: Optional[bool] = None) -> np.ndarray[Any, np.dtype[np.object_]]:
        ret = np.empty((), dtype=object)
        ret[()] = self
        return ret

# 创建不同类型的数组样本
AR_LIKE_b = [True, True, False]
AR_LIKE_u = [np.uint32(1), np.uint32(2), np.uint32(3)]
AR_LIKE_i = [1, 2, 3]
AR_LIKE_f = [1.0, 2.0, 3.0]
AR_LIKE_O = [Object(), Object(), Object()]

# 创建指定类型的空字符串数组
AR_U: np.ndarray[Any, np.dtype[np.str_]] = np.zeros(3, dtype="U5")

# 使用 np.fix 函数处理各种类型的数组样本
np.fix(AR_LIKE_b)  # 四舍五入为最接近的整数
np.fix(AR_LIKE_u)  # 四舍五入为最接近的整数
np.fix(AR_LIKE_i)  # 四舍五入为最接近的整数
np.fix(AR_LIKE_f)  # 四舍五入为最接近的整数
np.fix(AR_LIKE_O)  # 四舍五入为最接近的整数
np.fix(AR_LIKE_f, out=AR_U)  # 将结果存储到指定的字符串数组中

# 使用 np.isposinf 函数检查数组中元素是否为正无穷
np.isposinf(AR_LIKE_b)
np.isposinf(AR_LIKE_u)
np.isposinf(AR_LIKE_i)
np.isposinf(AR_LIKE_f)
np.isposinf(AR_LIKE_f, out=AR_U)

# 使用 np.isneginf 函数检查数组中元素是否为负无穷
np.isneginf(AR_LIKE_b)
np.isneginf(AR_LIKE_u)
np.isneginf(AR_LIKE_i)
np.isneginf(AR_LIKE_f)
np.isneginf(AR_LIKE_f, out=AR_U)
```