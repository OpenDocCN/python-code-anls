# `D:\src\scipysrc\numpy\numpy\typing\tests\data\fail\flatiter.pyi`

```
from typing import Any

import numpy as np
import numpy._typing as npt


class Index:
    def __index__(self) -> int:
        ...

# 定义一个类型为 flatiter 的变量 a，其元素类型为 float64 的 numpy 数组的迭代器类型
a: np.flatiter[npt.NDArray[np.float64]]
# 定义一个支持传递数组的类型变量 supports_array
supports_array: npt._SupportsArray[np.dtype[np.float64]]

# 尝试给属性赋值会引发错误，因为 flatiter 的属性 base、coords 和 index 是只读的
a.base = Any  # E: Property "base" defined in "flatiter" is read-only
a.coords = Any  # E: Property "coords" defined in "flatiter" is read-only
a.index = Any  # E: Property "index" defined in "flatiter" is read-only
# 调用 flatiter 的 copy 方法，传递了一个不期望的关键字参数 'order'
a.copy(order='C')  # E: Unexpected keyword argument

# 注意：与 `ndarray.__getitem__` 不同，`flatiter` 的对应方法不接受具有 `__array__` 或 `__index__` 协议的对象；
# 布尔索引是有问题的（gh-17175）
# 下面的三行尝试使用布尔索引、Index 类型的对象和 supports_array 进行索引，但都引发错误
a[np.bool()]  # E: No overload variant of "__getitem__"
a[Index()]  # E: No overload variant of "__getitem__"
a[supports_array]  # E: No overload variant of "__getitem__"
```