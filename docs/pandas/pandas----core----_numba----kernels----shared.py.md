# `D:\src\scipysrc\pandas\pandas\core\_numba\kernels\shared.py`

```
# 导入将来的类型注解支持
from __future__ import annotations

# 导入类型检查相关模块
from typing import TYPE_CHECKING

# 导入 numba 模块
import numba

# 如果正在进行类型检查
if TYPE_CHECKING:
    # 导入 numpy 作为 np
    import numpy as np

# 使用 numba.jit 装饰器装饰函数 is_monotonic_increasing
@numba.jit(
    # 错误: Any? 不可调用
    numba.boolean(numba.int64[:]),  # type: ignore[misc]
    nopython=True,  # 在无 Python 模式下编译
    nogil=True,     # 释放全局解释器锁
    parallel=False,  # 不并行处理
)
# 定义函数 is_monotonic_increasing，参数为 bounds 数组，返回布尔值
def is_monotonic_increasing(bounds: np.ndarray) -> bool:
    """Check if int64 values are monotonically increasing."""
    # 获取 bounds 数组长度
    n = len(bounds)
    # 如果数组长度小于 2，则认为是单调递增的
    if n < 2:
        return True
    # 初始化 prev 为数组第一个元素
    prev = bounds[0]
    # 遍历数组中的元素
    for i in range(1, n):
        # 当前元素赋值给 cur
        cur = bounds[i]
        # 如果当前元素 cur 小于前一个元素 prev，则不是单调递增的，返回 False
        if cur < prev:
            return False
        # 更新 prev 为当前元素 cur，继续检查下一个元素
        prev = cur
    # 若全部元素满足单调递增条件，则返回 True
    return True
```