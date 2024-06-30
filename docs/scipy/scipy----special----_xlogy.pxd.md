# `D:\src\scipysrc\scipy\scipy\special\_xlogy.pxd`

```
# 从 C 标准库中导入 log1p 函数
from libc.math cimport log1p

# 从 _complexstuff 模块中导入 zlog, zisnan, number_t 类型
from ._complexstuff cimport zlog, zisnan, number_t

# 从 _cunity 模块中导入 clog1p 函数
from ._cunity cimport clog1p

# 定义一个内联函数 xlogy，计算 x 和 y 的对数乘积
cdef inline number_t xlogy(number_t x, number_t y) noexcept nogil:
    # 如果 x 为 0 并且 y 不是 NaN，则返回 0
    if x == 0 and not zisnan(y):
        return 0
    else:
        # 否则返回 x * zlog(y)，zlog 是从 _complexstuff 模块中导入的函数
        return x * zlog(y)

# 定义一个内联函数 xlog1py，计算 x * log(1 + y) 或者 x * clog1p(y)
cdef inline number_t xlog1py(number_t x, number_t y) noexcept nogil:
    # 如果 x 为 0 并且 y 不是 NaN，则返回 0
    if x == 0 and not zisnan(y):
        return 0
    else:
        # 如果 number_t 是 double 类型，则返回 x * log1p(y)，log1p 是从 libc.math 中导入的函数
        if number_t is double:
            return x * log1p(y)
        # 否则返回 x * clog1p(y)，clog1p 是从 _cunity 模块中导入的函数
        else:
            return x * clog1p(y)
```