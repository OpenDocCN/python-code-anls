# `D:\src\scipysrc\scikit-learn\sklearn\utils\_heap.pxd`

```
# 堆操作的例程，用于各种 Cython 实现。

# 导入 floating 类型的 cython 模块
from cython cimport floating

# 从 _typedefs 模块中导入 intp_t 类型
from ._typedefs cimport intp_t

# 定义一个堆推入操作函数，接受以下参数：
# - values: 浮点数指针，用于存储堆中的值
# - indices: intp_t 类型指针，用于存储堆中值的索引
# - size: intp_t 类型，表示堆的大小
# - val: floating 类型，要推入堆的值
# - val_idx: intp_t 类型，val 对应的索引
# 使用 noexcept 修饰，表示此函数不会引发异常
# 使用 nogil 修饰，表示此函数在全局解锁状态下执行，没有 GIL(GIL 是 Python 中的全局解释器锁) 的限制
cdef int heap_push(
    floating* values,
    intp_t* indices,
    intp_t size,
    floating val,
    intp_t val_idx,
) noexcept nogil
```