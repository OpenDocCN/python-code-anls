# `D:\src\scipysrc\scikit-learn\sklearn\utils\_sorting.pxd`

```
# 导入 Cython 模块中的 _typedefs 子模块，引入 intp_t 类型的定义
from ._typedefs cimport intp_t

# 从 Cython 模块中直接导入 floating 类型
from cython cimport floating

# 定义了一个 Cython 编译器特定的函数 simultaneous_sort，接受以下参数：
# - dist: floating 指针，用于存储浮点数数组
# - idx: intp_t 指针，用于存储整数数组
# - size: intp_t 类型，表示数组的大小
# 函数被声明为 noexcept，表示不抛出异常；同时被声明为 nogil，表示不使用全局解释器锁（GIL）
```