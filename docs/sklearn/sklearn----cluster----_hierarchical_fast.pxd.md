# `D:\src\scipysrc\scikit-learn\sklearn\cluster\_hierarchical_fast.pxd`

```
# 导入 cimport 命令从当前包的 utils 模块中导入 intp_t 类型
from ..utils._typedefs cimport intp_t

# 定义一个 Cython 类 UnionFind，实现并查集数据结构
cdef class UnionFind:
    # 声明 Cython 成员变量
    # next_label 表示下一个标签值
    cdef intp_t next_label
    # parent 是一个整数数组，存储每个元素的父节点
    cdef intp_t[:] parent
    # size 是一个整数数组，存储每个集合的大小（元素个数）
    cdef intp_t[:] size

    # 声明一个 Cython 方法 union，用于合并两个集合
    cdef void union(self, intp_t m, intp_t n) noexcept
    # 声明一个 Cython 方法 fast_find，用于快速查找某个元素所在集合的根节点
    cdef intp_t fast_find(self, intp_t n) noexcept
```