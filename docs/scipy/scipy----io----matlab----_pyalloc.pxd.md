# `D:\src\scipysrc\scipy\scipy\io\matlab\_pyalloc.pxd`

```
# 导入特定于 Cython 的模块和函数声明
from cpython cimport PyBytes_FromStringAndSize, \
    PyBytes_AS_STRING, PyBytes_Size

# 分配内存并将其包装为 Python 字符串对象的函数
cdef inline object pyalloc_v(Py_ssize_t n, void **pp):
    # 使用 PyBytes_FromStringAndSize 分配 n 字节大小的空间，并返回一个 Python 字符串对象
    cdef object ob = PyBytes_FromStringAndSize(NULL, n)
    # 获取 Python 字符串对象的字符数据指针，并存储在 pp 参数所指向的位置
    pp[0] = <void*> PyBytes_AS_STRING(ob)
    # 返回分配的 Python 字符串对象
    return ob
```