# `D:\src\scipysrc\scipy\scipy\io\matlab\_mio_utils.pxd`

```
# 导入需要的 numpy 库，并使用 cimport 关键字进行 C 扩展导入
cimport numpy as np

# 定义一个 cdef 函数，用于将传入的 numpy 数组中的元素进行压缩
cpdef object squeeze_element(np.ndarray arr):
    ...

# 定义一个 cdef 函数，将传入的对象转换为 numpy 数组，并将数组中的字符转换为字符串
cpdef np.ndarray chars_to_strings(object obj):
    ...
```