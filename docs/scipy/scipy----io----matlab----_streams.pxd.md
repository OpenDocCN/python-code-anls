# `D:\src\scipysrc\scipy\scipy\io\matlab\_streams.pxd`

```
# -*- python -*- or rather like

# 定义一个 Cython 扩展的类 GenericStream
cdef class GenericStream:
    # 类中的成员变量，用于保存文件对象
    cdef object fobj

    # 定义一个公共的 Cython 函数，用于在流中定位到指定位置
    cpdef int seek(self, long int offset, int whence=*) except -1
    # 定义一个公共的 Cython 函数，返回当前流的位置
    cpdef long int tell(self) except -1
    # 定义一个私有的 Cython 函数，从流中读取数据到指定的缓冲区
    cdef int read_into(self, void *buf, size_t n) except -1
    # 定义一个公共的 Cython 函数，从流中读取指定大小的字符串数据
    cdef object read_string(self, size_t n, void **pp, int copy=*)
    # 定义一个公共的 Cython 函数，检查是否已经读取了所有数据
    cpdef int all_data_read(self) except *

# 定义一个 Cython 函数，用于创建并返回一个 GenericStream 对象
cpdef GenericStream make_stream(object fobj)
```