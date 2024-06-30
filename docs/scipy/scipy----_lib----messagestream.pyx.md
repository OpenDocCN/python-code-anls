# `D:\src\scipysrc\scipy\scipy\_lib\messagestream.pyx`

```
# 使用 Cython 导入声明，用于优化 Python 与 C 语言之间的接口
cimport cython
# 从 C 标准库中导入需要的函数和类型声明
from libc cimport stdio, stdlib
# 从 CPython 的 C 接口中导入特定函数以及对象的声明
from cpython cimport PyBytes_FromStringAndSize

# 导入标准的 Python 模块
import os
import tempfile

# 从外部头文件 "messagestream.h" 中导入 C 函数声明
cdef extern from "messagestream.h":
    # 声明一个用于创建内存流的 C 函数
    stdio.FILE *messagestream_open_memstream(char **, size_t *)

# 声明一个 Cython 的最终类，用于捕获发送到 FILE* 流的消息
@cython.final
cdef class MessageStream:
    """
    Capture messages emitted to FILE* streams. Do this by directing them
    to a temporary file, residing in memory (if possible) or on disk.
    """

    # 初始化方法，尝试首先使用内存中的文件流，如果不可用则使用临时文件
    def __cinit__(self):
        # 初始化内存流指针和大小
        self._memstream_ptr = NULL
        self.handle = messagestream_open_memstream(&self._memstream_ptr,
                                                   &self._memstream_size)
        # 如果成功创建内存流
        if self.handle != NULL:
            self._removed = 1  # 标记临时文件已移除
            return

        # 如果内存流不可用，则回退到创建临时文件
        fd, self._filename = tempfile.mkstemp(prefix=b'scipy-')

        # 尝试使用 POSIX 风格的删除文件标志
        try:
            os.remove(self._filename)
            self._removed = 1  # 标记临时文件已移除
        except PermissionError:
            self._removed = 0

        # 打开临时文件并准备写入
        self.handle = stdio.fdopen(fd, 'wb+')
        if self.handle == NULL:
            os.close(fd)
            if not self._removed:
                os.remove(self._filename)
            raise OSError(f"Failed to open file {self._filename}")

    # 析构方法，在实例销毁时自动调用
    def __dealloc__(self):
        self.close()  # 关闭当前文件流

    # 获取文件流中的内容并返回为字符串对象
    def get(self):
        cdef long pos
        cdef size_t nread
        cdef char *buf = NULL
        cdef bytes obj

        # 获取当前文件流的位置
        pos = stdio.ftell(self.handle)
        # 如果位置小于等于 0，则返回空字符串
        if pos <= 0:
            return ""

        # 如果使用内存流
        if self._memstream_ptr != NULL:
            stdio.fflush(self.handle)
            # 创建字节对象并返回
            obj = PyBytes_FromStringAndSize(self._memstream_ptr, pos)
        else:
            # 否则分配内存并从文件中读取数据
            buf = <char*>stdlib.malloc(pos)
            if buf == NULL:
                raise MemoryError()

            try:
                stdio.rewind(self.handle)
                nread = stdio.fread(buf, 1, pos, self.handle)
                if nread != <size_t>pos:
                    raise OSError("failed to read messages from buffer")

                # 创建字节对象并返回
                obj = PyBytes_FromStringAndSize(buf, nread)
            finally:
                stdlib.free(buf)  # 释放分配的内存

        # 将字节对象解码为字符串并返回
        return obj.decode('latin1')

    # 清空文件流中的内容
    def clear(self):
        stdio.rewind(self.handle)

    # 关闭文件流并进行清理操作
    cpdef close(self):
        if self.handle != NULL:
            stdio.fclose(self.handle)
            self.handle = NULL

        # 如果使用了内存流，释放相关的内存
        if self._memstream_ptr != NULL:
            stdlib.free(self._memstream_ptr)
            self._memstream_ptr = NULL

        # 如果临时文件未被移除，则在关闭时删除
        if not self._removed:
            os.remove(self._filename)
            self._removed = 1
```