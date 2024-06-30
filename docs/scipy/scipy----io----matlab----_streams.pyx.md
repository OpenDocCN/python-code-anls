# `D:\src\scipysrc\scipy\scipy\io\matlab\_streams.pyx`

```
# -*- python -*- or near enough

# 导入 zlib 模块，用于处理 zlib 压缩流
import zlib

# 从 cpython 中导入 PyBytes_AS_STRING 和 PyBytes_Size 函数
from cpython cimport PyBytes_AS_STRING, PyBytes_Size

# 从 _pyalloc 中导入 pyalloc_v 函数
from ._pyalloc cimport pyalloc_v

# 从 libc.string 中导入 memcpy 函数
from libc.string cimport memcpy

# 从 Python.h 中外部导入相关函数和结构体
cdef extern from "Python.h":
    void *PyCObject_Import(char *, char *) except NULL
    ctypedef struct PyTypeObject:
        pass
    ctypedef struct PyObject:
        pass
    ctypedef struct FILE

# 定义常量 _BLOCK_SIZE 并赋值为 131072
DEF _BLOCK_SIZE = 131072

# 将 _BLOCK_SIZE 赋值给公共常量 BLOCK_SIZE
BLOCK_SIZE = _BLOCK_SIZE  # public

# 定义 GenericStream 类
cdef class GenericStream:

    # 初始化方法，接受一个文件对象 fobj 作为参数
    def __init__(self, fobj):
        self.fobj = fobj

    # 定义 seek 方法，用于移动文件对象指针
    cpdef int seek(self, long int offset, int whence=0) except -1:
        self.fobj.seek(offset, whence)
        return 0

    # 定义 tell 方法，用于返回文件对象当前指针位置
    cpdef long int tell(self) except -1:
        return self.fobj.tell()

    # 定义 read 方法，从文件对象中读取指定字节数的数据
    def read(self, n_bytes):
        return self.fobj.read(n_bytes)

    # 定义 all_data_read 方法，返回整数 1，表示所有数据已读取
    cpdef int all_data_read(self) except *:
        return 1

    # 定义 read_into 方法，将流中的数据读取到预先分配的缓冲区 buf 中
    cdef int read_into(self, void *buf, size_t n) except -1:
        """ Read n bytes from stream into pre-allocated buffer `buf`
        """
        cdef char *p
        cdef size_t read_size, count

        # 使用 _BLOCK_SIZE 大小的块读取数据到 buf 中
        count = 0
        p = <char*>buf
        while count < n:
            read_size = min(n - count, _BLOCK_SIZE)
            data = self.fobj.read(read_size)
            read_size = len(data)
            if read_size == 0:
                break
            # 将读取的数据复制到 buf 中
            memcpy(p, <const char*>data, read_size)
            p += read_size
            count += read_size

        # 检查实际读取的字节数是否与预期相符，若不符则抛出 OSError 异常
        if count != n:
            raise OSError('could not read bytes')
        return 0

    # 定义 read_string 方法，从流中读取指定大小的数据，并将其包装为 Python 对象返回
    cdef object read_string(self, size_t n, void **pp, int copy=True):
        """Make new memory, wrap with object"""
        if copy != True:
            # 如果 copy 参数为 True，则从流中读取 n 字节的数据，并将其包装为 Python 对象返回
            data = self.fobj.read(n)
            if PyBytes_Size(data) != n:
                raise OSError('could not read bytes')
            # 将数据的内存地址存储在 pp 中，并返回数据对象
            pp[0] = <void*>PyBytes_AS_STRING(data)
            return data

        # 如果 copy 参数为 False，则分配新的内存空间，并从流中读取数据到该空间
        cdef object d_copy = pyalloc_v(n, pp)
        self.read_into(pp[0], n)
        return d_copy


# 定义 ZlibInputStream 类，继承自 GenericStream 类
cdef class ZlibInputStream(GenericStream):
    """
    File-like object uncompressing bytes from a zlib compressed stream.

    Parameters
    ----------
    stream : file-like
        Stream to read compressed data from.
    max_length : int
        Maximum number of bytes to read from the stream.

    Notes
    -----
    Some matlab files contain zlib streams without valid Z_STREAM_END
    termination.  To get round this, we use the decompressobj object, that
    allows you to decode an incomplete stream.  See discussion at
    https://bugs.python.org/issue8672

    """

    # 定义私有属性 _max_length，表示最大读取字节数
    cdef ssize_t _max_length

    # 定义私有属性 _decompressor，表示解压缩对象
    cdef object _decompressor

    # 定义私有属性 _buffer，表示缓冲区
    cdef bytes _buffer

    # 定义私有属性 _buffer_size，表示缓冲区大小
    cdef size_t _buffer_size

    # 定义私有属性 _buffer_position，表示缓冲区当前位置
    cdef size_t _buffer_position

    # 定义私有属性 _total_position，表示总体位置
    cdef size_t _total_position

    # 定义私有属性 _read_bytes，表示已读取字节数
    cdef size_t _read_bytes
    def __init__(self, fobj, ssize_t max_length):
        self.fobj = fobj  # 初始化对象的文件对象

        self._max_length = max_length  # 最大读取长度限制
        self._decompressor = zlib.decompressobj()  # 创建解压缩对象
        self._buffer = b''  # 初始化缓冲区为空字节串
        self._buffer_size = 0  # 缓冲区当前大小为0
        self._buffer_position = 0  # 缓冲区当前位置为0
        self._total_position = 0  # 总体读取位置为0
        self._read_bytes = 0  # 已读取字节数为0

    cdef inline void _fill_buffer(self) except *:
        cdef size_t read_size
        cdef bytes block

        if self._buffer_position < self._buffer_size:
            return  # 如果缓冲区中仍有数据未消耗，则直接返回

        read_size = min(_BLOCK_SIZE, self._max_length - self._read_bytes)  # 计算本次应读取的数据块大小

        block = self.fobj.read(read_size)  # 从文件对象中读取指定大小的数据块
        self._read_bytes += len(block)  # 更新已读取字节数

        self._buffer_position = 0  # 重置缓冲区位置为0
        if not block:
            self._buffer = self._decompressor.flush()  # 若读取块为空，则刷新解压缩对象
        else:
            self._buffer = self._decompressor.decompress(block)  # 否则，解压缩当前读取的数据块
        self._buffer_size = len(self._buffer)  # 更新缓冲区大小为解压后数据的实际大小

    cdef int read_into(self, void *buf, size_t n) except -1:
        """Read n bytes from stream into pre-allocated buffer `buf`
        """
        cdef char *dstp
        cdef char *srcp
        cdef size_t count, size

        dstp = <char*>buf  # 将缓冲区指针赋给目标指针
        count = 0  # 初始化已读取计数为0
        while count < n:
            self._fill_buffer()  # 填充缓冲区
            if self._buffer_size == 0:  # 如果缓冲区大小为0，表示没有数据可读
                break

            srcp = <char*>self._buffer  # 源指针指向缓冲区数据起始位置
            srcp += self._buffer_position  # 根据当前缓冲区位置更新源指针

            size = min(n - count, self._buffer_size - self._buffer_position)  # 计算本次应复制的数据块大小
            memcpy(dstp, srcp, size)  # 将数据从源指针复制到目标指针

            count += size  # 更新已复制数据字节数
            dstp += size  # 更新目标指针位置
            self._buffer_position += size  # 更新缓冲区位置

        self._total_position += count  # 更新总体读取位置

        if count != n:  # 如果实际读取字节数与目标字节数不一致
            raise OSError('could not read bytes')  # 抛出读取错误异常

        return 0  # 返回读取成功标志

    cdef object read_string(self, size_t n, void **pp, int copy=True):
        """Make new memory, wrap with object"""
        cdef object d_copy = pyalloc_v(n, pp)  # 分配内存并返回对象
        self.read_into(pp[0], n)  # 调用读取函数将数据读入预分配的缓冲区
        return d_copy  # 返回分配的内存对象

    def read(self, n_bytes):
        cdef void *p
        return self.read_string(n_bytes, &p)  # 调用读取字符串函数

    cpdef int all_data_read(self) except *:
        if self._read_bytes < self._max_length:
            # we might still have checksum bytes to read
            self._fill_buffer()  # 填充缓冲区以读取剩余的校验和字节
        return (self._max_length == self._read_bytes) and \
               (self._buffer_size == self._buffer_position)  # 返回是否所有数据已读取完成的布尔值

    cpdef long int tell(self) except -1:
        if self._total_position == -1:
            raise OSError("Invalid file position.")  # 如果总体读取位置为-1，则抛出文件位置无效异常
        return self._total_position  # 返回总体读取位置
    # 定义一个 CPython 扩展类型的方法 seek，返回一个整数，可能会引发 -1 异常
    cpdef int seek(self, long int offset, int whence=0) except -1:
        # 声明两个 C 语言风格的变量 new_pos 和 size
        cdef ssize_t new_pos, size
        
        # 根据 whence 参数确定新的位置 new_pos
        if whence == 1:
            new_pos = <ssize_t>self._total_position + offset
        elif whence == 0:
            new_pos = offset
        elif whence == 2:
            # 若 whence 为 2，则抛出 OSError 异常，表示 Zlib 流无法从文件末尾进行定位
            raise OSError("Zlib stream cannot seek from file end")
        else:
            # 若 whence 参数无效，则抛出 ValueError 异常
            raise ValueError("Invalid value for whence")
        
        # 如果新位置 new_pos 小于当前总位置 self._total_position，则抛出 OSError 异常，表示 Zlib 流无法向后定位
        if new_pos < self._total_position:
            raise OSError("Zlib stream cannot seek backwards")
        
        # 循环直到当前总位置 self._total_position 达到或超过新位置 new_pos
        while self._total_position < new_pos:
            # 调用内部方法 _fill_buffer 填充缓冲区
            self._fill_buffer()
            # 如果缓冲区大小为 0，则退出循环
            if self._buffer_size == 0:
                break
            
            # 计算本次循环要处理的数据大小 size，确保不超过需求和当前缓冲区的可用数据
            size = min(new_pos - self._total_position,
                       self._buffer_size - self._buffer_position)
            
            # 更新总位置和缓冲区位置
            self._total_position += size
            self._buffer_position += size
        
        # 返回 0 表示 seek 操作成功完成
        return 0
# 定义一个 Cython 函数，从指定的流对象 `st` 中读取 `n` 字节数据到一个 bytearray 中，并返回读取到的数据作为 bytes 对象
def _read_into(GenericStream st, size_t n):
    # 仅用于测试。应该使用 st.read 替代
    cdef char * d_ptr
    # 使用 bytearray 因为 bytes() 是不可变的
    my_str = bytearray(b' ' * n)
    # 将 bytearray 转换为 char*，并从流 `st` 中读取 `n` 字节数据填充到 `d_ptr` 指向的内存区域
    d_ptr = my_str
    st.read_into(d_ptr, n)
    # 将 bytearray 转换为不可变的 bytes 对象并返回
    return bytes(my_str)


# 定义一个 Cython 函数，从指定的流对象 `st` 中读取 `n` 字节数据，并将其复制到一个 bytearray 中，然后返回这个 bytearray 转换后的 bytes 对象
def _read_string(GenericStream st, size_t n):
    # 仅用于测试。应该使用 st.read 替代
    cdef void *d_ptr
    # 调用 `st.read_string` 方法读取 `n` 字节数据，返回读取的对象 `_obj` 和指向数据的 `d_ptr`
    _obj = st.read_string(n, &d_ptr, True)
    # 使用 bytearray 因为 bytes() 是不可变的
    my_str = bytearray(b'A' * n)
    # 将从 `d_ptr` 指向的内存区域中复制的 `n` 字节数据，复制到 `mys_ptr` 指向的 bytearray 内存区域中
    cdef char *mys_ptr = my_str
    memcpy(mys_ptr, d_ptr, n)
    # 将 bytearray 转换为不可变的 bytes 对象并返回
    return bytes(my_str)


# 定义一个 Cython 函数，根据给定的文件对象 `fobj` 创建一个正确类型的流对象并返回
cpdef GenericStream make_stream(object fobj):
    """ Make stream of correct type for file-like `fobj`
    """
    # 如果 `fobj` 是 `GenericStream` 的实例，则直接返回 `fobj`
    if isinstance(fobj, GenericStream):
        return fobj
    # 否则，使用 `GenericStream` 类型创建一个新的流对象，并返回
    return GenericStream(fobj)
```