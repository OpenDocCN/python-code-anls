# `D:\src\scipysrc\pandas\pandas\_libs\sas.pyx`

```
# cython: language_level=3, initializedcheck=False
# cython: warn.maybe_uninitialized=True, warn.unused=True
# 从 Cython 导入必要的类型和函数声明
from cython cimport Py_ssize_t
from libc.stddef cimport size_t
from libc.stdint cimport (
    int64_t,
    uint8_t,
    uint16_t,
    uint32_t,
    uint64_t,
)
from libc.stdlib cimport (
    calloc,  # 分配内存函数
    free,    # 释放内存函数
)

import numpy as np  # 导入 NumPy 库

import pandas.io.sas.sas_constants as const  # 导入 SAS 文件格式的常量


cdef object np_nan = np.nan  # 定义 Cython 对象 np_nan，初始化为 NumPy 的 NaN 值


cdef struct Buffer:
    # 缓冲区结构体，用于快速和安全地读写 uint8_t 数据
    # 在读取 SAS7BDAT 文件时，我们使用这个结构体来代替 np.array(..., dtype=np.uint8)，
    # 因为使用 NumPy 数组创建速度较慢，而我们需要多次创建 Buffer 实例（大约每读取一行数据创建一次）。
    uint8_t *data   # 指向 uint8_t 数据的指针
    size_t length   # 缓冲区长度


cdef uint8_t buf_get(Buffer buf, size_t offset) except? 255:
    # 从缓冲区 buf 中获取偏移量为 offset 处的数据
    assert offset < buf.length, "Out of bounds read"  # 确保读取操作在有效范围内
    return buf.data[offset]


cdef bint buf_set(Buffer buf, size_t offset, uint8_t value) except 0:
    # 设置缓冲区 buf 中偏移量为 offset 处的数据为 value
    assert offset < buf.length, "Out of bounds write"  # 确保写入操作在有效范围内
    buf.data[offset] = value  # 执行写入操作
    return True


cdef bytes buf_as_bytes(Buffer buf, size_t offset, size_t length):
    # 将缓冲区 buf 中从偏移量 offset 开始长度为 length 的数据转换为 bytes 对象
    assert offset + length <= buf.length, "Out of bounds read"  # 确保读取操作在有效范围内
    return buf.data[offset:offset+length]  # 返回对应的 bytes 对象


cdef Buffer buf_new(size_t length) except *:
    # 创建一个新的缓冲区，长度为 length
    cdef uint8_t *data = <uint8_t *>calloc(length, sizeof(uint8_t))  # 分配内存空间
    if data is NULL:
        raise MemoryError(f"Failed to allocate {length} bytes")  # 如果分配失败，抛出内存错误异常
    return Buffer(data, length)  # 返回创建的 Buffer 实例


cdef buf_free(Buffer buf):
    # 释放缓冲区 buf 所占用的内存
    if buf.data != NULL:
        free(buf.data)  # 调用 free 函数释放内存


# rle_decompress 使用 Run Length Encoding 算法解压缩数据
# 该算法的部分文档可以在以下链接找到：
#
# https://cran.r-project.org/package=sas7bdat/vignettes/sas7bdat.pdf
# 授权协议请参见 LICENSES/SAS7BDAT_LICENSE
cdef int rle_decompress(Buffer inbuff, Buffer outbuff) except? 0:
    # 解压缩函数定义

    cdef:
        uint8_t control_byte, x
        int rpos = 0
        int i, nbytes, end_of_first_byte
        size_t ipos = 0
        Py_ssize_t _  # Python 中的 ssize_t 类型

    return rpos  # 返回解压后数据的位置指针


# rdc_decompress 使用 Ross Data Compression 算法解压缩数据
#
# 参考链接：
# http://collaboration.cmc.ec.gc.ca/science/rpn/biblio/ddj/Website/articles/CUJ/1992/9210/ross/ross.htm
cdef int rdc_decompress(Buffer inbuff, Buffer outbuff) except? 0:
    # 解压缩函数定义

    cdef:
        uint8_t cmd
        uint16_t ctrl_bits = 0, ctrl_mask = 0, ofs, cnt
        int rpos = 0, k, ii
        size_t ipos = 0

    ii = -1  # 初始化 ii 变量为 -1
    # 当输入缓冲区中还有数据时循环执行以下操作
    while ipos < inbuff.length:
        # 增加循环计数器
        ii += 1
        # 控制位掩码右移一位
        ctrl_mask = ctrl_mask >> 1
        # 如果控制位掩码为0，则重新加载控制位
        if ctrl_mask == 0:
            # 读取两个字节作为控制位
            ctrl_bits = ((<uint16_t>buf_get(inbuff, ipos) << 8) +
                         <uint16_t>buf_get(inbuff, ipos + 1))
            ipos += 2
            ctrl_mask = 0x8000

        # 如果控制位为0，则复制一个字节到输出缓冲区
        if ctrl_bits & ctrl_mask == 0:
            buf_set(outbuff, rpos, buf_get(inbuff, ipos))
            ipos += 1
            rpos += 1
            continue

        # 读取命令和计数值
        cmd = (buf_get(inbuff, ipos) >> 4) & 0x0F
        cnt = <uint16_t>(buf_get(inbuff, ipos) & 0x0F)
        ipos += 1

        # 处理短的重复长度编码（Short RLE）
        if cmd == 0:
            # 计数值加3，表示实际的复制次数
            cnt += 3
            # 复制cnt个字节到输出缓冲区
            for k in range(cnt):
                buf_set(outbuff, rpos + k, buf_get(inbuff, ipos))
            rpos += cnt
            ipos += 1

        # 处理长的重复长度编码（Long RLE）
        elif cmd == 1:
            # 计数值通过后续一个字节和一个偏移量进行计算
            cnt += <uint16_t>buf_get(inbuff, ipos) << 4
            cnt += 19
            ipos += 1
            # 复制cnt个字节到输出缓冲区
            for k in range(cnt):
                buf_set(outbuff, rpos + k, buf_get(inbuff, ipos))
            rpos += cnt
            ipos += 1

        # 处理长模式（Long Pattern）
        elif cmd == 2:
            # 计算偏移量和计数值
            ofs = cnt + 3
            ofs += <uint16_t>buf_get(inbuff, ipos) << 4
            ipos += 1
            cnt = <uint16_t>buf_get(inbuff, ipos)
            ipos += 1
            cnt += 16
            # 根据偏移量从输出缓冲区复制cnt个字节到当前位置
            for k in range(cnt):
                buf_set(outbuff, rpos + k, buf_get(outbuff, rpos - <int>ofs + k))
            rpos += cnt

        # 处理短模式（Short Pattern）
        else:
            # 计算偏移量
            ofs = cnt + 3
            ofs += <uint16_t>buf_get(inbuff, ipos) << 4
            ipos += 1
            # 根据偏移量从输出缓冲区复制cmd个字节到当前位置
            for k in range(cmd):
                buf_set(outbuff, rpos + k, buf_get(outbuff, rpos - <int>ofs + k))
            rpos += cmd

    # 返回处理后的输出缓冲区位置
    return rpos
# 定义一个枚举类型 ColumnTypes，包含两个成员变量 column_type_decimal 和 column_type_string
cdef enum ColumnTypes:
    column_type_decimal = 1
    column_type_string = 2


# Const 别名定义
# 确保 const.page_meta_types 的长度为 2
assert len(const.page_meta_types) == 2
cdef:
    # 将 const.page_meta_types 的第一个和第二个元素分别赋值给 page_meta_types_0 和 page_meta_types_1
    int page_meta_types_0 = const.page_meta_types[0]
    int page_meta_types_1 = const.page_meta_types[1]
    # 定义并初始化一些常量
    int page_mix_type = const.page_mix_type
    int page_data_type = const.page_data_type
    int subheader_pointers_offset = const.subheader_pointers_offset

    # subheader_signature_to_index 的复制版本，用于更快速的查找
    # 在 get_subheader_index 中进行查找操作，C 结构在 _init_subheader_signatures() 中初始化
    uint32_t subheader_signatures_32bit[13]
    int subheader_indices_32bit[13]
    uint64_t subheader_signatures_64bit[17]
    int subheader_indices_64bit[17]
    # 初始化数据子头索引为 const.SASIndex.data_subheader_index
    int data_subheader_index = const.SASIndex.data_subheader_index


# 初始化子头签名数组
def _init_subheader_signatures():
    # 获取长度为 4 的子头签名及其索引
    subheaders_32bit = [
        (sig, idx)
        for sig, idx in const.subheader_signature_to_index.items()
        if len(sig) == 4
    ]
    # 获取长度为 8 的子头签名及其索引
    subheaders_64bit = [
        (sig, idx)
        for sig, idx in const.subheader_signature_to_index.items()
        if len(sig) == 8
    ]
    # 确保长度为 13 的 32 位子头签名数组和长度为 17 的 64 位子头签名数组
    assert len(subheaders_32bit) == 13
    assert len(subheaders_64bit) == 17
    # 确保 const.subheader_signature_to_index 的总长度为 13 + 17
    assert len(const.subheader_signature_to_index) == 13 + 17
    # 将子头签名和索引分别复制到对应的数组中
    for i, (signature, idx) in enumerate(subheaders_32bit):
        subheader_signatures_32bit[i] = (<uint32_t *><char *>signature)[0]
        subheader_indices_32bit[i] = idx
    for i, (signature, idx) in enumerate(subheaders_64bit):
        subheader_signatures_64bit[i] = (<uint64_t *><char *>signature)[0]
        subheader_indices_64bit[i] = idx


# 调用 _init_subheader_signatures 函数进行初始化
_init_subheader_signatures()


# 根据子头签名获取索引的快速版本
def get_subheader_index(bytes signature):
    """Fast version of 'subheader_signature_to_index.get(signature)'."""
    cdef:
        uint32_t sig32
        uint64_t sig64
        Py_ssize_t i
    # 确保签名长度为 4 或 8
    assert len(signature) in (4, 8)
    if len(signature) == 4:
        # 将 4 字节签名转换为 uint32_t 类型
        sig32 = (<uint32_t *><char *>signature)[0]
        # 在 subheader_signatures_32bit 数组中查找匹配的签名并返回对应的索引
        for i in range(len(subheader_signatures_32bit)):
            if subheader_signatures_32bit[i] == sig32:
                return subheader_indices_32bit[i]
    else:
        # 将 8 字节签名转换为 uint64_t 类型
        sig64 = (<uint64_t *><char *>signature)[0]
        # 在 subheader_signatures_64bit 数组中查找匹配的签名并返回对应的索引
        for i in range(len(subheader_signatures_64bit)):
            if subheader_signatures_64bit[i] == sig64:
                return subheader_indices_64bit[i]

    # 如果未找到匹配的签名，返回 data_subheader_index
    return data_subheader_index


# 定义一个名为 Parser 的 Cython 类
cdef class Parser:
    # 类主体部分还未提供，这里只是类的声明，后续可能包括数据成员和方法
    cdef:
        int column_count                      # 声明整数变量 column_count，用于存储列数
        int64_t[:] lengths                    # 声明 int64_t 类型的数组 lengths，存储列数据的长度
        int64_t[:] offsets                    # 声明 int64_t 类型的数组 offsets，存储列数据的偏移量
        int64_t[:] column_types               # 声明 int64_t 类型的数组 column_types，存储列的类型信息
        uint8_t[:, :] byte_chunk              # 声明 uint8_t 类型的二维数组 byte_chunk，存储字节数据块
        object[:, :] string_chunk             # 声明 object 类型的二维数组 string_chunk，存储字符串数据块
        uint8_t *cached_page                  # 声明 uint8_t 类型指针 cached_page，缓存页面数据
        int cached_page_len                   # 声明整数变量 cached_page_len，缓存页面数据的长度
        int current_row_on_page_index         # 声明整数变量 current_row_on_page_index，当前页面中的行索引
        int current_page_block_count          # 声明整数变量 current_page_block_count，当前页面的数据块数量
        int current_page_data_subheader_pointers_len  # 声明整数变量 current_page_data_subheader_pointers_len，当前页面数据子头指针的长度
        int current_page_subheaders_count     # 声明整数变量 current_page_subheaders_count，当前页面子头的数量
        int current_row_in_chunk_index        # 声明整数变量 current_row_in_chunk_index，当前块中的行索引
        int current_row_in_file_index         # 声明整数变量 current_row_in_file_index，当前文件中的行索引
        bint blank_missing                   # 声明布尔值变量 blank_missing，指示是否有空白或缺失值
        int header_length                     # 声明整数变量 header_length，存储头部长度信息
        int row_length                        # 声明整数变量 row_length，存储行长度信息
        int bit_offset                        # 声明整数变量 bit_offset，存储位偏移量信息
        int subheader_pointer_length          # 声明整数变量 subheader_pointer_length，存储子头指针长度信息
        int current_page_type                 # 声明整数变量 current_page_type，存储当前页面类型信息
        bint is_little_endian                # 声明布尔值变量 is_little_endian，指示是否小端字节顺序
        int (*decompress)(Buffer, Buffer) except? 0  # 声明指向函数的指针 decompress，用于解压缩数据
        object parser                         # 声明 object 类型变量 parser，用于存储解析器对象

    def __init__(self, object parser):
        cdef:
            int j                             # 声明整数变量 j
            char[:] column_types              # 声明字符数组 column_types

        self.parser = parser                  # 初始化对象变量 parser
        self.blank_missing = parser.blank_missing  # 初始化布尔值变量 blank_missing，表示是否存在空白或缺失值
        self.header_length = self.parser.header_length  # 初始化整数变量 header_length，存储头部长度
        self.column_count = parser.column_count  # 初始化整数变量 column_count，存储列数
        self.lengths = parser.column_data_lengths()  # 初始化数组 lengths，存储列数据的长度信息
        self.offsets = parser.column_data_offsets()  # 初始化数组 offsets，存储列数据的偏移量信息
        self.byte_chunk = parser._byte_chunk    # 初始化二维数组 byte_chunk，存储字节数据块
        self.string_chunk = parser._string_chunk  # 初始化二维数组 string_chunk，存储字符串数据块
        self.row_length = parser.row_length    # 初始化整数变量 row_length，存储行长度信息
        self.bit_offset = self.parser._page_bit_offset  # 初始化整数变量 bit_offset，存储页面位偏移量
        self.subheader_pointer_length = self.parser._subheader_pointer_length  # 初始化整数变量 subheader_pointer_length，存储子头指针长度
        self.is_little_endian = parser.byte_order == "<"  # 初始化布尔值变量 is_little_endian，判断是否为小端字节顺序
        self.column_types = np.empty(self.column_count, dtype="int64")  # 初始化数组 column_types，存储列的类型信息

        # 更新到下一个页面
        self.update_next_page()

        column_types = parser.column_types()    # 获取解析器对象中的列类型信息

        # 映射列类型
        for j in range(self.column_count):
            if column_types[j] == b"d":         # 如果列类型为 'd'，表示十进制数据
                self.column_types[j] = column_type_decimal  # 映射为 column_type_decimal
            elif column_types[j] == b"s":       # 如果列类型为 's'，表示字符串数据
                self.column_types[j] = column_type_string   # 映射为 column_type_string
            else:
                raise ValueError(f"unknown column type: {self.parser.columns[j].ctype}")  # 抛出值错误异常，未知列类型

        # 压缩处理
        if parser.compression == const.rle_compression:   # 如果压缩类型为 RLE 压缩
            self.decompress = rle_decompress  # 设置解压函数为 rle_decompress
        elif parser.compression == const.rdc_compression:  # 如果压缩类型为 RDC 压缩
            self.decompress = rdc_decompress  # 设置解压函数为 rdc_decompress
        else:
            self.decompress = NULL            # 否则设置解压函数为空

        # 更新解析器的当前状态
        self.current_row_in_chunk_index = parser._current_row_in_chunk_index  # 初始化当前块中的行索引
        self.current_row_in_file_index = parser._current_row_in_file_index    # 初始化当前文件中的行索引
        self.current_row_on_page_index = parser._current_row_on_page_index    # 初始化当前页面中的行索引
    # 读取指定行数的数据，使用Cython的定义变量语法声明变量
    def read(self, int nrows):
        # 定义布尔变量done和Py_ssize_t类型的变量_
        cdef:
            bint done
            Py_ssize_t _

        # 循环读取指定行数的数据
        for _ in range(nrows):
            # 调用self.readline()方法，将结果赋给done
            done = self.readline()
            # 如果读取到数据，则跳出循环
            if done:
                break

        # 更新解析器对象的当前行索引信息
        self.parser._current_row_on_page_index = self.current_row_on_page_index
        self.parser._current_row_in_chunk_index = self.current_row_in_chunk_index
        self.parser._current_row_in_file_index = self.current_row_in_file_index

    # 使用Cython定义布尔变量类型，可能会抛出异常
    cdef bint read_next_page(self) except? True:
        # 定义布尔变量done
        cdef bint done

        # 调用self.parser._read_next_page()方法，将结果赋给done
        done = self.parser._read_next_page()
        # 如果完成读取，则将self.cached_page设置为NULL
        if done:
            self.cached_page = NULL
        else:
            # 否则调用self.update_next_page()方法更新下一页数据
            self.update_next_page()
        # 返回done
        return done

    # 更新当前页面的数据
    cdef update_next_page(self):
        # 将解析器对象的缓存页面转换为uint8_t指针并赋给self.cached_page
        self.cached_page = <uint8_t *>self.parser._cached_page
        # 计算缓存页面的长度并赋给self.cached_page_len
        self.cached_page_len = len(self.parser._cached_page)
        # 将当前页面行索引置为0
        self.current_row_on_page_index = 0
        # 将当前页面类型设置为解析器对象的当前页面类型
        self.current_page_type = self.parser._current_page_type
        # 将当前页面块数设置为解析器对象的当前页面块数
        self.current_page_block_count = self.parser._current_page_block_count
        # 计算当前页面数据子标题指针列表的长度并赋给self.current_page_data_subheader_pointers_len
        self.current_page_data_subheader_pointers_len = len(
            self.parser._current_page_data_subheader_pointers
        )
        # 将当前页面子标题数设置为解析器对象的当前页面子标题数
        self.current_page_subheaders_count = self.parser._current_page_subheaders_count
    # 声明并初始化变量，定义异常处理
    cdef bint readline(self) except? True:

        # 声明变量
        cdef:
            int offset, length, bit_offset, align_correction
            int subheader_pointer_length, mn
            bint done, flag

        # 将类成员变量 self.bit_offset 赋给局部变量 bit_offset
        bit_offset = self.bit_offset
        # 将类成员变量 self.subheader_pointer_length 赋给局部变量 subheader_pointer_length
        subheader_pointer_length = self.subheader_pointer_length

        # 如果没有缓存的页面（cached_page == NULL），则跳至头部结束处并读取下一页数据
        if self.cached_page == NULL:
            self.parser._path_or_buf.seek(self.header_length)
            # 调用 read_next_page() 方法读取下一页数据，如果完成则返回 True
            done = self.read_next_page()
            if done:
                return True

        # 循环直到读取到数据行为止
        while True:
            # 如果当前页面类型是 page_meta_types_0 或 page_meta_types_1
            if self.current_page_type in (page_meta_types_0, page_meta_types_1):
                # 判断当前行索引是否超出当前页数据子头指针的长度
                flag = self.current_row_on_page_index >=\
                    self.current_page_data_subheader_pointers_len
                if flag:
                    # 读取下一页数据，如果完成则返回 True
                    done = self.read_next_page()
                    if done:
                        return True
                    continue
                # 从当前页数据子头指针中获取偏移量和长度
                offset, length = self.parser._current_page_data_subheader_pointers[
                    self.current_row_on_page_index
                ]
                # 调用 process_byte_array_with_data() 处理偏移量和长度指定的字节数组数据
                self.process_byte_array_with_data(offset, length)
                return False
            # 如果当前页面类型是 page_mix_type
            elif self.current_page_type == page_mix_type:
                # 计算对齐修正值
                align_correction = (
                    bit_offset
                    + subheader_pointers_offset
                    + self.current_page_subheaders_count * subheader_pointer_length
                )
                align_correction = align_correction % 8
                # 计算偏移量
                offset = bit_offset + align_correction
                offset += subheader_pointers_offset
                offset += self.current_page_subheaders_count * subheader_pointer_length
                offset += self.current_row_on_page_index * self.row_length
                # 调用 process_byte_array_with_data() 处理偏移量和指定长度的字节数组数据
                self.process_byte_array_with_data(offset, self.row_length)
                # 获取行数的最小值，用于判断是否读取完当前页面的所有行
                mn = min(self.parser.row_count, self.parser._mix_page_row_count)
                if self.current_row_on_page_index == mn:
                    # 读取下一页数据，如果完成则返回 True
                    done = self.read_next_page()
                    if done:
                        return True
                return False
            # 如果当前页面类型是 page_data_type
            elif self.current_page_type == page_data_type:
                # 调用 process_byte_array_with_data() 处理指定偏移量和长度的字节数组数据
                self.process_byte_array_with_data(
                    bit_offset
                    + subheader_pointers_offset
                    + self.current_row_on_page_index * self.row_length,
                    self.row_length,
                )
                # 判断是否当前行索引等于当前页面的块数
                flag = self.current_row_on_page_index == self.current_page_block_count
                if flag:
                    # 读取下一页数据，如果完成则返回 True
                    done = self.read_next_page()
                    if done:
                        return True
                return False
            else:
                # 抛出异常，指示未知的页面类型
                raise ValueError(f"unknown page type: {self.current_page_type}")
    # 定义函数process_byte_array_with_data，处理字节数组中的数据，可能会引发任何异常
    cdef void process_byte_array_with_data(self, int offset, int length) except *:

        cdef:
            Py_ssize_t j  # 循环变量，用于迭代处理列
            int s, k, m, jb, js, current_row, rpos  # 各种整型变量，用于索引和计数
            int64_t lngt, start, ct  # 长整型变量，存储长度、起始位置、列类型
            Buffer source, decompressed_source  # 缓冲区对象，存储原始和解压后的数据
            int64_t[:] column_types  # 长整型数组，存储列的数据类型
            int64_t[:] lengths  # 长整型数组，存储数据长度
            int64_t[:] offsets  # 长整型数组，存储数据的偏移量
            uint8_t[:, :] byte_chunk  # 二维无符号整型数组，用于存储字节数据
            object[:, :] string_chunk  # 二维对象数组，用于存储字符串数据
            bint compressed  # 布尔变量，指示数据是否压缩

        # 断言确保偏移量加上长度不超过缓存页面的长度，防止越界访问
        assert offset + length <= self.cached_page_len, "Out of bounds read"
        # 从缓存页面中获取指定偏移量和长度的数据，存储到source缓冲区对象中
        source = Buffer(&self.cached_page[offset], length)

        # 检查数据是否压缩，并且长度是否小于行长度
        compressed = self.decompress != NULL and length < self.row_length
        if compressed:
            # 如果数据压缩，创建新的缓冲区对象存储解压后的数据，并执行解压操作
            decompressed_source = buf_new(self.row_length)
            rpos = self.decompress(source, decompressed_source)
            # 检查解压后的数据长度是否符合预期长度，如果不符则引发错误
            if rpos != self.row_length:
                raise ValueError(
                    f"Expected decompressed line of length {self.row_length} bytes "
                    f"but decompressed {rpos} bytes"
                )
            # 将source指向解压后的数据
            source = decompressed_source

        # 初始化各种计数器和索引变量
        current_row = self.current_row_in_chunk_index
        column_types = self.column_types
        lengths = self.lengths
        offsets = self.offsets
        byte_chunk = self.byte_chunk
        string_chunk = self.string_chunk
        s = 8 * self.current_row_in_chunk_index  # 计算当前行的偏移量
        js = 0
        jb = 0
        # 遍历处理每一列的数据
        for j in range(self.column_count):
            lngt = lengths[j]  # 获取当前列的数据长度
            if lngt == 0:
                break
            start = offsets[j]  # 获取当前列数据在缓冲区中的起始位置
            ct = column_types[j]  # 获取当前列的数据类型
            if ct == column_type_decimal:
                # 如果数据类型是decimal（十进制）
                if self.is_little_endian:
                    m = s + 8 - lngt  # 计算数据在字节块中的起始位置
                else:
                    m = s
                # 将数据从source缓冲区复制到byte_chunk中对应的位置
                for k in range(lngt):
                    byte_chunk[jb, m + k] = buf_get(source, start + k)
                jb += 1  # 更新byte_chunk的行索引
            elif column_types[j] == column_type_string:
                # 如果数据类型是string（字符串）
                # 跳过字符串末尾的空白字符，类似于.rstrip(b"\x00 ")
                while lngt > 0 and buf_get(source, start + lngt - 1) in b"\x00 ":
                    lngt -= 1
                # 如果长度为0且设置了空缺数据的处理，则在string_chunk中记录NaN
                if lngt == 0 and self.blank_missing:
                    string_chunk[js, current_row] = np_nan
                else:
                    # 否则将字符串数据复制到string_chunk中
                    string_chunk[js, current_row] = buf_as_bytes(source, start, lngt)
                js += 1  # 更新string_chunk的行索引

        # 更新当前行在页面、块和文件中的索引
        self.current_row_on_page_index += 1
        self.current_row_in_chunk_index += 1
        self.current_row_in_file_index += 1

        # 如果数据经过压缩，则释放解压缩后的数据占用的内存空间
        if compressed:
            buf_free(decompressed_source)
```