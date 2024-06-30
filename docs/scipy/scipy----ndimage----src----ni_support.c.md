# `D:\src\scipysrc\scipy\scipy\ndimage\src\ni_support.c`

```
/* 版权所有（C）2003-2005年 Peter J. Verveer
 *
 * 源代码和二进制形式的再发布和使用，无论是否进行修改，
 * 都是允许的，前提是满足以下条件：
 *
 * 1. 源代码的再发布必须保留上述版权声明、此条件列表和以下免责声明。
 *
 * 2. 以二进制形式再发布时，必须在文档和/或其他提供的材料中重现上述版权声明、此条件列表和以下免责声明。
 *
 * 3. 未经特定的事先书面许可，不得使用作者的名称来认可或推广从本软件衍生的产品。
 *
 * 本软件按原样提供，作者不提供任何明示或暗示的保证，
 * 包括但不限于对适销性和特定用途的适用性的暗示保证。
 * 作者在任何情况下均不对任何直接、间接、偶然、特殊、惩罚性或后果性的损害承担责任，
 * 包括但不限于替代商品或服务的获取、使用数据或利润的损失或业务中断，无论是在合同、严格责任或侵权行为（包括疏忽或其他方式）引起的，即使事先已告知此类损害的可能性。
 */

#include "ni_support.h"

/* 初始化单个数组元素的迭代器： */
int NI_InitPointIterator(PyArrayObject *array, NI_Iterator *iterator)
{
    int ii;

    // 获取数组的秩（维数）减1
    iterator->rank_m1 = PyArray_NDIM(array) - 1;
    for(ii = 0; ii < PyArray_NDIM(array); ii++) {
        /* 适应宏使用的维度： */
        // 设置维度，减去1以适应宏的使用
        iterator->dimensions[ii] = PyArray_DIM(array, ii) - 1;
        /* 初始化坐标： */
        // 初始化坐标为0
        iterator->coordinates[ii] = 0;
        /* 初始化步长： */
        // 初始化步长
        iterator->strides[ii] = PyArray_STRIDE(array, ii);
        /* 计算在轴末尾移动回的步长： */
        // 计算在轴末尾移动回的步长
        iterator->backstrides[ii] =
                PyArray_STRIDE(array, ii) * iterator->dimensions[ii];
    }
    return 1;
}

/* 初始化在较低子空间上的迭代： */
int NI_SubspaceIterator(NI_Iterator *iterator, npy_uint32 axes)
{
    int ii, last = 0;

    for(ii = 0; ii <= iterator->rank_m1; ii++) {
        if (axes & (((npy_uint32)1) << ii)) {
            if (last != ii) {
                // 将较低子空间的维度、步长和末尾步长初始化为与主空间相同
                iterator->dimensions[last] = iterator->dimensions[ii];
                iterator->strides[last] = iterator->strides[ii];
                iterator->backstrides[last] = iterator->backstrides[ii];
            }
            ++last;
        }
    }
    iterator->rank_m1 = last - 1;
    return 1;
}

/* 初始化数组线的迭代： */
int NI_LineIterator(NI_Iterator *iterator, int axis)
{
    // 将轴转换为位掩码
    npy_int32 axes = ((npy_uint32)1) << axis;
    // 初始化在数组线上的迭代
    return NI_SubspaceIterator(iterator, ~axes);
}
/******************************************************************/
/* Line buffers */
/******************************************************************/

/* Allocate line buffer data */
# 分配线性缓冲区数据
int NI_AllocateLineBuffer(PyArrayObject* array, int axis, npy_intp size1,
        npy_intp size2, npy_intp *lines, npy_intp max_size, double **buffer)
{
    npy_intp line_size, max_lines;

    /* the number of lines of the array is an upper limit for the
         number of lines in the buffer: */
    // 数组的行数是缓冲区中行数的上限：
    max_lines = PyArray_SIZE(array);
    if (PyArray_NDIM(array) > 0 && PyArray_DIM(array, axis) > 0) {
        max_lines /= PyArray_DIM(array, axis);
    }
    /* calculate the space needed for one line, including space to
         support the boundary conditions: */
    // 计算每行所需的空间，包括支持边界条件的空间：
    line_size = sizeof(double) * (PyArray_DIM(array, axis) + size1 + size2);
    /* if *lines < 1, no number of lines is proposed, so we calculate it
         from the maximum size allowed: */
    // 如果 *lines < 1，则没有提议的行数，因此我们根据允许的最大大小计算它：
    if (*lines < 1) {
        *lines = line_size > 0 ? max_size / line_size : 0;
        if (*lines < 1)
            *lines = 1;
    }
    /* no need to allocate too many lines: */
    // 不需要分配太多行：
    if (*lines > max_lines)
        *lines = max_lines;
    /* allocate data for the buffer: */
    // 为缓冲区分配数据：
    *buffer = malloc(*lines * line_size);
    if (!*buffer) {
        PyErr_NoMemory();
        return 0;
    }
    return 1;
}

/* Some NumPy types are ambiguous */
# 一些 NumPy 类型是模糊的
int NI_CanonicalType(int type_num)
{
    switch (type_num) {
        case NPY_INT:
            return NPY_INT32;

        case NPY_LONG:
#if NPY_SIZEOF_LONG == 4
            return NPY_INT32;
#else
            return NPY_INT64;
#endif

        case NPY_LONGLONG:
            return NPY_INT64;

        case NPY_UINT:
            return NPY_UINT32;

        case NPY_ULONG:
#if NPY_SIZEOF_LONG == 4
            return NPY_UINT32;
#else
            return NPY_UINT64;
#endif

        case NPY_ULONGLONG:
            return NPY_UINT64;

        default:
            return type_num;
    }
}

/* Initialize a line buffer */
# 初始化线性缓冲区
int NI_InitLineBuffer(PyArrayObject *array, int axis, npy_intp size1,
        npy_intp size2, npy_intp buffer_lines, double *buffer_data,
        NI_ExtendMode extend_mode, double extend_value, NI_LineBuffer *buffer)
{
    npy_intp line_length = 0, array_lines = 0, size;
    int array_type;

    size = PyArray_SIZE(array);
    /* check if the buffer is big enough: */
    // 检查缓冲区是否足够大：
    if (size > 0 && buffer_lines < 1) {
        PyErr_SetString(PyExc_RuntimeError, "buffer too small");
        return 0;
    }
    /*
     * Check that the data type is supported, against the types listed in
     * NI_ArrayToLineBuffer
     */
    // 检查数据类型是否受支持，与 NI_ArrayToLineBuffer 中列出的类型进行比较
    array_type = NI_CanonicalType(PyArray_TYPE(array));
    switch (array_type) {
    case NPY_BOOL:
    case NPY_UBYTE:
    case NPY_USHORT:
    case NPY_UINT:
    case NPY_ULONG:
    case NPY_ULONGLONG:
    case NPY_BYTE:
    case NPY_SHORT:
    case NPY_INT:
    case NPY_LONG:
    case NPY_LONGLONG:
    case NPY_FLOAT:
    case NPY_DOUBLE:
        break;
    }
}
    default:
        PyErr_Format(PyExc_RuntimeError, "array type %R not supported",
                     (PyObject *)PyArray_DTYPE(array));
        return 0;
    }

    /* 格式化运行时错误，指示不支持的数组类型 */
    PyErr_Format(PyExc_RuntimeError, "array type %R not supported",
                 (PyObject *)PyArray_DTYPE(array));
    // 返回 0，表示初始化失败
    return 0;
    /* Initialize a line iterator to move over the array: */
    // 初始化一个点迭代器来遍历数组
    if (!NI_InitPointIterator(array, &(buffer->iterator)))
        // 如果初始化迭代器失败，则返回 0
        return 0;
    // 初始化一个线迭代器来遍历数组的指定轴
    if (!NI_LineIterator(&(buffer->iterator), axis))
        // 如果初始化线迭代器失败，则返回 0
        return 0;
    // 计算线的长度，如果数组的维度大于 0，则取指定轴上的维度值，否则默认为 1
    line_length = PyArray_NDIM(array) > 0 ? PyArray_DIM(array, axis) : 1;
    if (line_length > 0) {
        // 如果线的长度大于 0，则计算数组中线的数量
        array_lines = line_length > 0 ? size / line_length : 1;
    }
    /* initialize the buffer structure: */
    // 初始化缓冲区结构体
    buffer->array_data = (void *)PyArray_DATA(array);
    buffer->buffer_data = buffer_data;
    buffer->buffer_lines = buffer_lines;
    buffer->array_type = array_type;
    buffer->array_lines = array_lines;
    buffer->next_line = 0;
    buffer->size1 = size1;
    buffer->size2 = size2;
    buffer->line_length = line_length;
    buffer->line_stride =
                    PyArray_NDIM(array) > 0 ? PyArray_STRIDE(array, axis) : 0;
    buffer->extend_mode = extend_mode;
    buffer->extend_value = extend_value;
    // 返回 1，表示初始化成功
    return 1;


这段代码主要用于处理数组（可能是 NumPy 数组）的初始化和设置缓冲区结构。
/* 
   扩展内存中的一行以实现边界条件：
   这个函数用于扩展内存中的一行数据，以处理边界条件。

   参数说明：
   - buffer: 指向内存中行数据起始位置的指针
   - line_length: 行的长度
   - size_before: 行前面要填充的大小
   - size_after: 行后面要填充的大小
   - extend_mode: 扩展模式，指定如何填充边界
   - extend_value: 扩展值，用于某些填充模式下指定填充的值
*/
int NI_ExtendLine(double *buffer, npy_intp line_length,
                  npy_intp size_before, npy_intp size_after,
                  NI_ExtendMode extend_mode, double extend_value)
{
    // 指向第一个有效数据的指针，即 buffer 指针向后移动 size_before 个位置
    double *first = buffer + size_before;
    // 指向最后一个有效数据的后一个位置，即 first 指针向后移动 line_length 个位置
    double *last = first + line_length;
    // 源指针、目标指针、以及值的变量声明
    double *src, *dst, val;

    // 如果行长度为1，并且扩展模式为镜像模式，则改为最近邻模式
    if ((line_length == 1) && (extend_mode == NI_EXTEND_MIRROR))
    {
        extend_mode = NI_EXTEND_NEAREST;
    }

    // 以下部分代码继续处理行数据的扩展...
    switch (extend_mode) {
        /* 根据扩展模式选择不同的填充方式 */

        /* aaaaaaaa|abcd|dddddddd */
        case NI_EXTEND_NEAREST:
            // 设置源和目标指针为第一个和缓冲区的起始位置
            src = first;
            dst = buffer;
            // 从源中读取值，并将其填充到目标中，重复直到 size_before 为零
            val = *src;
            while (size_before--) {
                *dst++ = val;
            }
            // 设置源指针为最后一个元素的前一个位置，目标指针为最后一个元素的起始位置
            src = last - 1;
            dst = last;
            // 从源中读取值，并将其填充到目标中，重复直到 size_after 为零
            val = *src;
            while (size_after--) {
                *dst++ = val;
            }
            break;

        /* abcdabcd|abcd|abcdabcd */
        case NI_EXTEND_WRAP:
        case NI_EXTEND_GRID_WRAP:
            // 设置源指针为最后一个元素的前一个位置，目标指针为第一个元素的前一个位置
            src = last - 1;
            dst = first - 1;
            // 从源中读取值，并将其填充到目标中，重复直到 size_before 为零
            while (size_before--) {
                *dst-- = *src--;
            }
            // 设置源指针为第一个元素的起始位置，目标指针为最后一个元素的起始位置
            src = first;
            dst = last;
            // 从源中读取值，并将其填充到目标中，重复直到 size_after 为零
            while (size_after--) {
                *dst++ = *src++;
            }
            break;

        /* abcddcba|abcd|dcbaabcd */
        case NI_EXTEND_REFLECT:
            // 设置源和目标指针为第一个和第一个元素的前一个位置
            src = first;
            dst = first - 1;
            // 反射填充，直到 size_before 为零或者源指针超出最后一个元素
            while (size_before && src < last) {
                *dst-- = *src++;
                --size_before;
            }
            // 设置源指针为最后一个元素的前一个位置
            src = last - 1;
            // 填充剩余的 size_before 次数
            while (size_before--) {
                *dst-- = *src--;
            }
            // 设置源指针为最后一个元素的前一个位置，目标指针为最后一个元素的起始位置
            src = last - 1;
            dst = last;
            // 反射填充，直到 size_after 为零或者源指针小于第一个元素
            while (size_after && src >= first) {
                *dst++ = *src--;
                --size_after;
            }
            // 设置源指针为第一个元素的起始位置
            src = first;
            // 填充剩余的 size_after 次数
            while (size_after--) {
                *dst++ = *src++;
            }
            break;

        /* cbabcdcb|abcd|cbabcdcb */
        case NI_EXTEND_MIRROR:
            // 设置源指针为第一个元素的下一个位置，目标指针为第一个元素的前一个位置
            src = first + 1;
            dst = first - 1;
            // 镜像填充，直到 size_before 为零或者源指针超出最后一个元素
            while (size_before && src < last) {
                *dst-- = *src++;
                --size_before;
            }
            // 设置源指针为最后一个元素的前一个位置
            src = last - 2;
            // 填充剩余的 size_before 次数
            while (size_before--) {
                *dst-- = *src--;
            }
            // 设置源指针为最后一个元素的前一个位置，目标指针为最后一个元素的起始位置
            src = last - 2;
            dst = last;
            // 镜像填充，直到 size_after 为零或者源指针小于第一个元素
            while (size_after && src >= first) {
                *dst++ = *src--;
                --size_after;
            }
            // 设置源指针为第一个元素的下一个位置
            src = first + 1;
            // 填充剩余的 size_after 次数
            while (size_after--) {
                *dst++ = *src++;
            }
            break;

        /* kkkkkkkk|abcd]kkkkkkkk */
        case NI_EXTEND_CONSTANT:
        case NI_EXTEND_GRID_CONSTANT:
            // 使用指定的扩展值填充前 size_before 个位置
            val = extend_value;
            dst = buffer;
            while (size_before--) {
                *dst++ = val;
            }
            // 设置目标指针为最后一个元素的起始位置
            dst = last;
            // 使用指定的扩展值填充后 size_after 个位置
            while (size_after--) {
                *dst++ = val;
            }
            break;

        default:
            // 报告运行时错误，指出不支持的扩展模式
            PyErr_Format(PyExc_RuntimeError,
                         "mode %d not supported", extend_mode);
            return 0;
    }
    // 返回成功标志
    return 1;
/* 
   定义一个宏，用于将数组中的数据复制到行缓冲区中：
   _TYPE: 数据类型的宏变量
   _type: 实际的数据类型
   _pi: 源数组的指针
   _po: 目标缓冲区的指针
   _length: 要复制的元素数量
   _stride: 源数组中元素之间的跨度
*/
#define CASE_COPY_DATA_TO_LINE(_TYPE, _type, _pi, _po, _length, _stride) \
case _TYPE:                                                              \
{                                                                        \
    npy_intp _ii;                                                        \
    for (_ii = 0; _ii < _length; ++_ii) {                                \
        _po[_ii] = (double)*(_type *)_pi;                                \
        _pi += _stride;                                                  \
    }                                                                    \
}                                                                        \
break

/*
   将数组中的一行数据复制到缓冲区中：
   buffer: 行缓冲区对象的指针
   number_of_lines: 输出参数，记录复制的行数
   more: 输出参数，指示是否还有更多行要处理
*/
int NI_ArrayToLineBuffer(NI_LineBuffer *buffer,
                         npy_intp *number_of_lines, int *more)
{
    double *pb = buffer->buffer_data;   // 缓冲区的数据指针
    char *pa;                           // 源数组的数据指针
    npy_intp length = buffer->line_length;  // 行的长度

    pb += buffer->size1;    // 移动缓冲区数据指针，跳过已存储的行数据
    *number_of_lines = 0;   // 初始化行数为0
    /* 
       填充缓冲区，直到所有数组中的行都被处理完，或者缓冲区已满：
    */
    while (buffer->next_line < buffer->array_lines &&
                 *number_of_lines < buffer->buffer_lines) {
        // 当缓冲区中未处理完所有数组行并且已复制行数小于缓冲区行数时执行循环

        pa = buffer->array_data;
        /* copy the data from the array to the buffer: */
        // 从数组中复制数据到缓冲区

        switch (buffer->array_type) {
            CASE_COPY_DATA_TO_LINE(NPY_BOOL, npy_bool,
                                   pa, pb, length, buffer->line_stride);
            CASE_COPY_DATA_TO_LINE(NPY_UBYTE, npy_ubyte,
                                   pa, pb, length, buffer->line_stride);
            CASE_COPY_DATA_TO_LINE(NPY_USHORT, npy_ushort,
                                   pa, pb, length, buffer->line_stride);
            CASE_COPY_DATA_TO_LINE(NPY_UINT, npy_uint,
                                   pa, pb, length, buffer->line_stride);
            CASE_COPY_DATA_TO_LINE(NPY_ULONG, npy_ulong,
                                   pa, pb, length, buffer->line_stride);
            CASE_COPY_DATA_TO_LINE(NPY_ULONGLONG, npy_ulonglong,
                                   pa, pb, length, buffer->line_stride);
            CASE_COPY_DATA_TO_LINE(NPY_BYTE, npy_byte,
                                   pa, pb, length, buffer->line_stride);
            CASE_COPY_DATA_TO_LINE(NPY_SHORT, npy_short,
                                   pa, pb, length, buffer->line_stride);
            CASE_COPY_DATA_TO_LINE(NPY_INT, npy_int,
                                   pa, pb, length, buffer->line_stride);
            CASE_COPY_DATA_TO_LINE(NPY_LONG, npy_long,
                                   pa, pb, length, buffer->line_stride);
            CASE_COPY_DATA_TO_LINE(NPY_LONGLONG, npy_longlong,
                                   pa, pb, length, buffer->line_stride);
            CASE_COPY_DATA_TO_LINE(NPY_FLOAT, npy_float,
                                   pa, pb, length, buffer->line_stride);
            CASE_COPY_DATA_TO_LINE(NPY_DOUBLE, npy_double,
                                   pa, pb, length, buffer->line_stride);
        default:
            PyErr_Format(PyExc_RuntimeError, "array type %d not supported",
                         buffer->array_type);
            return 0;
        }
        // 根据数组类型执行相应的数据复制操作，如果类型不支持则返回运行时错误

        /* goto next line in the array: */
        // 转到数组中的下一行

        NI_ITERATOR_NEXT(buffer->iterator, buffer->array_data);

        /* implement boundary conditions to the line: */
        // 对行实施边界条件

        if (buffer->size1 + buffer->size2 > 0) {
            if (!NI_ExtendLine(pb - buffer->size1, length, buffer->size1,
                               buffer->size2, buffer->extend_mode,
                               buffer->extend_value)) {
                return 0;
            }
        }
        // 如果存在边界条件，则对行进行扩展处理，如果扩展失败则返回 0

        /* The number of the array lines copied: */
        // 已复制的数组行数

        ++(buffer->next_line);

        /* keep track of (and return) the number of lines in the buffer: */
        // 记录并返回缓冲区中的行数

        ++(*number_of_lines);

        pb += buffer->line_length + buffer->size1 + buffer->size2;
    }
    // 结束循环，如果未处理完所有数组行，则将 *more 设置为 true

    /* if not all array lines were processed, *more is set true: */
    // 如果未处理完所有数组行，则将 *more 设置为 true

    *more = buffer->next_line < buffer->array_lines;
    return 1;
    // 返回 1 表示处理完成
/* 定义宏：将缓冲区中的一行复制到数组中 */
#define CASE_COPY_LINE_TO_DATA(_TYPE, _type, _pi, _po, _length, _stride) \
case _TYPE:                                                              \
{                                                                        \
    npy_intp _ii;                                                        \
    for (_ii = 0; _ii < _length; ++_ii) {                                \
        *(_type *)_po = (_type)_pi[_ii];                                 \
        _po += _stride;                                                  \
    }                                                                    \
}                                                                        \
break

/* 将缓冲区中的一行复制到数组中 */
int NI_LineBufferToArray(NI_LineBuffer *buffer)
{
    double *pb = buffer->buffer_data;  // 指向缓冲区数据的双精度浮点指针
    char *pa;                          // 字符指针，用于指向数组数据
    npy_intp jj, length = buffer->line_length;  // 数组长度为缓冲区行长度

    pb += buffer->size1;  // 将pb移动到缓冲区数据的第二部分，通常是从缓冲区头部移动
    // 遍历缓冲区中的每一行数据
    for(jj = 0; jj < buffer->buffer_lines; jj++) {
        /* 如果所有数组行都已经复制，则返回 */
        if (buffer->next_line == buffer->array_lines)
            break;
        // 指向数组数据的指针
        pa = buffer->array_data;
        /* 从缓冲区复制数据到数组中 */
        switch (buffer->array_type) {
            // 不同的数组类型使用不同的宏进行复制
            CASE_COPY_LINE_TO_DATA(NPY_BOOL, npy_bool,
                                   pb, pa, length, buffer->line_stride);
            CASE_COPY_LINE_TO_DATA(NPY_UBYTE, npy_ubyte,
                                   pb, pa, length, buffer->line_stride);
            CASE_COPY_LINE_TO_DATA(NPY_USHORT, npy_ushort,
                                   pb, pa, length, buffer->line_stride);
            CASE_COPY_LINE_TO_DATA(NPY_UINT, npy_uint,
                                   pb, pa, length, buffer->line_stride);
            CASE_COPY_LINE_TO_DATA(NPY_ULONG, npy_ulong,
                                   pb, pa, length, buffer->line_stride);
            CASE_COPY_LINE_TO_DATA(NPY_ULONGLONG, npy_ulonglong,
                                   pb, pa, length, buffer->line_stride);
            CASE_COPY_LINE_TO_DATA(NPY_BYTE, npy_byte,
                                   pb, pa, length, buffer->line_stride);
            CASE_COPY_LINE_TO_DATA(NPY_SHORT, npy_short,
                                   pb, pa, length, buffer->line_stride);
            CASE_COPY_LINE_TO_DATA(NPY_INT, npy_int,
                                   pb, pa, length, buffer->line_stride);
            CASE_COPY_LINE_TO_DATA(NPY_LONG, npy_long,
                                   pb, pa, length, buffer->line_stride);
            CASE_COPY_LINE_TO_DATA(NPY_LONGLONG, npy_longlong,
                                   pb, pa, length, buffer->line_stride);
            CASE_COPY_LINE_TO_DATA(NPY_FLOAT, npy_float,
                                   pb, pa, length, buffer->line_stride);
            CASE_COPY_LINE_TO_DATA(NPY_DOUBLE, npy_double,
                                   pb, pa, length, buffer->line_stride);
        default:
            // 如果数组类型不支持，则抛出运行时错误
            PyErr_Format(PyExc_RuntimeError, "array type %d not supported",
                         buffer->array_type);
            return 0;
        }
        /* 移动到数组中的下一行数据 */
        NI_ITERATOR_NEXT(buffer->iterator, buffer->array_data);
        /* 已复制的行数加一 */
        ++(buffer->next_line);
        /* 将缓冲区数据指针移动到下一行 */
        pb += buffer->line_length + buffer->size1 + buffer->size2;
    }
    // 返回成功复制的标志
    return 1;
/******************************************************************/
/* Multi-dimensional filter support functions */
/******************************************************************/

/* Initialize a filter iterator: */
int
NI_InitFilterIterator(int rank, npy_intp *filter_shape,
                    npy_intp filter_size, npy_intp *array_shape,
                    npy_intp *origins, NI_FilterIterator *iterator)
{
    int ii;
    npy_intp fshape[NPY_MAXDIMS], forigins[NPY_MAXDIMS];

    // 初始化滤波器形状和原点数组
    for(ii = 0; ii < rank; ii++) {
        fshape[ii] = *filter_shape++;
        forigins[ii] = origins ? *origins++ : 0;
    }

    /* calculate the strides, used to move the offsets pointer through
         the offsets table: */
    // 计算步长，用于在偏移表中移动指针
    if (rank > 0) {
        iterator->strides[rank - 1] = filter_size;
        for(ii = rank - 2; ii >= 0; ii--) {
            npy_intp step = array_shape[ii + 1] < fshape[ii + 1] ?
                                                                         array_shape[ii + 1] : fshape[ii + 1];
            iterator->strides[ii] =  iterator->strides[ii + 1] * step;
        }
    }

    // 初始化滤波器迭代器的反向步长和边界扩展大小
    for(ii = 0; ii < rank; ii++) {
        npy_intp step = array_shape[ii] < fshape[ii] ?
                                                                                         array_shape[ii] : fshape[ii];
        npy_intp orgn = fshape[ii] / 2 + forigins[ii];
        /* stride for stepping back to previous offsets: */
        iterator->backstrides[ii] = (step - 1) * iterator->strides[ii];
        /* initialize boundary extension sizes: */
        iterator->bound1[ii] = orgn;
        iterator->bound2[ii] = array_shape[ii] - fshape[ii] + orgn;
    }

    return 1;
}

/* Calculate the offsets to the filter points, for all border regions and
     the interior of the array: */
int NI_InitFilterOffsets(PyArrayObject *array, npy_bool *footprint,
         npy_intp *filter_shape, npy_intp* origins,
         NI_ExtendMode mode, npy_intp **offsets, npy_intp *border_flag_value,
         npy_intp **coordinate_offsets)
{
    int rank, ii;
    npy_intp kk, ll, filter_size = 1, offsets_size = 1, max_size = 0;
    npy_intp max_stride = 0, *ashape = NULL, *astrides = NULL;
    npy_intp footprint_size = 0, coordinates[NPY_MAXDIMS], position[NPY_MAXDIMS];
    npy_intp fshape[NPY_MAXDIMS], forigins[NPY_MAXDIMS], *po, *pc = NULL;

    // 获取数组的维度和步长
    rank = PyArray_NDIM(array);
    ashape = PyArray_DIMS(array);
    astrides = PyArray_STRIDES(array);

    // 初始化滤波器形状和原点数组
    for(ii = 0; ii < rank; ii++) {
        fshape[ii] = *filter_shape++;
        forigins[ii] = origins ? *origins++ : 0;
    }

    /* the size of the footprint array: */
    // 计算足迹数组的大小
    for(ii = 0; ii < rank; ii++)
        filter_size *= fshape[ii];

    /* calculate the number of non-zero elements in the footprint: */
    // 计算足迹中非零元素的数量
    if (footprint) {
        for(kk = 0; kk < filter_size; kk++)
            if (footprint[kk])
                ++footprint_size;
    } else {
        footprint_size = filter_size;
    }

    /* calculate how many sets of offsets must be stored: */
    // 计算需要存储的偏移集合数量
    // 对于每一个维度进行循环，计算偏移量的总大小
    for(ii = 0; ii < rank; ii++)
        offsets_size *= (ashape[ii] < fshape[ii] ? ashape[ii] : fshape[ii]);
    /* 分配偏移数据空间： */
    *offsets = malloc(offsets_size * footprint_size * sizeof(npy_intp));
    // 检查内存是否成功分配
    if (!*offsets) {
        PyErr_NoMemory();
        // 如果分配失败，则跳转到退出标签
        goto exit;
    }
    // 如果需要坐标偏移数据：
    if (coordinate_offsets) {
        // 分配坐标偏移数据空间
        *coordinate_offsets = malloc(offsets_size * rank
                                     * footprint_size * sizeof(npy_intp));
        // 检查内存是否成功分配
        if (!*coordinate_offsets) {
            PyErr_NoMemory();
            // 如果分配失败，则跳转到退出标签
            goto exit;
        }
    }
    // 对于每一个维度进行循环：
    for(ii = 0; ii < rank; ii++) {
        npy_intp stride;
        /* 查找最大的轴大小： */
        // 检查当前维度的大小是否大于最大值
        if (ashape[ii] > max_size)
            max_size = ashape[ii];
        /* 查找最大的步长： */
        // 计算当前维度的步长，并与当前的最大步长比较
        stride = astrides[ii] < 0 ? -astrides[ii] : astrides[ii];
        if (stride > max_stride)
            max_stride = stride;
        /* 迭代核心元素的坐标： */
        // 初始化迭代核心元素的坐标为0
        coordinates[ii] = 0;
        /* 跟踪核心位置： */
        // 初始化核心位置为0
        position[ii] = 0;
    }
    /* 表示超出边界的标志位必须具有比任何可能的偏移量都大的值： */
    // 计算超出边界的标志位的值，确保比任何可能的偏移量都大
    *border_flag_value = max_size * max_stride + 1;
    /* 计算所有可能的偏移量，用于访问滤波器内核中的元素，
         适用于数组中的所有区域（内部和边界区域）： */
    // 计算用于访问滤波器内核中元素的所有可能偏移量，包括内部和边界区域
    po = *offsets;
    if (coordinate_offsets) {
        pc = *coordinate_offsets;
    }
    /* 迭代所有区域： */
    // 迭代所有区域
    }

 exit:
    // 如果出现 Python 异常：
    if (PyErr_Occurred()) {
        // 释放偏移量数据空间
        free(*offsets);
        // 如果存在坐标偏移数据空间，也释放它
        if (coordinate_offsets) {
            free(*coordinate_offsets);
        }
        // 返回0表示失败
        return 0;
    } else {
        // 没有异常发生，返回1表示成功
        return 1;
    }
}

# 初始化坐标列表，分配内存并设置初始属性
NI_CoordinateList* NI_InitCoordinateList(int size, int rank)
{
    # 分配内存以存储坐标列表结构
    NI_CoordinateList *list = malloc(sizeof(NI_CoordinateList));
    if (!list) {
        return NULL;
    }
    # 设置坐标列表的块大小和等级
    list->block_size = size;
    list->rank = rank;
    # 初始时没有块分配
    list->blocks = NULL;
    return list;
}

# 将坐标列表中的块从一个列表移动到另一个列表
int NI_CoordinateListStealBlocks(NI_CoordinateList *list1,
                                 NI_CoordinateList *list2)
{
    # 如果两个坐标列表的块大小或等级不匹配，抛出运行时错误
    if (list1->block_size != list2->block_size ||
            list1->rank != list2->rank) {
        PyErr_SetString(PyExc_RuntimeError, "coordinate lists not compatible");
        return 1;
    }
    # 如果第一个坐标列表不为空，抛出运行时错误
    if (list1->blocks) {
        PyErr_SetString(PyExc_RuntimeError, "first is list not empty");
        return 1;
    }
    # 将第二个坐标列表的块指针赋给第一个坐标列表，并将第二个列表的块指针置空
    list1->blocks = list2->blocks;
    list2->blocks = NULL;
    return 0;
}

# 向坐标列表中添加一个新的坐标块
NI_CoordinateBlock* NI_CoordinateListAddBlock(NI_CoordinateList *list)
{
    # 初始化坐标块为 NULL
    NI_CoordinateBlock* block = NULL;
    # 分配内存以存储坐标块结构
    block = malloc(sizeof(NI_CoordinateBlock));
    if (!block) {
        return NULL;
    }
    # 分配内存以存储坐标块中的坐标数据
    block->coordinates = malloc(list->block_size * list->rank
                                * sizeof(npy_intp));
    if (!block->coordinates) {
        free(block);
        return NULL;
    }
    # 将新的坐标块插入到坐标列表的头部
    block->next = list->blocks;
    list->blocks = block;
    # 设置坐标块的大小为 0，表示初始时没有有效的坐标数据
    block->size = 0;

    return block;
}

# 从坐标列表中删除第一个坐标块
NI_CoordinateBlock* NI_CoordinateListDeleteBlock(NI_CoordinateList *list)
{
    # 获取坐标列表中的第一个坐标块
    NI_CoordinateBlock* block = list->blocks;
    if (block) {
        # 将坐标列表的头部指向下一个坐标块
        list->blocks = block->next;
        # 释放当前坐标块中的坐标数据
        free(block->coordinates);
        # 释放当前坐标块结构
        free(block);
    }
    return list->blocks;
}

# 释放坐标列表及其所有的坐标块和数据
void NI_FreeCoordinateList(NI_CoordinateList *list)
{
    if (list) {
        # 遍历并释放坐标列表中的所有坐标块
        NI_CoordinateBlock *block = list->blocks;
        while (block) {
            NI_CoordinateBlock *tmp = block;
            block = block->next;
            # 释放当前坐标块中的坐标数据
            free(tmp->coordinates);
            # 释放当前坐标块结构
            free(tmp);
        }
        # 将坐标列表的块指针置空
        list->blocks = NULL;
        # 释放坐标列表结构
        free(list);
    }
}
```