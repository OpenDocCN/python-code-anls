# `D:\src\scipysrc\scipy\scipy\ndimage\src\ni_filters.c`

```
/*
 * 作者：Peter J. Verveer，版权声明
 *
 * 可以在源代码和二进制形式下重新分发和使用，包括但不限于对其进行修改，
 * 前提是满足以下条件：
 * 
 * 1. 源代码的再分发必须保留上述版权声明、条件列表和以下免责声明。
 * 
 * 2. 以二进制形式再分发时，必须在文档和/或其他提供的材料中复制上述版权声明、
 *    条件列表和以下免责声明。
 * 
 * 3. 未经特定的事先书面许可，不得使用作者的名称来认可或推广从本软件派生的产品。
 * 
 * 本软件按原样提供，不提供任何明示或暗示的担保，包括但不限于对适销性和特定
 * 目的的适用性的暗示担保。在任何情况下，无论是合同诉讼、严格责任还是侵权行为，
 * 作者都不对任何直接、间接、偶然、特殊、示范性或后果性损害负责，即使事先已告知
 * 可能发生此类损害的可能性。
 */

#include "ni_support.h"    // 导入相应的头文件
#include "ni_filters.h"    // 导入相应的头文件
#include <math.h>          // 导入数学库函数

#define BUFFER_SIZE 256000  // 定义缓冲区大小常量

int NI_Correlate1D(PyArrayObject *input, PyArrayObject *weights,
                   int axis, PyArrayObject *output, NI_ExtendMode mode,
                   double cval, npy_intp origin)
{
    int symmetric = 0, more;  // 声明并初始化对称性标志和more变量
    npy_intp ii, jj, ll, lines, length, size1, size2, filter_size;  // 声明整型变量和数组索引变量
    double *ibuffer = NULL, *obuffer = NULL;  // 声明输入和输出缓冲区指针
    npy_double *fw;  // 声明权重数组指针
    NI_LineBuffer iline_buffer, oline_buffer;  // 声明输入和输出线性缓冲区对象
    NPY_BEGIN_THREADS_DEF;  // 定义多线程相关宏

    /* 检测权重数组是否对称或反对称：*/
    filter_size = PyArray_SIZE(weights);  // 获取权重数组的大小
    size1 = filter_size / 2;  // 计算前半部分大小
    size2 = filter_size - size1 - 1;  // 计算后半部分大小
    fw = (void *)PyArray_DATA(weights);  // 获取权重数组的数据指针
    if (filter_size & 0x1) {  // 如果权重数组的大小为奇数
        symmetric = 1;  // 假设数组对称
        for(ii = 1; ii <= filter_size / 2; ii++) {  // 遍历前半部分
            if (fabs(fw[ii + size1] - fw[size1 - ii]) > DBL_EPSILON) {  // 检测对称性
                symmetric = 0;  // 不对称
                break;
            }
        }
        if (symmetric == 0) {  // 如果不对称
            symmetric = -1;  // 假设数组反对称
            for(ii = 1; ii <= filter_size / 2; ii++) {  // 遍历前半部分
                if (fabs(fw[size1 + ii] + fw[size1 - ii]) > DBL_EPSILON) {  // 检测反对称性
                    symmetric = 0;  // 不反对称
                    break;
                }
            }
        }
    }
    /* 分配并初始化线性缓冲区：*/
    lines = -1;  // 初始化行数为-1
    if (!NI_AllocateLineBuffer(input, axis, size1 + origin, size2 - origin,
                                                         &lines, BUFFER_SIZE, &ibuffer))  // 分配线性缓冲区
        goto exit;  // 如果分配失败，跳转到退出标签
    // 分配输出行缓冲区
    if (!NI_AllocateLineBuffer(output, axis, 0, 0, &lines, BUFFER_SIZE, &obuffer))
        goto exit;
    
    // 初始化输入行缓冲区
    if (!NI_InitLineBuffer(input, axis, size1 + origin, size2 - origin, lines, ibuffer, mode, cval, &iline_buffer))
        goto exit;
    
    // 初始化输出行缓冲区
    if (!NI_InitLineBuffer(output, axis, 0, 0, lines, obuffer, mode, 0.0, &oline_buffer))
        goto exit;

    // 启动多线程处理
    NPY_BEGIN_THREADS;
    
    // 确定输入数组的长度
    length = PyArray_NDIM(input) > 0 ? PyArray_DIM(input, axis) : 1;
    
    // 调整权重数组的起始点
    fw += size1;
    
    /* 遍历所有数组行： */
    do {
        /* 从数组复制行到缓冲区： */
        if (!NI_ArrayToLineBuffer(&iline_buffer, &lines, &more)) {
            goto exit;
        }
        
        /* 遍历缓冲区中的行： */
        for(ii = 0; ii < lines; ii++) {
            /* 获取输入和输出行的指针： */
            double *iline = NI_GET_LINE(iline_buffer, ii) + size1;
            double *oline = NI_GET_LINE(oline_buffer, ii);
            
            /* 计算相关性： */
            if (symmetric > 0) {
                for(ll = 0; ll < length; ll++) {
                    oline[ll] = iline[0] * fw[0];
                    for(jj = -size1 ; jj < 0; jj++)
                        oline[ll] += (iline[jj] + iline[-jj]) * fw[jj];
                    ++iline;
                }
            } else if (symmetric < 0) {
                for(ll = 0; ll < length; ll++) {
                    oline[ll] = iline[0] * fw[0];
                    for(jj = -size1 ; jj < 0; jj++)
                        oline[ll] += (iline[jj] - iline[-jj]) * fw[jj];
                    ++iline;
                }
            } else {
                for(ll = 0; ll < length; ll++) {
                    oline[ll] = iline[size2] * fw[size2];
                    for(jj = -size1; jj < size2; jj++)
                        oline[ll] += iline[jj] * fw[jj];
                    ++iline;
                }
            }
        }
        
        /* 从缓冲区复制行到数组： */
        if (!NI_LineBufferToArray(&oline_buffer)) {
            goto exit;
        }
    } while(more);
/* 退出函数，执行一些清理工作并返回状态码 */
exit:
    /* 结束线程保护 */
    NPY_END_THREADS;
    /* 释放输入和输出缓冲区 */
    free(ibuffer);
    free(obuffer);
    /* 检查是否发生了 Python 异常，根据异常状态返回 0 或 1 */
    return PyErr_Occurred() ? 0 : 1;
}

/* 定义宏：处理相关性运算的特定情况 */
#define CASE_CORRELATE_POINT(_TYPE, _type, _pi, _weights, _offsets,        \
                             _filter_size, _cvalue, _res, _mv)             \
case _TYPE:                                                                \
{                                                                          \
    npy_intp _ii, _offset;                                                 \
    /* 遍历所有滤波器大小 */
    for (_ii = 0; _ii < _filter_size; ++_ii) {                             \
        _offset = _offsets[_ii];                                           \
        /* 如果偏移量等于 _mv，则使用给定的 _cvalue */
        if (_offset == _mv) {                                              \
            _res += _weights[_ii] * _cvalue;                               \
        }                                                                  \
        /* 否则使用指针偏移后的数据进行相关性计算 */
        else {                                                             \
            _res += _weights[_ii] * (double)(*((_type *)(_pi + _offset))); \
        }                                                                  \
    }                                                                      \
}                                                                          \
break

/* 定义宏：处理滤波后的数据输出 */
#define CASE_FILTER_OUT(_TYPE, _type, _po, _tmp) \
case _TYPE:                                      \
    /* 将 _tmp 转换为 _type 类型并写入 _po 指向的位置 */ \
    *(_type *)_po = (_type)_tmp;                 \
    break

/* 定义宏：安全处理浮点数到无符号数的转换 */
#define CASE_FILTER_OUT_SAFE(_TYPE, _type, _po, _tmp)                  \
case _TYPE:                                                            \
    /* 如果 _tmp 大于 -1，则直接转换为 _type 类型，否则取其绝对值后再转换 */ \
    *(_type *)_po = (_tmp) > -1. ? (_type)(_tmp) : -(_type)(-_tmp);    \
    break

/* 定义函数：执行相关性运算 */
int NI_Correlate(PyArrayObject* input, PyArrayObject* weights,
                 PyArrayObject* output, NI_ExtendMode mode,
                 double cvalue, npy_intp *origins)
{
    npy_bool *pf = NULL;
    npy_intp fsize, jj, kk, filter_size = 0, border_flag_value;
    npy_intp *offsets = NULL, *oo, size;
    NI_FilterIterator fi;
    NI_Iterator ii, io;
    char *pi, *po;
    npy_double *pw;
    npy_double *ww = NULL;
    int err = 0;
    NPY_BEGIN_THREADS_DEF;

    /* 获取滤波器的大小 */
    fsize = PyArray_SIZE(weights);
    /* 获取权重数据的指针 */
    pw = (npy_double*)PyArray_DATA(weights);
    /* 分配用于存储布尔标志的内存 */
    pf = malloc(fsize * sizeof(npy_bool));
    if (!pf) {
        /* 分配内存失败，设置内存错误异常并跳转到退出标签 */
        PyErr_NoMemory();
        goto exit;
    }
    /* 遍历权重数据，检查每个权重是否大于 DBL_EPSILON */
    for(jj = 0; jj < fsize; jj++) {
        if (fabs(pw[jj]) > DBL_EPSILON) {
            pf[jj] = 1;
            ++filter_size;
        } else {
            pf[jj] = 0;
        }
    }
    /* 分配连续内存并复制有效权重 */
    ww = malloc(filter_size * sizeof(npy_double));
    if (!ww) {
        /* 分配内存失败，设置内存错误异常并跳转到退出标签 */
        PyErr_NoMemory();
        goto exit;
    }
    jj = 0;
    for(kk = 0; kk < fsize; kk++) {
        if (pf[kk]) {
            ww[jj++] = pw[kk];
        }
    }
    /* 初始化滤波器偏移量 */
    # 如果初始化滤波器偏移量失败，则跳转到退出标签
    if (!NI_InitFilterOffsets(input, pf, PyArray_DIMS(weights), origins,
                              mode, &offsets, &border_flag_value, NULL)) {
        goto exit;
    }
    /* 初始化滤波器迭代器： */
    # 如果初始化滤波器迭代器失败，则跳转到退出标签
    if (!NI_InitFilterIterator(PyArray_NDIM(input), PyArray_DIMS(weights),
                               filter_size, PyArray_DIMS(input), origins,
                               &fi)) {
        goto exit;
    }
    /* 初始化输入元素迭代器： */
    # 如果初始化输入元素迭代器失败，则跳转到退出标签
    if (!NI_InitPointIterator(input, &ii))
        goto exit;
    /* 初始化输出元素迭代器： */
    # 如果初始化输出元素迭代器失败，则跳转到退出标签
    if (!NI_InitPointIterator(output, &io))
        goto exit;

    NPY_BEGIN_THREADS;
    /* 获取数据指针和数组大小： */
    # 获取输入数组的数据指针并转换为 void 指针类型
    pi = (void *)PyArray_DATA(input);
    # 获取输出数组的数据指针并转换为 void 指针类型
    po = (void *)PyArray_DATA(output);
    # 获取输入数组的元素总数
    size = PyArray_SIZE(input);
    /* 迭代处理每个元素： */
    # 初始化 oo 为偏移量数组的起始位置
    oo = offsets;
exit:
    NPY_END_THREADS;
    // 结束线程并退出当前函数

    if (err == 1) {
        // 如果 err 等于 1，设置运行时错误信息为 "array type not supported"
        PyErr_SetString(PyExc_RuntimeError, "array type not supported");
    }

    // 释放动态分配的内存
    free(offsets);
    free(ww);
    free(pf);

    // 返回是否发生异常：如果异常发生，返回 0；否则返回 1
    return PyErr_Occurred() ? 0 : 1;
}

int
NI_UniformFilter1D(PyArrayObject *input, npy_intp filter_size,
                   int axis, PyArrayObject *output, NI_ExtendMode mode,
                   double cval, npy_intp origin)
{
    npy_intp lines, kk, ll, length, size1, size2;
    int more;
    double *ibuffer = NULL, *obuffer = NULL;
    NI_LineBuffer iline_buffer, oline_buffer;
    NPY_BEGIN_THREADS_DEF;

    size1 = filter_size / 2;
    size2 = filter_size - size1 - 1;

    /* allocate and initialize the line buffers: */
    // 分配并初始化线性缓冲区：
    lines = -1;
    if (!NI_AllocateLineBuffer(input, axis, size1 + origin, size2 - origin,
                                                         &lines, BUFFER_SIZE, &ibuffer))
        goto exit;
    if (!NI_AllocateLineBuffer(output, axis, 0, 0, &lines, BUFFER_SIZE,
                                                         &obuffer))
        goto exit;
    if (!NI_InitLineBuffer(input, axis, size1 + origin, size2 - origin,
                                                            lines, ibuffer, mode, cval, &iline_buffer))
        goto exit;
    if (!NI_InitLineBuffer(output, axis, 0, 0, lines, obuffer, mode, 0.0,
                                                 &oline_buffer))
        goto exit;

    NPY_BEGIN_THREADS;

    length = PyArray_NDIM(input) > 0 ? PyArray_DIM(input, axis) : 1;

    /* iterate over all the array lines: */
    // 遍历所有数组行：
    do {
        /* copy lines from array to buffer: */
        // 将数组行复制到缓冲区：
        if (!NI_ArrayToLineBuffer(&iline_buffer, &lines, &more)) {
            goto exit;
        }

        /* iterate over the lines in the buffers: */
        // 遍历缓冲区中的行：
        for(kk = 0; kk < lines; kk++) {
            /* get lines: */
            // 获取行：
            double *iline = NI_GET_LINE(iline_buffer, kk);
            double *oline = NI_GET_LINE(oline_buffer, kk);

            /* do the uniform filter: */
            // 执行均匀滤波：
            double tmp = 0.0;
            double *l1 = iline;
            double *l2 = iline + filter_size;
            for (ll = 0; ll < filter_size; ++ll) {
                tmp += iline[ll];
            }
            oline[0] = tmp / filter_size;
            for (ll = 1; ll < length; ++ll) {
                tmp += *l2++ - *l1++;
                oline[ll] = tmp / filter_size;
            }
        }

        /* copy lines from buffer to array: */
        // 将缓冲区中的行复制回数组：
        if (!NI_LineBufferToArray(&oline_buffer)) {
            goto exit;
        }

    } while(more);

 exit:
    NPY_END_THREADS;

    // 释放动态分配的内存
    free(ibuffer);
    free(obuffer);

    // 返回是否发生异常：如果异常发生，返回 0；否则返回 1
    return PyErr_Occurred() ? 0 : 1;
}

#define INCREASE_RING_PTR(ptr) \
    (ptr)++;                   \
    if ((ptr) >= end) {        \
        (ptr) = ring;          \
    }

#define DECREASE_RING_PTR(ptr) \
    if ((ptr) == ring) {       \
        (ptr) = end;           \
    }                          \
    (ptr)--;

int
    NI_MinOrMaxFilter1D(PyArrayObject *input, npy_intp filter_size,
                        int axis, PyArrayObject *output, NI_ExtendMode mode,
                        double cval, npy_intp origin, int minimum)
{
    npy_intp lines, kk, ll, length, size1, size2;
    int more;
    double *ibuffer = NULL, *obuffer = NULL;
    NI_LineBuffer iline_buffer, oline_buffer;

    struct pairs {
        double value;
        npy_intp death;
    } *ring = NULL, *minpair, *end, *last;

    NPY_BEGIN_THREADS_DEF;

    // 计算过滤器的两侧尺寸
    size1 = filter_size / 2;
    size2 = filter_size - size1 - 1;
    /* allocate and initialize the line buffers: */
    // 分配和初始化输入和输出的线缓冲区
    lines = -1;
    if (!NI_AllocateLineBuffer(input, axis, size1 + origin, size2 - origin,
                                            &lines, BUFFER_SIZE, &ibuffer))
        goto exit;
    if (!NI_AllocateLineBuffer(output, axis, 0, 0, &lines, BUFFER_SIZE,
                                                                &obuffer))
        goto exit;
    if (!NI_InitLineBuffer(input, axis, size1 + origin, size2 - origin,
                                lines, ibuffer, mode, cval, &iline_buffer))
        goto exit;
    if (!NI_InitLineBuffer(output, axis, 0, 0, lines, obuffer, mode, 0.0,
                                                            &oline_buffer))
        goto exit;

    NPY_BEGIN_THREADS;
    length = PyArray_NDIM(input) > 0 ? PyArray_DIM(input, axis) : 1;

    /* ring is a dequeue of pairs implemented as a circular array */
    // ring 是作为循环数组实现的一对队列
    ring = malloc(filter_size * sizeof(struct pairs));
    if (!ring) {
        goto exit;
    }
    end = ring + filter_size;

    /* iterate over all the array lines: */
    // 迭代处理所有数组行：
    do {
        /* 从数组复制行到缓冲区： */
        /* 如果无法将数组行复制到缓冲区，则跳转到退出标签 */
        if (!NI_ArrayToLineBuffer(&iline_buffer, &lines, &more)) {
            goto exit;
        }
        
        /* 遍历缓冲区中的行： */
        for(kk = 0; kk < lines; kk++) {
            /* 获取输入和输出行的指针： */
            double *iline = NI_GET_LINE(iline_buffer, kk);
            double *oline = NI_GET_LINE(oline_buffer, kk);

            /* 如果滤波器大小为1，则直接复制输入到输出 */
            if (filter_size == 1) {
                memcpy(oline, iline, sizeof(double) * length);
            }
            else {
                /*
                 * Richard Harter 原始代码，改编自：
                 * http://www.richardhartersworld.com/cri/2001/slidingmin.html
                 */
                /* 初始化环形缓冲区的起始位置 */
                minpair = ring;
                minpair->value = *iline++;
                minpair->death = filter_size;
                last = ring;

                /* 遍历输入行并进行滑动窗口最小/最大值滤波 */
                for (ll = 1; ll < filter_size + length - 1; ll++) {
                    double val = *iline++;
                    
                    /* 如果当前最小/最大值过期，则调整环形缓冲区指针 */
                    if (minpair->death == ll) {
                        INCREASE_RING_PTR(minpair)
                    }
                    
                    /* 根据最小化或最大化标志更新环形缓冲区的最小/最大值 */
                    if ((minimum && val <= minpair->value) ||
                        (!minimum && val >= minpair->value)) {
                        minpair->value = val;
                        minpair->death = ll + filter_size;
                        last = minpair;
                    }
                    else {
                        while ((minimum && last->value >= val) ||
                               (!minimum && last->value <= val)) {
                            DECREASE_RING_PTR(last)
                        }
                        INCREASE_RING_PTR(last)
                        last->value = val;
                        last->death = ll + filter_size;
                    }
                    
                    /* 当滤波窗口满足时，将当前最小/最大值存入输出行 */
                    if (ll >= filter_size - 1) {
                        *oline++ = minpair->value;
                    }
                }
            }
        }
        
        /* 将缓冲区中的行复制回数组： */
        /* 如果无法将缓冲区行复制到数组，则跳转到退出标签 */
        if (!NI_LineBufferToArray(&oline_buffer)) {
            goto exit;
        }
    } while(more);

 exit:
    /* 结束 NumPy 线程 */
    NPY_END_THREADS;
    /* 释放申请的内存：输入缓冲区、输出缓冲区、环形缓冲区 */
    free(ibuffer);
    free(obuffer);
    free(ring);
    /* 如果出现异常，则返回0；否则返回1 */
    return PyErr_Occurred() ? 0 : 1;
#undef DECREASE_RING_PTR
#undef INCREASE_RING_PTR

#define CASE_MIN_OR_MAX_POINT(_TYPE, _type, _pi, _offsets, _filter_size, \
                              _cval, _minimum, _res, _mv, _ss)           \
case _TYPE:                                                              \
{                                                                        \
    npy_intp _ii;                                                        \
    npy_intp _oo = _offsets[0];                                          \
    _type _tmp;                                                          \
    _type _cv = (_type)_cval;                                            \
    // 初始化结果为第一个偏移量处的值或给定常数值
    _res = _oo == _mv ? _cv : *(_type *)(_pi + _oo);                     \
    // 如果有结构元素的偏移量数组，则添加结构元素的偏移量
    if (_ss != NULL) {                                                   \
        _res += _ss[0];                                                  \
    }                                                                    \
    // 遍历剩余的偏移量并更新结果值
    for (_ii = 1; _ii < _filter_size; ++_ii) {                           \
        _oo = _offsets[_ii];                                             \
        // 获取当前偏移量处的值或给定常数值
        _tmp = _oo == _mv ? _cv : *(_type *)(_pi + _oo);                 \
        // 如果有结构元素的偏移量数组，则添加结构元素的偏移量
        if (_ss != NULL) {                                               \
            _tmp += (_type) _ss[_ii];                                    \
        }                                                                \
        // 根据需要更新结果值为最小值或最大值
        if (_minimum) {                                                  \
            if (_tmp < _res) {                                           \
                _res = _tmp;                                             \
            }                                                            \
        }                                                                \
        else {                                                           \
            if (_tmp > _res) {                                           \
                _res = _tmp;                                             \
            }                                                            \
        }                                                                \
    }                                                                    \
}                                                                        \
break

int NI_MinOrMaxFilter(PyArrayObject* input, PyArrayObject* footprint,
                      PyArrayObject* structure, PyArrayObject* output,
                      NI_ExtendMode mode, double cvalue, npy_intp *origins,
                      int minimum)
{
    npy_bool *pf = NULL;
    npy_intp fsize, jj, kk, filter_size = 0, border_flag_value;
    npy_intp *offsets = NULL, *oo, size;
    NI_FilterIterator fi;
    NI_Iterator ii, io;
    char *pi, *po;
    int err = 0;
    double *ss = NULL;
    npy_double *ps;
    NPY_BEGIN_THREADS_DEF;

    /* get the footprint: */
    // 获取结构元素的大小
    fsize = PyArray_SIZE(footprint);
    // 获取结构元素的数据指针
    pf = (npy_bool*)PyArray_DATA(footprint);


这段代码主要是一个宏定义和一个函数定义。宏 `CASE_MIN_OR_MAX_POINT` 用于根据类型进行最小值或最大值的计算，函数 `NI_MinOrMaxFilter` 则是一个用于最小值或最大值滤波的函数，涉及到对输入数组的处理和结果输出。
    /* 计算滤波器的大小： */
    for(jj = 0; jj < fsize; jj++) {
        if (pf[jj]) {
            ++filter_size;
        }
    }

    /* 获取结构体： */
    if (structure) {
        ss = malloc(filter_size * sizeof(double));
        if (!ss) {
            PyErr_NoMemory();
            goto exit;
        }
        /* 将权重复制到连续内存中： */
        ps = (npy_double*)PyArray_DATA(structure);
        jj = 0;
        for(kk = 0; kk < fsize; kk++)
            if (pf[kk])
                ss[jj++] = minimum ? -ps[kk] : ps[kk];
    }

    /* 初始化滤波器偏移量： */
    if (!NI_InitFilterOffsets(input, pf, PyArray_DIMS(footprint), origins,
                              mode, &offsets, &border_flag_value, NULL)) {
        goto exit;
    }

    /* 初始化滤波器迭代器： */
    if (!NI_InitFilterIterator(PyArray_NDIM(input), PyArray_DIMS(footprint),
                               filter_size, PyArray_DIMS(input), origins,
                               &fi)) {
        goto exit;
    }

    /* 初始化输入元素迭代器： */
    if (!NI_InitPointIterator(input, &ii))
        goto exit;

    /* 初始化输出元素迭代器： */
    if (!NI_InitPointIterator(output, &io))
        goto exit;

    NPY_BEGIN_THREADS;

    /* 获取数据指针和数组大小： */
    pi = (void *)PyArray_DATA(input);
    po = (void *)PyArray_DATA(output);
    size = PyArray_SIZE(input);

    /* 迭代处理元素： */
    oo = offsets;
    oo = offsets;
    /* 初始化滤波器偏移量数组 */
    size = PyArray_SIZE(input);
    /* 获取输入数组的大小 */
    ii = NI_InitFilterIterator(input, footprint, mode, origins, &fi);
    /* 使用输入数组、足印、扩展模式、起始位置数组和迭代器结构初始化滤波器迭代器 */
    if (!ii) {
        PyErr_SetString(PyExc_RuntimeError, "rank filter iteration failed");
        err = 1;
        goto exit;
    }
    /* 若初始化迭代器失败，设置运行时错误并跳转至退出标签，标记错误为1 */

    NPY_BEGIN_THREADS;
    /* 开始多线程执行 */

    switch(PyArray_TYPE(input)) {
        /* 根据输入数组的数据类型进行不同的处理 */
        CASE_RANK_POINT(NPY_BOOL, npy_bool, pi, offsets, filter_size,
                        cvalue != 0.0, rank, buffer, * ((npy_bool*)po), -1);
        /* 处理布尔类型的情况 */
        CASE_RANK_POINT(NPY_UINT8, npy_uint8, pi, offsets, filter_size,
                        cvalue, rank, buffer, * ((npy_uint8*)po), 255);
        /* 处理无符号8位整数类型的情况 */
        CASE_RANK_POINT(NPY_UINT16, npy_uint16, pi, offsets, filter_size,
                        cvalue, rank, buffer, * ((npy_uint16*)po), 65535);
        /* 处理无符号16位整数类型的情况 */
        CASE_RANK_POINT(NPY_UINT32, npy_uint32, pi, offsets, filter_size,
                        cvalue, rank, buffer, * ((npy_uint32*)po), 4294967295U);
        /* 处理无符号32位整数类型的情况 */
        CASE_RANK_POINT(NPY_UINT64, npy_uint64, pi, offsets, filter_size,
                        cvalue, rank, buffer, * ((npy_uint64*)po), 18446744073709551615ULL);
        /* 处理无符号64位整数类型的情况 */
        CASE_RANK_POINT(NPY_INT8, npy_int8, pi, offsets, filter_size,
                        cvalue, rank, buffer, * ((npy_int8*)po), -128);
        /* 处理有符号8位整数类型的情况 */
        CASE_RANK_POINT(NPY_INT16, npy_int16, pi, offsets, filter_size,
                        cvalue, rank, buffer, * ((npy_int16*)po), -32768);
        /* 处理有符号16位整数类型的情况 */
        CASE_RANK_POINT(NPY_INT32, npy_int32, pi, offsets, filter_size,
                        cvalue, rank, buffer, * ((npy_int32*)po), -2147483648);
        /* 处理有符号32位整数类型的情况 */
        CASE_RANK_POINT(NPY_INT64, npy_int64, pi, offsets, filter_size,
                        cvalue, rank, buffer, * ((npy_int64*)po), -9223372036854775807LL - 1);
        /* 处理有符号64位整数类型的情况 */
        CASE_RANK_POINT(NPY_FLOAT32, npy_float32, pi, offsets, filter_size,
                        cvalue, rank, buffer, * ((npy_float32*)po), NPY_NANF);
        /* 处理单精度浮点数类型的情况 */
        CASE_RANK_POINT(NPY_FLOAT64, npy_float64, pi, offsets, filter_size,
                        cvalue, rank, buffer, * ((npy_float64*)po), NPY_NAN);
        /* 处理双精度浮点数类型的情况 */
        default:
            PyErr_SetString(PyExc_RuntimeError, "data type not supported");
            err = 1;
            goto exit;
            /* 若数据类型不受支持，设置运行时错误并跳转至退出标签，标记错误为1 */
    }

exit:
    NPY_END_THREADS;
    /* 结束多线程执行 */
    if (err == 1) {
        PyErr_SetString(PyExc_RuntimeError, "array type not supported");
    }
    /* 若发生错误，设置运行时错误信息 */
    free(offsets);
    /* 释放偏移量数组的内存 */
    free(buffer);
    /* 释放缓冲区的内存 */
    return PyErr_Occurred() ? 0 : 1;
    /* 若发生异常，则返回0，否则返回1 */
}
    # 如果初始化滤波器偏移量失败，则跳转到退出标签
    if (!NI_InitFilterOffsets(input, pf, PyArray_DIMS(footprint), origins,
                              mode, &offsets, &border_flag_value, NULL)) {
        goto exit;
    }
    /* 初始化滤波器迭代器: */
    # 如果初始化滤波器迭代器失败，则跳转到退出标签
    if (!NI_InitFilterIterator(PyArray_NDIM(input), PyArray_DIMS(footprint),
                               filter_size, PyArray_DIMS(input), origins,
                               &fi)) {
        goto exit;
    }
    /* 初始化输入元素迭代器: */
    # 如果初始化输入元素迭代器失败，则跳转到退出标签
    if (!NI_InitPointIterator(input, &ii))
        goto exit;
    /* 初始化输出元素迭代器: */
    # 如果初始化输出元素迭代器失败，则跳转到退出标签
    if (!NI_InitPointIterator(output, &io))
        goto exit;

    NPY_BEGIN_THREADS;
    /* 获取数据指针和数组大小: */
    # 将输入数据指针转换为 void 指针类型
    pi = (void *)PyArray_DATA(input);
    # 将输出数据指针转换为 void 指针类型
    po = (void *)PyArray_DATA(output);
    # 获取输入数组的大小
    size = PyArray_SIZE(input);
    /* 迭代元素: */
    # 将 offsets 赋值给 oo，用于迭代
    oo = offsets;
exit:
    NPY_END_THREADS;
    // 结束 NumPy 线程

    if (err == 1) {
        // 如果错误码为1，设置运行时错误异常，说明数组类型不受支持
        PyErr_SetString(PyExc_RuntimeError, "array type not supported");
    }

    // 释放动态分配的内存
    free(offsets);
    free(buffer);

    // 检查是否发生异常，返回相应的结果（异常发生返回0，否则返回1）
    return PyErr_Occurred() ? 0 : 1;
}

int NI_GenericFilter1D(PyArrayObject *input,
            int (*function)(double*, npy_intp, double*, npy_intp, void*),
            void* data, npy_intp filter_size, int axis, PyArrayObject *output,
            NI_ExtendMode mode, double cval, npy_intp origin)
{
    int more;
    npy_intp ii, lines, length, size1, size2;
    double *ibuffer = NULL, *obuffer = NULL;
    NI_LineBuffer iline_buffer, oline_buffer;

    /* allocate and initialize the line buffers: */
    // 计算卷积核的两侧大小
    size1 = filter_size / 2;
    size2 = filter_size - size1 - 1;
    // 初始化行数为-1
    lines = -1;

    // 分配输入数据的行缓冲区
    if (!NI_AllocateLineBuffer(input, axis, size1 + origin, size2 - origin,
                                                         &lines, BUFFER_SIZE, &ibuffer))
        goto exit;

    // 分配输出数据的行缓冲区
    if (!NI_AllocateLineBuffer(output, axis, 0, 0, &lines, BUFFER_SIZE,
                                                         &obuffer))
        goto exit;

    // 初始化输入数据的行缓冲区
    if (!NI_InitLineBuffer(input, axis, size1 + origin, size2 - origin,
                                                            lines, ibuffer, mode, cval, &iline_buffer))
        goto exit;

    // 初始化输出数据的行缓冲区
    if (!NI_InitLineBuffer(output, axis, 0, 0, lines, obuffer, mode, 0.0,
                                                 &oline_buffer))
        goto exit;

    // 获取输入数组的长度
    length = PyArray_NDIM(input) > 0 ? PyArray_DIM(input, axis) : 1;

    /* iterate over all the array lines: */
    // 迭代处理所有数组行
    do {
        // 将数组行复制到缓冲区中
        if (!NI_ArrayToLineBuffer(&iline_buffer, &lines, &more)) {
            goto exit;
        }

        // 迭代处理缓冲区中的行
        for(ii = 0; ii < lines; ii++) {
            // 获取输入和输出行数据指针
            double *iline = NI_GET_LINE(iline_buffer, ii);
            double *oline = NI_GET_LINE(oline_buffer, ii);

            // 调用指定的处理函数处理行数据
            if (!function(iline, length + size1 + size2, oline, length, data)) {
                // 如果处理函数返回失败且没有设置异常，设置运行时错误异常
                if (!PyErr_Occurred()) {
                    PyErr_SetString(PyExc_RuntimeError,
                                "unknown error in line processing function");
                }
                goto exit;
            }
        }

        // 将缓冲区中的行数据写回到数组中
        if (!NI_LineBufferToArray(&oline_buffer)) {
            goto exit;
        }
    } while(more);

exit:
    // 释放动态分配的内存
    free(ibuffer);
    free(obuffer);

    // 检查是否发生异常，返回相应的结果（异常发生返回0，否则返回1）
    return PyErr_Occurred() ? 0 : 1;
}

#define CASE_FILTER_POINT(_TYPE, _type, _pi, _offsets, _filter_size, _cvalue, \
                          _res, _mv, _function, _data, _buffer)               \
case _TYPE:                                                                   \
{                                                                             \
    npy_intp _ii;                                                             \
    for (_ii = 0; _ii < _filter_size; ++_ii) {                                \
        const npy_intp _offset = _offsets[_ii];                               \
        // 获取当前偏移量
        if (_offset == _mv) {                                                 \
            // 如果偏移量等于 _mv，将 _cvalue 转换为 double 存入 _buffer
            _buffer[_ii] = (double)_cvalue;                                   \
        }                                                                     \
        else {                                                                \
            // 否则，从 _pi 加上偏移量处读取数据，转换为 double 存入 _buffer
            _buffer[_ii] = (double)(*(_type*)(_pi + _offset));                \
        }                                                                     \
    }                                                                         \
    // 调用 _function 函数，传入 _buffer、_filter_size、_res 和 _data
    if (!_function(_buffer, _filter_size, &_res, _data)) {                    \
        // 如果 _function 返回 false
        if (!PyErr_Occurred()) {                                              \
            // 如果没有其他 Python 异常发生，设置运行时错误异常
            PyErr_SetString(PyExc_RuntimeError,                               \
                            "unknown error in filter function");              \
            // 跳转到 exit 标签处，执行清理工作
            goto exit;                                                        \
        }                                                                     \
    }                                                                         \
/* 
   此函数实现了一个通用的滤波器操作，用于处理输入和输出数组中的数据。
   函数签名表明它是一个 CPython 函数，用于 NumPy 数组的滤波操作。

   参数说明：
   - input: 输入的 PyArrayObject 对象，包含待处理的数据。
   - function: 指向双精度浮点数数组的函数指针，用于定义滤波器的操作。
   - data: 一个指向任意类型数据的指针，传递给滤波函数以进行定制操作。
   - footprint: PyArrayObject 对象，定义了滤波器的形状。
   - output: 输出的 PyArrayObject 对象，用于存储滤波结果。
   - mode: 定义了滤波器在边界处的扩展模式。
   - cvalue: 在边界扩展模式为 constant 时，指定的常数值。
   - origins: 一个整数数组，定义了滤波器的原点位置。

   局部变量说明：
   - pf: 一个指向布尔型数组的指针，存储了滤波器的形状信息。
   - fsize: footprint 数组的大小。
   - jj: 用于迭代的计数器。
   - filter_size: 滤波器中有效元素的数量。
   - border_flag_value: 定义了边界扩展模式时的标志值。
   - offsets: 滤波器在输入数组中的偏移量数组。
   - oo: 用于偏移量数组迭代的指针。
   - size: 输入数组的总元素数量。
   - fi, ii, io: 分别是滤波器迭代器、输入元素迭代器和输出元素迭代器。
   - pi, po: 分别指向输入数组和输出数组的数据起始位置。
   - buffer: 用于存储滤波器计算中间结果的缓冲区。

   函数流程：
   1. 获取 footprint 的大小和数据指针。
   2. 计算滤波器中有效元素的数量。
   3. 初始化滤波器的偏移量。
   4. 初始化滤波器迭代器。
   5. 初始化输入和输出元素迭代器。
   6. 分配滤波器计算过程中需要使用的缓冲区。
   7. 迭代处理输入数组中的每个元素，执行滤波操作。
   8. 在退出时释放内存并检查错误状态。

   返回值：
   - 如果没有错误发生，则返回 1；否则返回 0。

   注意事项：
   - 在使用完滤波器操作后，需要手动释放分配的内存。
   - 此函数对应了 CPython 和 NumPy 库的接口，用于高效处理多维数组的滤波操作。
*/
int NI_GenericFilter(PyArrayObject* input,
                     int (*function)(double*, npy_intp, double*, void*), void *data,
                     PyArrayObject* footprint, PyArrayObject* output,
                     NI_ExtendMode mode, double cvalue, npy_intp *origins)
{
    npy_bool *pf = NULL;
    npy_intp fsize, jj, filter_size = 0, border_flag_value;
    npy_intp *offsets = NULL, *oo, size;
    NI_FilterIterator fi;
    NI_Iterator ii, io;
    char *pi, *po;
    double *buffer = NULL;

    /* 获取 footprint 的大小: */
    fsize = PyArray_SIZE(footprint);
    /* 获取 footprint 的数据指针: */
    pf = (npy_bool*)PyArray_DATA(footprint);
    /* 计算滤波器中有效元素的数量: */
    for(jj = 0; jj < fsize; jj++) {
        if (pf[jj])
            ++filter_size;
    }
    /* 初始化滤波器的偏移量: */
    if (!NI_InitFilterOffsets(input, pf, PyArray_DIMS(footprint), origins,
                              mode, &offsets, &border_flag_value, NULL)) {
        goto exit;
    }
    /* 初始化滤波器迭代器: */
    if (!NI_InitFilterIterator(PyArray_NDIM(input), PyArray_DIMS(footprint),
                               filter_size, PyArray_DIMS(input), origins,
                               &fi)) {
        goto exit;
    }
    /* 初始化输入元素迭代器: */
    if (!NI_InitPointIterator(input, &ii))
        goto exit;
    /* 初始化输出元素迭代器: */
    if (!NI_InitPointIterator(output, &io))
        goto exit;
    /* 获取输入和输出数组的数据指针及大小: */
    pi = (void *)PyArray_DATA(input);
    po = (void *)PyArray_DATA(output);
    size = PyArray_SIZE(input);
    /* 为滤波器计算分配缓冲区: */
    buffer = malloc(filter_size * sizeof(double));
    if (!buffer) {
        PyErr_NoMemory();
        goto exit;
    }
    /* 迭代处理每个元素: */
    oo = offsets;
    /* 退出处理: */
exit:
    free(offsets);
    free(buffer);
    /* 检查是否有错误发生，返回相应值: */
    return PyErr_Occurred() ? 0 : 1;
}
```