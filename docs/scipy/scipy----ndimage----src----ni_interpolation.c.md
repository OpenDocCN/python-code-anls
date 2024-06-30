# `D:\src\scipysrc\scipy\scipy\ndimage\src\ni_interpolation.c`

```
/*
 * Copyright (C) 2003-2005 Peter J. Verveer
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above
 *    copyright notice, this list of conditions and the following
 *    disclaimer in the documentation and/or other materials provided
 *    with the distribution.
 *
 * 3. The name of the author may not be used to endorse or promote
 *    products derived from this software without specific prior
 *    written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS
 * OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
 * GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "ni_support.h"            // 引入自定义的支持函数头文件
#include "ni_interpolation.h"      // 引入自定义的插值函数头文件
#include "ni_splines.h"            // 引入自定义的样条函数头文件
#include <stdlib.h>                // 标准库：包含内存分配、随机数生成等函数
#include <math.h>                  // 标准库：包含数学函数

/* 根据指定的边界条件，映射超出边界的坐标 */
static double
map_coordinate(double in, npy_intp len, int mode)
{
    # 如果输入值小于0，执行以下操作
    if (in < 0) {
        # 根据模式进行不同的边界扩展操作
        switch (mode) {
        case NI_EXTEND_MIRROR:
            # 如果长度小于等于1，则将输入值设为0
            if (len <= 1) {
                in = 0;
            } else {
                # 计算镜像扩展后的索引位置
                npy_intp sz2 = 2 * len - 2;
                in = sz2 * (npy_intp)(-in / sz2) + in;
                in = in <= 1 - len ? in + sz2 : -in;
            }
            break;
        case NI_EXTEND_REFLECT:
            # 如果长度小于等于1，则将输入值设为0
            if (len <= 1) {
                in = 0;
            } else {
                # 计算反射扩展后的索引位置
                npy_intp sz2 = 2 * len;
                if (in < -sz2)
                    in = sz2 * (npy_intp)(-in / sz2) + in;
                // -1e-15 check to avoid possibility that: (-in - 1) == -1
                in = in < -len ? in + sz2 : (in > -1e-15 ? 1e-15 : -in) - 1;
            }
            break;
        case NI_EXTEND_WRAP:
            # 如果长度小于等于1，则将输入值设为0
            if (len <= 1) {
                in = 0;
            } else {
                npy_intp sz = len - 1;
                # 计算环绕扩展后的索引位置
                in += sz * ((npy_intp)(-in / sz) + 1);
            }
            break;
        case NI_EXTEND_GRID_WRAP:
            # 如果长度小于等于1，则将输入值设为0
            if (len <= 1) {
                in = 0;
            } else {
                # 计算网格环绕扩展后的索引位置
                in += len * ((npy_intp)((-1 - in) / len) + 1);
            }
            break;
        case NI_EXTEND_NEAREST:
            # 最近邻扩展模式下，将输入值设为0
            in = 0;
            break;
        case NI_EXTEND_CONSTANT:
            # 常数扩展模式下，将输入值设为-1
            in = -1;
            break;
        }
    } else if (in > len-1) {
        # 如果输入值大于长度减1，执行以下操作
        switch (mode) {
        case NI_EXTEND_MIRROR:
            # 如果长度小于等于1，则将输入值设为0
            if (len <= 1) {
                in = 0;
            } else {
                # 计算镜像扩展后的索引位置
                npy_intp sz2 = 2 * len - 2;
                in -= sz2 * (npy_intp)(in / sz2);
                if (in >= len)
                    in = sz2 - in;
            }
            break;
        case NI_EXTEND_REFLECT:
            # 如果长度小于等于1，则将输入值设为0
            if (len <= 1) {
                in = 0;
            } else {
                # 计算反射扩展后的索引位置
                npy_intp sz2 = 2 * len;
                in -= sz2 * (npy_intp)(in / sz2);
                if (in >= len)
                    in = sz2 - in - 1;
            }
            break;
        case NI_EXTEND_WRAP:
            # 如果长度小于等于1，则将输入值设为0
            if (len <= 1) {
                in = 0;
            } else {
                npy_intp sz = len - 1;
                # 计算环绕扩展后的索引位置
                in -= sz * (npy_intp)(in / sz);
            }
            break;
        case NI_EXTEND_GRID_WRAP:
            # 如果长度小于等于1，则将输入值设为0
            if (len <= 1) {
                in = 0;
            } else {
                # 计算网格环绕扩展后的索引位置
                in -= len * (npy_intp)(in / len);
            }
            break;
        case NI_EXTEND_NEAREST:
            # 最近邻扩展模式下，将输入值设为长度减1
            in = len - 1;
            break;
        case NI_EXTEND_CONSTANT:
            # 常数扩展模式下，将输入值设为-1
            in = -1;
            break;
        }
    }

    # 返回处理后的索引值
    return in;
}

#define BUFFER_SIZE 256000             // 定义缓冲区大小为 256000
#define TOLERANCE 1e-15                // 定义容差为 1e-15


/* one-dimensional spline filter: */
// 定义一维样条滤波器函数，接受输入数组、滤波器阶数、轴向、扩展模式和输出数组作为参数
int NI_SplineFilter1D(PyArrayObject *input, int order, int axis,
                      NI_ExtendMode mode, PyArrayObject *output)
{
    int npoles = 0, more;
    npy_intp kk, lines, len;
    double *buffer = NULL, poles[MAX_SPLINE_FILTER_POLES];   // 声明双精度数组和缓冲区指针
    NI_LineBuffer iline_buffer, oline_buffer;                // 声明输入和输出线缓冲区结构体
    NPY_BEGIN_THREADS_DEF;                                   // 定义线程开始标志

    len = PyArray_NDIM(input) > 0 ? PyArray_DIM(input, axis) : 1;  // 获取输入数组指定轴的长度，若轴数大于0，则获取该轴的维度，否则长度为1
    if (len < 1)
        goto exit;                                           // 若长度小于1，则跳转至退出标签

    /* these are used in the spline filter calculation below: */
    // 下面的样条滤波计算中使用这些变量：
    if (get_filter_poles(order, &npoles, poles)) {           // 调用函数获取滤波器极点信息，并更新滤波器阶数和极点数组，若失败则跳转至退出标签
        goto exit;
    }

    /* allocate an initialize the line buffer, only a single one is used,
         because the calculation is in-place: */
    // 分配和初始化线缓冲区，因为计算是原位的，只使用一个线缓冲区：
    lines = -1;                                              // 初始化行数为-1
    if (!NI_AllocateLineBuffer(input, axis, 0, 0, &lines, BUFFER_SIZE,
                               &buffer)) {                    // 调用函数分配线缓冲区，并设置缓冲区大小，若分配失败则跳转至退出标签
        goto exit;
    }
    if (!NI_InitLineBuffer(input, axis, 0, 0, lines, buffer,
                           NI_EXTEND_DEFAULT, 0.0, &iline_buffer)) {  // 初始化输入线缓冲区结构体，若失败则跳转至退出标签
        goto exit;
    }
    if (!NI_InitLineBuffer(output, axis, 0, 0, lines, buffer,
                           NI_EXTEND_DEFAULT, 0.0, &oline_buffer)) {  // 初始化输出线缓冲区结构体，若失败则跳转至退出标签
        goto exit;
    }
    NPY_BEGIN_THREADS;                                        // 开始线程处理

    /* iterate over all the array lines: */
    // 遍历所有数组行：
    do {
        /* copy lines from array to buffer: */
        // 将数组行复制到缓冲区：
        if (!NI_ArrayToLineBuffer(&iline_buffer, &lines, &more)) {  // 调用函数将数组内容复制到输入线缓冲区，若失败则跳转至退出标签
            goto exit;
        }
        /* iterate over the lines in the buffer: */
        // 遍历缓冲区中的行：
        for(kk = 0; kk < lines; kk++) {
            /* get line: */
            // 获取行：
            double *ln = NI_GET_LINE(iline_buffer, kk);       // 获取输入线缓冲区中第kk行的指针
            /* spline filter: */
            // 样条滤波：
            if (len > 1) {
                apply_filter(ln, len, poles, npoles, mode);   // 调用函数应用样条滤波器到当前行数据，根据长度、极点、滤波器阶数和模式进行操作
            }
        }

        /* copy lines from buffer to array: */
        // 将缓冲区中的行复制回数组：
        if (!NI_LineBufferToArray(&oline_buffer)) {           // 调用函数将输出线缓冲区内容复制回数组，若失败则跳转至退出标签
            goto exit;
        }
    } while(more);

 exit:
    NPY_END_THREADS;                                          // 结束线程处理
    free(buffer);                                             // 释放缓冲区内存
    return PyErr_Occurred() ? 0 : 1;                          // 返回异常状态，若异常则返回0，否则返回1
}

/* copy row of coordinate array from location at _p to _coor */
// 从位置_p复制坐标数组的行到_coor
#define CASE_MAP_COORDINATES(_TYPE, _type, _p, _coor, _rank, _stride) \
case _TYPE:                                                           \
{                                                                     \
    npy_intp _hh;                                                     \
    for (_hh = 0; _hh < _rank; ++_hh) {                               \
        _coor[_hh] = *(_type *)_p;                                    // 将_p指向的类型转换为_type指针解引用并赋值给_coor[_hh]
        _p += _stride;                                                // 更新_p指针位置
    }                                                                 \
}                                                                     \
break

#define CASE_INTERP_COEFF(_TYPE, _type, _coeff, _pi, _idx) \
case _TYPE:                                                \
    _coeff = *(_type *)(_pi + _idx);                       // 将(_pi + _idx)处的数据转换为_type类型并赋值给_coeff
    break

#define CASE_INTERP_OUT(_TYPE, _type, _po, _t) \
    case _TYPE:                                    \
        // 将输入的_t值转换为_type类型，并存储到指定地址_po中
        *(_type *)_po = (_type)_t;                 \
        // 结束当前的case分支
        break

#define CASE_INTERP_OUT_UINT(_TYPE, _type, _po, _t)  \
    case NPY_##_TYPE:                                    \
        // 将_t值加上0.5（四舍五入），确保_t大于等于0
        _t = _t > 0 ? _t + 0.5 : 0;                      \
        // 如果_t超过了NPY_MAX_##_TYPE，则将_t设为NPY_MAX_##_TYPE
        _t = _t > NPY_MAX_##_TYPE ? NPY_MAX_##_TYPE : t; \
        // 如果_t小于0，则将_t设为0
        _t = _t < 0 ? 0 : t;                             \
        // 将_t值转换为_type类型，并存储到指定地址_po中
        *(_type *)_po = (_type)_t;                       \
        // 结束当前的case分支
        break

#define CASE_INTERP_OUT_INT(_TYPE, _type, _po, _t)   \
    case NPY_##_TYPE:                                    \
        // 将_t值加上0.5或者减去0.5（四舍五入），确保_t在范围内
        _t = _t > 0 ? _t + 0.5 : _t - 0.5;               \
        // 如果_t超过了NPY_MAX_##_TYPE，则将_t设为NPY_MAX_##_TYPE
        _t = _t > NPY_MAX_##_TYPE ? NPY_MAX_##_TYPE : t; \
        // 如果_t小于NPY_MIN_##_TYPE，则将_t设为NPY_MIN_##_TYPE
        _t = _t < NPY_MIN_##_TYPE ? NPY_MIN_##_TYPE : t; \
        // 将_t值转换为_type类型，并存储到指定地址_po中
        *(_type *)_po = (_type)_t;                       \
        // 结束当前的case分支
        break

int _get_spline_boundary_mode(int mode)
{
    // 如果mode为NI_EXTEND_CONSTANT或者NI_EXTEND_WRAP，则返回NI_EXTEND_MIRROR
    if ((mode == NI_EXTEND_CONSTANT) || (mode == NI_EXTEND_WRAP))
        // 没有分析前处理或显式前填充的模式使用镜像扩展
        return NI_EXTEND_MIRROR;
    // 否则返回mode本身
    return mode;
}

int
NI_GeometricTransform(PyArrayObject *input, int (*map)(npy_intp*, double*,
                int, int, void*), void* map_data, PyArrayObject* matrix_ar,
                PyArrayObject* shift_ar, PyArrayObject *coordinates,
                PyArrayObject *output, int order, int mode, double cval,
                int nprepad)
{
    // 指针初始化
    char *po, *pi, *pc = NULL;
    // 数组和指针的声明
    npy_intp **edge_offsets = NULL, **data_offsets = NULL, filter_size;
    // 边界格点的常量指针数组
    char **edge_grid_const = NULL;
    // 临时整型数组，坐标和偏移
    npy_intp ftmp[NPY_MAXDIMS], *fcoordinates = NULL, *foffsets = NULL;
    // 坐标步长，循环变量声明
    npy_intp cstride = 0, kk, hh, ll, jj;
    // 数组大小，样条值的数组，输入维度和步长
    npy_intp size;
    double **splvals = NULL, icoor[NPY_MAXDIMS];
    // 输入和输出的维度和步长
    npy_intp idimensions[NPY_MAXDIMS], istrides[NPY_MAXDIMS];
    // 迭代器定义
    NI_Iterator io, ic;
    // 矩阵和偏移数组指针
    npy_double *matrix = matrix_ar ? (npy_double*)PyArray_DATA(matrix_ar) : NULL;
    npy_double *shift = shift_ar ? (npy_double*)PyArray_DATA(shift_ar) : NULL;
    // 输入和输出的秩，样条模式
    int irank = 0, orank, spline_mode;
    // 线程开始
    NPY_BEGIN_THREADS_DEF;

    NPY_BEGIN_THREADS;

    // 遍历输入数组的维度
    for(kk = 0; kk < PyArray_NDIM(input); kk++) {
        // 存储输入数组的维度和步长
        idimensions[kk] = PyArray_DIM(input, kk);
        istrides[kk] = PyArray_STRIDE(input, kk);
    }
    // 存储输入数组的秩
    irank = PyArray_NDIM(input);
    // 存储输出数组的秩
    orank = PyArray_NDIM(output);

    /* if the mapping is from array coordinates: */
    // 如果映射是从数组坐标进行的：
    if (coordinates) {
        /* initialize a line iterator along the first axis: */
        // 初始化沿第一个轴的线迭代器：
        if (!NI_InitPointIterator(coordinates, &ic))
            goto exit;
        // 存储坐标数组第一个轴的步长
        cstride = ic.strides[0];
        // 初始化坐标数组的线迭代器
        if (!NI_LineIterator(&ic, 0))
            goto exit;
        // 将坐标数组数据转换为void指针类型
        pc = (void *)(PyArray_DATA(coordinates));
    }

    /* offsets used at the borders: */
    // 边界使用的偏移量：
    // 分配内存以存储边界和数据偏移量
    edge_offsets = malloc(irank * sizeof(npy_intp*));
    data_offsets = malloc(irank * sizeof(npy_intp*));
    // 如果内存分配失败，结束线程，抛出内存错误，跳转到exit标签
    if (NPY_UNLIKELY(!edge_offsets || !data_offsets)) {
        NPY_END_THREADS;
        PyErr_NoMemory();
        goto exit;
    }
    if (mode == NI_EXTEND_GRID_CONSTANT) {
        // 如果模式是扩展到常量边界

        // 分配内存以存储边界网格常量的指针数组
        edge_grid_const = malloc(irank * sizeof(char*));
        if (NPY_UNLIKELY(!edge_grid_const)) {
            // 如果内存分配失败，则结束线程并报告内存错误
            NPY_END_THREADS;
            PyErr_NoMemory();
            goto exit;
        }
        // 初始化每个指针为NULL
        for(jj = 0; jj < irank; jj++)
            edge_grid_const[jj] = NULL;
        // 为每个指针分配内存，存储常量边界的状态
        for(jj = 0; jj < irank; jj++) {
            edge_grid_const[jj] = malloc((order + 1) * sizeof(char));
            if (NPY_UNLIKELY(!edge_grid_const[jj])) {
                // 如果内存分配失败，则结束线程并报告内存错误
                NPY_END_THREADS;
                PyErr_NoMemory();
                goto exit;
            }
        }
    }

    // 初始化每个数据偏移的指针为NULL
    for(jj = 0; jj < irank; jj++)
        data_offsets[jj] = NULL;
    // 为每个数据偏移分配内存
    for(jj = 0; jj < irank; jj++) {
        data_offsets[jj] = malloc((order + 1) * sizeof(npy_intp));
        if (NPY_UNLIKELY(!data_offsets[jj])) {
            // 如果内存分配失败，则结束线程并报告内存错误
            NPY_END_THREADS;
            PyErr_NoMemory();
            goto exit;
        }
    }
    /* 将用于存储样条插值系数的数组初始化: */
    // 分配内存以存储每个维度的样条插值系数
    splvals = malloc(irank * sizeof(double*));
    if (NPY_UNLIKELY(!splvals)) {
        // 如果内存分配失败，则结束线程并报告内存错误
        NPY_END_THREADS;
        PyErr_NoMemory();
        goto exit;
    }
    // 初始化每个指针为NULL
    for(jj = 0; jj < irank; jj++)
        splvals[jj] = NULL;
    // 为每个维度的每个插值点分配内存
    for(jj = 0; jj < irank; jj++) {
        splvals[jj] = malloc((order + 1) * sizeof(double));
        if (NPY_UNLIKELY(!splvals[jj])) {
            // 如果内存分配失败，则结束线程并报告内存错误
            NPY_END_THREADS;
            PyErr_NoMemory();
            goto exit;
        }
    }

    // 计算滤波器的大小
    filter_size = 1;
    for(jj = 0; jj < irank; jj++)
        filter_size *= order + 1;

    /* 初始化输出迭代器: */
    // 初始化输出迭代器，若失败则跳转至退出标签
    if (!NI_InitPointIterator(output, &io))
        goto exit;

    /* 获取数据指针: */
    // 获取输入和输出数组的数据指针
    pi = (void *)PyArray_DATA(input);
    po = (void *)PyArray_DATA(output);

    /* 创建所有可能在样条滤波器内的坐标表: */
    // 分配内存以存储样条滤波器内的所有坐标
    fcoordinates = malloc(irank * filter_size * sizeof(npy_intp));
    /* 创建样条滤波器内所有偏移的表: */
    // 分配内存以存储样条滤波器内的所有偏移量
    foffsets = malloc(filter_size * sizeof(npy_intp));
    if (NPY_UNLIKELY(!fcoordinates || !foffsets)) {
        // 如果内存分配失败，则结束线程并报告内存错误
        NPY_END_THREADS;
        PyErr_NoMemory();
        goto exit;
    }
    // 初始化临时数组ftmp为0
    for(jj = 0; jj < irank; jj++)
        ftmp[jj] = 0;
    kk = 0;
    // 生成样条滤波器内的所有坐标和偏移
    for(hh = 0; hh < filter_size; hh++) {
        for(jj = 0; jj < irank; jj++)
            fcoordinates[jj + hh * irank] = ftmp[jj];
        foffsets[hh] = kk;
        for(jj = irank - 1; jj >= 0; jj--) {
            if (ftmp[jj] < order) {
                ftmp[jj]++;
                kk += istrides[jj];
                break;
            } else {
                ftmp[jj] = 0;
                kk -= istrides[jj] * order;
            }
        }
    }

    // 获取样条插值边界模式
    spline_mode = _get_spline_boundary_mode(mode);

    // 获取输出数组的大小
    size = PyArray_SIZE(output);
    }

 exit:
    // 结束线程
    NPY_END_THREADS;
    // 释放内存
    free(edge_offsets);
    # 如果 edge_grid_const 不为 NULL，则释放其内存
    if (edge_grid_const) {
        for(jj = 0; jj < irank; jj++)
            free(edge_grid_const[jj]);
        free(edge_grid_const);
    }
    # 如果 data_offsets 不为 NULL，则释放其内存
    if (data_offsets) {
        for(jj = 0; jj < irank; jj++)
            free(data_offsets[jj]);
        free(data_offsets);
    }
    # 如果 splvals 不为 NULL，则释放其内存
    if (splvals) {
        for(jj = 0; jj < irank; jj++)
            free(splvals[jj]);
        free(splvals);
    }
    # 释放 foffsets 所指向的内存
    free(foffsets);
    # 释放 fcoordinates 所指向的内存
    free(fcoordinates);
    # 检查是否有 Python 异常发生，若有则返回 0，否则返回 1
    return PyErr_Occurred() ? 0 : 1;
    // 指向输出数组和输入数组的指针
    char *po, *pi;
    // 存储用于不同处理的数组
    npy_intp **zeros = NULL, **offsets = NULL, ***edge_offsets = NULL;
    // 用于临时存储数组和偏移量的数组
    npy_intp ftmp[NPY_MAXDIMS], *fcoordinates = NULL, *foffsets = NULL;
    // 用于存储各维度大小的数组、滤波器大小、输出数组的维度
    npy_intp jj, hh, kk, filter_size, odimensions[NPY_MAXDIMS];
    // 输入数组的维度、步幅、大小
    npy_intp idimensions[NPY_MAXDIMS], istrides[NPY_MAXDIMS];
    // 大小
    npy_intp size;
    // 存储边缘网格的常数数组
    char ***edge_grid_const = NULL;
    // 存储插值系数的数组
    double ***splvals = NULL;
    // 迭代器对象
    NI_Iterator io;
    // 缩放数组和平移数组的指针
    npy_double *zooms = zoom_ar ? (npy_double*)PyArray_DATA(zoom_ar) : NULL;
    npy_double *shifts = shift_ar ? (npy_double*)PyArray_DATA(shift_ar) : NULL;
    // 维度的排列顺序
    int rank = 0;
    // 启动线程
    NPY_BEGIN_THREADS_DEF;

    NPY_BEGIN_THREADS;

    // 初始化维度信息
    for (kk = 0; kk < PyArray_NDIM(input); kk++) {
        idimensions[kk] = PyArray_DIM(input, kk);
        istrides[kk] = PyArray_STRIDE(input, kk);
        odimensions[kk] = PyArray_DIM(output, kk);
    }
    rank = PyArray_NDIM(input);

    // 如果模式是 'constant'，分配零值数组
    if (mode == NI_EXTEND_CONSTANT) {
        zeros = malloc(rank * sizeof(npy_intp*));
        if (NPY_UNLIKELY(!zeros)) {
            NPY_END_THREADS;
            PyErr_NoMemory();
            goto exit;
        }
        for(jj = 0; jj < rank; jj++)
            zeros[jj] = NULL;
        for(jj = 0; jj < rank; jj++) {
            zeros[jj] = malloc(odimensions[jj] * sizeof(npy_intp));
            if (NPY_UNLIKELY(!zeros[jj])) {
                NPY_END_THREADS;
                PyErr_NoMemory();
                goto exit;
            }
        }
    } else if (mode == NI_EXTEND_GRID_CONSTANT) {
        // 如果模式是 'grid constant'，分配边缘网格常数数组
        edge_grid_const = malloc(rank * sizeof(char*));
        if (NPY_UNLIKELY(!edge_grid_const)) {
            NPY_END_THREADS;
            PyErr_NoMemory();
            goto exit;
        }
        for(jj = 0; jj < rank; jj++)
            edge_grid_const[jj] = NULL;
        for(jj = 0; jj < rank; jj++) {
            edge_grid_const[jj] = malloc(odimensions[jj] * sizeof(char*));
            if (NPY_UNLIKELY(!edge_grid_const[jj])) {
                NPY_END_THREADS;
                PyErr_NoMemory();
                goto exit;
            }
            for(hh = 0; hh < odimensions[jj]; hh++) {
                edge_grid_const[jj][hh] = NULL;
            }
        }
    }

    // 分配偏移数组
    offsets = malloc(rank * sizeof(npy_intp*));
    // 分配插值系数数组
    splvals = malloc(rank * sizeof(double**));
    // 如果分配不成功，结束线程
    if (NPY_UNLIKELY(!offsets || !splvals)) {
        NPY_END_THREADS;
        PyErr_NoMemory();
        goto exit;
    }
    for(jj = 0; jj < rank; jj++) {
        offsets[jj] = NULL;
        splvals[jj] = NULL;
    }
    for(jj = 0; jj < rank; jj++) {
        // 分配偏移数组和插值数组的内存空间
        offsets[jj] = malloc(odimensions[jj] * sizeof(npy_intp));
        splvals[jj] = malloc(odimensions[jj] * sizeof(double*));
        // 检查内存分配是否成功，若失败则释放已分配的资源并跳转到退出标签
        if (NPY_UNLIKELY(!offsets[jj] || !splvals[jj])) {
            NPY_END_THREADS;
            PyErr_NoMemory();
            goto exit;
        }
        // 初始化插值数组为 NULL
        for(hh = 0; hh < odimensions[jj]; hh++) {
            splvals[jj][hh] = NULL;
        }
    }

    // 根据模式分配边缘偏移数组的内存空间
    if (mode != NI_EXTEND_GRID_CONSTANT){
        edge_offsets = malloc(rank * sizeof(npy_intp**));
        // 检查内存分配是否成功，若失败则释放已分配的资源并跳转到退出标签
        if (NPY_UNLIKELY(!edge_offsets)) {
            NPY_END_THREADS;
            PyErr_NoMemory();
            goto exit;
        }
        // 初始化边缘偏移数组为 NULL
        for(jj = 0; jj < rank; jj++) {
            edge_offsets[jj] = NULL;
        }
        // 分配每个维度的边缘偏移数组的内存空间
        for(jj = 0; jj < rank; jj++) {
            edge_offsets[jj] = malloc(odimensions[jj] * sizeof(npy_intp*));
            // 检查内存分配是否成功，若失败则释放已分配的资源并跳转到退出标签
            if (NPY_UNLIKELY(!edge_offsets[jj])) {
                NPY_END_THREADS;
                PyErr_NoMemory();
                goto exit;
            }
            // 初始化每个维度的边缘偏移为 NULL
            for(hh = 0; hh < odimensions[jj]; hh++) {
                edge_offsets[jj][hh] = NULL;
            }
        }
    }

    // 获取样条插值的边界模式
    int spline_mode = _get_spline_boundary_mode(mode);

    // 计算滤波器的大小
    filter_size = 1;
    for(jj = 0; jj < rank; jj++)
        filter_size *= order + 1;

    // 初始化输出点迭代器
    if (!NI_InitPointIterator(output, &io))
        goto exit;

    // 获取输入和输出数组的数据指针
    pi = (void *)PyArray_DATA(input);
    po = (void *)PyArray_DATA(output);

    /* store all coordinates and offsets with filter: */
    // 分配存储所有坐标和偏移量的内存空间
    fcoordinates = malloc(rank * filter_size * sizeof(npy_intp));
    foffsets = malloc(filter_size * sizeof(npy_intp));
    // 检查内存分配是否成功，若失败则释放已分配的资源并跳转到退出标签
    if (NPY_UNLIKELY(!fcoordinates || !foffsets)) {
        NPY_END_THREADS;
        PyErr_NoMemory();
        goto exit;
    }

    // 初始化临时数组为 0
    for(jj = 0; jj < rank; jj++)
        ftmp[jj] = 0;
    kk = 0;
    // 填充坐标和偏移数组
    for(hh = 0; hh < filter_size; hh++) {
        for(jj = 0; jj < rank; jj++)
            fcoordinates[jj + hh * rank] = ftmp[jj];
        foffsets[hh] = kk;
        // 更新临时数组以生成所有可能的坐标
        for(jj = rank - 1; jj >= 0; jj--) {
            if (ftmp[jj] < order) {
                ftmp[jj]++;
                kk += istrides[jj];
                break;
            } else {
                ftmp[jj] = 0;
                kk -= istrides[jj] * order;
            }
        }
    }
    // 获取输出数组的大小
    size = PyArray_SIZE(output);
    // 退出标签，清理线程，处理内存释放
    }

 exit:
    // 结束线程
    NPY_END_THREADS;
    // 释放内存：释放零数组
    if (zeros) {
        for(jj = 0; jj < rank; jj++)
            free(zeros[jj]);
        free(zeros);
    }
    // 释放内存：释放偏移数组
    if (offsets) {
        for(jj = 0; jj < rank; jj++)
            free(offsets[jj]);
        free(offsets);
    }
    // 释放内存：释放插值数组
    if (splvals) {
        for(jj = 0; jj < rank; jj++) {
            if (splvals[jj]) {
                for(hh = 0; hh < odimensions[jj]; hh++)
                    free(splvals[jj][hh]);
                free(splvals[jj]);
            }
        }
        free(splvals);
    }
    // 检查并释放 edge_offsets 数组及其内存
    if (edge_offsets) {
        // 遍历每个维度
        for(jj = 0; jj < rank; jj++) {
            // 如果当前维度的 edge_offsets 不为 NULL
            if (edge_offsets[jj]) {
                // 释放当前维度的每个元素内存
                for(hh = 0; hh < odimensions[jj]; hh++)
                    free(edge_offsets[jj][hh]);
                // 释放当前维度的整体内存
                free(edge_offsets[jj]);
            }
        }
        // 释放 edge_offsets 的内存
        free(edge_offsets);
    }
    
    // 检查并释放 edge_grid_const 数组及其内存
    if (edge_grid_const) {
        // 遍历每个维度
        for(jj = 0; jj < rank; jj++) {
            // 如果当前维度的 edge_grid_const 不为 NULL
            if (edge_grid_const[jj]) {
                // 释放当前维度的每个元素内存
                for(hh = 0; hh < odimensions[jj]; hh++)
                    free(edge_grid_const[jj][hh]);
                // 释放当前维度的整体内存
                free(edge_grid_const[jj]);
            }
        }
        // 释放 edge_grid_const 的内存
        free(edge_grid_const);
    }
    
    // 释放 foffsets 数组的内存
    free(foffsets);
    // 释放 fcoordinates 数组的内存
    free(fcoordinates);
    // 返回是否发生 Python 异常的结果（0 表示有异常，1 表示无异常）
    return PyErr_Occurred() ? 0 : 1;
}
```