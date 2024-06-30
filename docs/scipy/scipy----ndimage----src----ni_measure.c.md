# `D:\src\scipysrc\scipy\scipy\ndimage\src\ni_measure.c`

```
/*
 * 版权所有 (C) 2003-2005 Peter J. Verveer
 *
 * 在源代码和二进制形式中重新分发和使用，无论是否修改，都是允许的，只要满足以下条件：
 *
 * 1. 必须保留上述版权声明、本条件列表以及以下免责声明。
 *
 * 2. 在文档和/或其他提供的材料中，必须重现上述版权声明、本条件列表以及以下免责声明。
 *
 * 3. 未经特定书面许可，不得使用作者的名字来认可或推广基于本软件的产品。
 *
 * 本软件由作者"按现状"提供，任何明示或暗示的保证，包括但不限于对适销性和特定用途适用性的暗示保证，都是被拒绝的。
 * 在任何情况下，无论是在合同诉讼、严格责任或侵权 (包括疏忽或其他方式) 的情况下，作者均不对任何直接、间接、偶发、特殊、
 * 惩罚性或后果性损害承担责任，即使事先已被告知发生这种损害的可能性。
 */

#include "ni_support.h"
#include "ni_measure.h"
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <assert.h>


#define CASE_FIND_OBJECT_POINT(_TYPE, _type, _pi, _regions, _array,  \
                               _max_label, _ii)                      \
case _TYPE:                                                          \
{                                                                    \
    int _kk;                                                         \
    // 获取数组的维度数
    npy_intp _rank = PyArray_NDIM(_array);                           \
    // 获取数组索引起始位置
    npy_intp _sindex = *(_type *)_pi - 1;                            \
    // 检查 _sindex 是否在有效范围内，并且小于 _max_label
    if (_sindex >= 0 && _sindex < _max_label) {                      \
        // 如果 _rank 大于 0，则执行以下代码块
        if (_rank > 0) {                                             \
            // 将 _sindex 扩展为其在 _regions 数组中的索引位置
            _sindex *= 2 * _rank;                                    \
            // 如果 _regions[_sindex] 小于 0，则进行以下操作
            if (_regions[_sindex] < 0) {                             \
                // 遍历 _rank 次，将 _ii.coordinates 中的坐标存入 _regions 数组中
                for (_kk = 0; _kk < _rank; _kk++) {                  \
                    npy_intp _cc = _ii.coordinates[_kk];             \
                    _regions[_sindex + _kk] = _cc;                   \
                    _regions[_sindex + _kk + _rank] = _cc + 1;       \
                }                                                    \
            }                                                        \
            // 否则，执行以下代码块
            else {                                                   \
                // 再次遍历 _rank 次，根据 _cc 和 _regions[_sindex + _kk] 的比较更新 _regions 数组
                for(_kk = 0; _kk < _rank; _kk++) {                   \
                    npy_intp _cc = _ii.coordinates[_kk];             \
                    if (_cc < _regions[_sindex + _kk]) {             \
                        _regions[_sindex + _kk] = _cc;               \
                    }                                                \
                    if (_cc + 1 > _regions[_sindex + _kk + _rank]) { \
                        _regions[_sindex + _kk + _rank] = _cc + 1;   \
                    }                                                \
                }                                                    \
            }                                                        \
        }                                                            \
        // 如果 _rank 小于等于 0，则将 _regions[_sindex] 设置为 1
        else {                                                       \
            _regions[_sindex] = 1;                                   \
        }                                                            \
    }                                                                \
}                                                                    \
break



int NI_FindObjects(PyArrayObject* input, npy_intp max_label,
                                     npy_intp* regions)
{
    npy_intp size, jj;
    NI_Iterator ii;
    char *pi;
    NPY_BEGIN_THREADS_DEF;

    NPY_BEGIN_THREADS;

    /* 获取输入数据、大小和迭代器： */
    pi = (void *)PyArray_DATA(input);  // 获取输入数组的数据指针
    size = PyArray_SIZE(input);  // 获取输入数组的大小
    if (!NI_InitPointIterator(input, &ii))  // 初始化点迭代器
        goto exit;
    if (PyArray_NDIM(input) > 0) {
        for (jj = 0; jj < 2 * PyArray_NDIM(input) * max_label; jj++) {
            regions[jj] = -1;  // 初始化 regions 数组为 -1
        }
    } else {
        for(jj = 0; jj < max_label; jj++)
            regions[jj] = -1;  // 初始化 regions 数组为 -1
    }
    /* 迭代处理所有点： */
    for(jj = 0 ; jj < size; jj++) {
        switch (PyArray_TYPE(input)) {
            CASE_FIND_OBJECT_POINT(NPY_BOOL, npy_bool,
                                   pi, regions, input, max_label, ii);  // 处理布尔类型数据点
            CASE_FIND_OBJECT_POINT(NPY_UBYTE, npy_ubyte,
                                   pi, regions, input, max_label, ii);  // 处理无符号字节类型数据点
            CASE_FIND_OBJECT_POINT(NPY_USHORT, npy_ushort,
                                   pi, regions, input, max_label, ii);  // 处理无符号短整型数据点
            CASE_FIND_OBJECT_POINT(NPY_UINT, npy_uint,
                                   pi, regions, input, max_label, ii);  // 处理无符号整型数据点
            CASE_FIND_OBJECT_POINT(NPY_ULONG, npy_ulong,
                                   pi, regions, input, max_label, ii);  // 处理无符号长整型数据点
            CASE_FIND_OBJECT_POINT(NPY_ULONGLONG, npy_ulonglong,
                                   pi, regions, input, max_label, ii);  // 处理无符号长长整型数据点
            CASE_FIND_OBJECT_POINT(NPY_BYTE, npy_byte,
                                   pi, regions, input, max_label, ii);  // 处理有符号字节类型数据点
            CASE_FIND_OBJECT_POINT(NPY_SHORT, npy_short,
                                   pi, regions, input, max_label, ii);  // 处理短整型数据点
            CASE_FIND_OBJECT_POINT(NPY_INT, npy_int,
                                   pi, regions, input, max_label, ii);  // 处理整型数据点
            CASE_FIND_OBJECT_POINT(NPY_LONG, npy_long,
                                   pi, regions, input, max_label, ii);  // 处理长整型数据点
            CASE_FIND_OBJECT_POINT(NPY_LONGLONG, npy_longlong,
                                   pi, regions, input, max_label, ii);  // 处理长长整型数据点
            CASE_FIND_OBJECT_POINT(NPY_FLOAT, npy_float,
                                   pi, regions, input, max_label, ii);  // 处理单精度浮点型数据点
            CASE_FIND_OBJECT_POINT(NPY_DOUBLE, npy_double,
                                   pi, regions, input, max_label, ii);  // 处理双精度浮点型数据点
        default:
            NPY_END_THREADS;  // 结束线程
            PyErr_SetString(PyExc_RuntimeError, "data type not supported");  // 抛出运行时错误
            goto exit;  // 跳转到 exit 标签处
        }
        NI_ITERATOR_NEXT(ii, pi);  // 迭代器指向下一个点
    }
 exit:
    NPY_END_THREADS;  // 结束线程
    return PyErr_Occurred() ? 0 : 1;  // 检查是否有异常，有异常返回 0，否则返回 1
}


#define WS_GET_INDEX(_TYPE, _type, _index, _c_strides,      \
                     _b_strides, _rank, _out,  _contiguous) \
do {                                                        \
    # 如果 `_contiguous` 为真，则执行以下操作：
    _out = _index * sizeof(_type);                      
    # 计算输出索引 `_out`，为输入索引 `_index` 乘以类型大小 `_type` 的字节数
    
    # 如果 `_contiguous` 为假，则执行以下操作：
    int _qq;                                           
    # 声明循环变量 `_qq`，用于迭代数组维度
    npy_intp _cc;                                       
    # 声明当前维度的坐标变量 `_cc`
    npy_intp _idx = _index;                             
    # 将输入索引 `_index` 复制到 `_idx`
    _out = 0;                                           
    # 初始化输出索引 `_out` 为 0
    for (_qq = 0; _qq < _rank; _qq++) {                 
        # 循环迭代数组的每一个维度
        _cc = _idx / _c_strides[_qq];                   
        # 计算当前维度的坐标 `_cc`
        _idx -= _cc * _c_strides[_qq];                  
        # 更新剩余索引 `_idx`，减去当前维度坐标乘以当前维度步长 `_c_strides[_qq]`
        _out += _b_strides[_qq] * _cc;                  
        # 更新输出索引 `_out`，加上当前维度步长乘以当前维度坐标 `_cc`
    }                                                   
    # 循环结束后，`_out` 包含了输入索引 `_index` 对应的非连续存储中的位置
} while(0)



#define CASE_GET_INPUT(_TYPE, _type, _ival, _pi) \
case _TYPE:                                      \
    _ival = *(_type *)_pi;                       \
    break



#define CASE_GET_LABEL(_TYPE, _type, _label, _pm) \
case _TYPE:                                       \
    _label = *(_type *)_pm;                       \
    break



#define CASE_PUT_LABEL(_TYPE, _type, _label, _pl) \
case _TYPE:                                       \
    *(_type *)_pl = _label;                       \
    break



#define CASE_WINDEX1(_TYPE, _type, _v_index, _p_index, _strides, _istrides, \
                     _irank, _icont, _p_idx, _v_idx, _pi, _vval, _pval)     \
case _TYPE:                                                                 \
    WS_GET_INDEX(_TYPE, _type, _v_index, _strides, _istrides, _irank,       \
                 _p_idx, _icont);                                           \
    WS_GET_INDEX(_TYPE, _type, _p_index, _strides, _istrides, _irank,       \
                 _v_idx, _icont);                                           \
    _vval = *(_type *)(_pi + _v_idx);                                       \
    _pval = *(_type *)(_pi + _p_idx);                                       \
    break



#define CASE_WINDEX2(_TYPE, _type, _v_index, _strides, _ostrides, \
                     _irank, _idx, _ocont, _label, _pl)           \
case _TYPE:                                                       \
    WS_GET_INDEX(_TYPE, _type, _v_index, _strides, _ostrides,     \
                 _irank, _idx, _ocont);                           \
    _label = *(_type *)(_pl + _idx);                              \
    break



#define CASE_WINDEX3(_TYPE, _type, _p_index, _strides, _ostrides, \
                     _irank, _idx, _ocont, _label, _pl)           \
case _TYPE:                                                       \
    WS_GET_INDEX(_TYPE, _type, _p_index, _strides, _ostrides,     \
                 _irank, _idx, _ocont);                           \
    *(_type *)(_pl + _idx) = _label;                              \
    break



#define WS_MAXDIM 7



typedef struct {
    npy_intp index;
    void *next, *prev;
    npy_uint32 cost;
    npy_uint8 done;
} NI_WatershedElement;



int NI_WatershedIFT(PyArrayObject* input, PyArrayObject* markers,
                                        PyArrayObject* strct, PyArrayObject* output)
{
    char *pl, *pm, *pi;
    int ll;
    npy_intp size, jj, hh, kk, maxval;
    npy_intp strides[WS_MAXDIM], coordinates[WS_MAXDIM];
    npy_intp *nstrides = NULL, nneigh, ssize;
    int i_contiguous, o_contiguous;
    NI_WatershedElement *temp = NULL, **first = NULL, **last = NULL;
    npy_bool *ps = NULL;
    NI_Iterator mi, ii, li;
    NPY_BEGIN_THREADS_DEF;

    i_contiguous = PyArray_ISCONTIGUOUS(input);
    o_contiguous = PyArray_ISCONTIGUOUS(output);
    ssize = PyArray_SIZE(strct);
    if (PyArray_NDIM(input) > WS_MAXDIM) {
        PyErr_SetString(PyExc_RuntimeError, "too many dimensions");
        goto exit;
    }



exit:


These annotations provide a clear explanation of each macro, variable declaration, and initial setup within the `NI_WatershedIFT` function, ensuring clarity and understanding of the code's purpose and functionality.
    size = PyArray_SIZE(input);
    /* 计算输入数组的元素总数 */

    temp = malloc(size * sizeof(NI_WatershedElement));
    /* 为临时队列数据分配内存空间 */

    if (!temp) {
        PyErr_NoMemory();
        goto exit;
    }
    /* 检查内存分配是否成功，如失败则触发内存错误并跳转到退出标签 */

    NPY_BEGIN_THREADS;
    /* 进入 NumPy 线程安全模式 */

    pi = (void *)PyArray_DATA(input);
    /* 获取输入数组的数据指针 */

    if (!NI_InitPointIterator(input, &ii))
        goto exit;
    /* 初始化输入数组的迭代器，若失败则跳转到退出标签 */

    /* 初始化并找到输入数组的最大值 */
    maxval = 0;
    for(jj = 0; jj < size; jj++) {
        npy_intp ival = 0;
        switch (PyArray_TYPE(input)) {
            CASE_GET_INPUT(NPY_UINT8, npy_uint8, ival, pi);
            CASE_GET_INPUT(NPY_UINT16, npy_uint16, ival, pi);
        default:
            NPY_END_THREADS;
            PyErr_SetString(PyExc_RuntimeError, "data type not supported");
            goto exit;
        }
        temp[jj].index = jj;
        temp[jj].done = 0;
        if (ival > maxval)
            maxval = ival;
        NI_ITERATOR_NEXT(ii, pi);
    }
    /* 根据输入数组类型获取每个元素的值，计算最大值，并存储到临时数组中 */

    pi = (void *)PyArray_DATA(input);
    /* 重新获取输入数组的数据指针 */

    /* 分配并初始化队列存储空间 */
    first = malloc((maxval + 1) * sizeof(NI_WatershedElement*));
    last = malloc((maxval + 1) * sizeof(NI_WatershedElement*));
    if (NPY_UNLIKELY(!first || !last)) {
        NPY_END_THREADS;
        PyErr_NoMemory();
        goto exit;
    }
    /* 检查队列存储空间的分配情况，若失败则触发内存错误并跳转到退出标签 */

    for(hh = 0; hh <= maxval; hh++) {
        first[hh] = NULL;
        last[hh] = NULL;
    }
    /* 初始化队列数组，将每个位置置空 */

    if (!NI_InitPointIterator(markers, &mi))
        goto exit;
    /* 初始化标记数组的迭代器，若失败则跳转到退出标签 */

    if (!NI_InitPointIterator(output, &li))
        goto exit;
    /* 初始化输出数组的迭代器，若失败则跳转到退出标签 */

    pm = (void *)PyArray_DATA(markers);
    pl = (void *)PyArray_DATA(output);
    /* 获取标记数组和输出数组的数据指针 */

    /* 初始化所有节点 */
    for (ll = 0; ll < PyArray_NDIM(input); ll++) {
        coordinates[ll] = 0;
    }
    /* 将坐标数组的每个维度都初始化为0 */

    pl = (void *)PyArray_DATA(output);
    ps = (npy_bool*)PyArray_DATA(strct);
    nneigh = 0;
    for (kk = 0; kk < ssize; kk++)
        if (ps[kk] && kk != (ssize / 2))
            ++nneigh;
    nstrides = malloc(nneigh * sizeof(npy_intp));
    if (NPY_UNLIKELY(!nstrides)) {
        NPY_END_THREADS;
        PyErr_NoMemory();
        goto exit;
    }
    /* 计算邻域结构的大小，并分配内存空间 */

    strides[PyArray_NDIM(input) - 1] = 1;
    for (ll = PyArray_NDIM(input) - 2; ll >= 0; ll--) {
        strides[ll] = PyArray_DIM(input, ll + 1) * strides[ll + 1];
    }
    /* 计算数组的步幅 */

    for (ll = 0; ll < PyArray_NDIM(input); ll++) {
        coordinates[ll] = -1;
    }
    /* 将坐标数组的每个维度初始化为-1 */

    for(kk = 0; kk < nneigh; kk++)
        nstrides[kk] = 0;
    /* 将邻域步幅数组初始化为0 */

    jj = 0;
    for(kk = 0; kk < ssize; kk++) {
        if (ps[kk]) {
            int offset = 0;
            for (ll = 0; ll < PyArray_NDIM(input); ll++) {
                offset += coordinates[ll] * strides[ll];
            }
            if (offset != 0)
                nstrides[jj++] += offset;
        }
        for (ll = PyArray_NDIM(input) - 1; ll >= 0; ll--) {
            if (coordinates[ll] < 1) {
                coordinates[ll]++;
                break;
            } else {
                coordinates[ll] = -1;
            }
        }
    }
    /* 计算邻域的偏移，并更新邻域步幅数组 */

    /* 扩散阶段 */
    }

exit:
    NPY_END_THREADS;
    /* 退出 NumPy 线程安全模式 */
    // 释放动态分配的 temp 变量所占用的内存
    free(temp);
    // 释放动态分配的 first 变量所占用的内存
    free(first);
    // 释放动态分配的 last 变量所占用的内存
    free(last);
    // 释放动态分配的 nstrides 变量所占用的内存
    free(nstrides);
    // 如果有 Python 异常发生，返回 0；否则返回 1
    return PyErr_Occurred() ? 0 : 1;
}



# 这行代码关闭了一个代码块，与之匹配的是前面的一个开放的大括号或者其他类似的语句。
```