# `D:\src\scipysrc\scipy\scipy\ndimage\src\ni_morphology.c`

```
/*
 * 版权所有 (C) 2003-2005 Peter J. Verveer
 *
 * 在源代码和二进制形式中重新分发和使用，无论是否经过修改，
 * 都允许，前提是满足以下条件：
 *
 * 1. 源代码的再分发必须保留上述版权声明、此条件列表以及以下免责声明。
 *
 * 2. 以二进制形式再分发时，必须在提供的文档和/或其他材料中重复
 *    上述版权声明、此条件列表以及以下免责声明。
 *
 * 3. 未经特定的书面许可，不得使用作者的名称来认可或推广从本软件
 *    衍生的产品。
 *
 * 本软件按原样提供，作者不承担任何明示或暗示的担保责任，
 * 包括但不限于对适销性和特定用途的适用性的担保责任。
 * 在任何情况下，无论是合同责任、严格责任还是侵权行为（包括疏忽或其他）
 * ，都不应对任何直接、间接、偶然、特殊、示范性或后果性损害
 * （包括但不限于替代商品或服务的采购、使用、数据或利润损失；
 * 或业务中断）负责，即使事先已被告知此类损害的可能性。
 */

#include "ni_support.h"
#include "ni_morphology.h"
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <float.h>

#define LIST_SIZE 100000

#define CASE_GET_MASK(_TYPE, _type, _msk_value, _pm) \
case _TYPE:                                          \
    _msk_value = *(_type *)_pm ? 1 : 0;              \
    break

#define CASE_OUTPUT(_TYPE, _type, _po, _out) \
case _TYPE:                                  \
    *(_type *)_po = (_type)_out;             \
    break

#define CASE_NI_ERODE_POINT(_TYPE, _type, _pi, _out, _offsets,        \
                            _filter_size, _mv,  _border_value, _bv,   \
                            _center_is_true, _true, _false, _changed) \
case _TYPE:                                                           \
{                                                                     \
    npy_intp _ii, _oo;                                                \
    int _in = *(_type *)_pi ? 1 : 0;                                  \
*/
    # 如果条件 _mv 为真，则执行以下语句块
    if (_mv) {                                                        \
        # 如果 _center_is_true 为真且 _in 为假，则执行以下语句块
        if (_center_is_true && _in == _false) {                       \
            # 将 _changed 设为 0
            _changed = 0;                                             \
            # 将 _out 设为 _in
            _out = _in;                                               \
        }                                                             \
        # 否则，执行以下语句块
        else {                                                        \
            # 将 _out 设为 _true
            _out = _true;                                             \
            # 对于每一个滤波器的偏移量 _oo
            for (_ii = 0; _ii < _filter_size; _ii++) {                \
                # 取出偏移量 _oo
                _oo = _offsets[_ii];                                  \
                # 如果 _oo 等于 _bv，则执行以下语句块
                if (_oo == _bv) {                                     \
                    # 如果 _border_value 不为真，则将 _out 设为 _false 并跳出循环
                    if (!_border_value) {                             \
                        _out = _false;                                \
                        break;                                        \
                    }                                                 \
                }                                                     \
                # 否则，执行以下语句块
                else {                                                \
                    # 定义局部变量 _nn，根据 _pi + _oo 处的值判断真假，并赋给 _nn
                    int _nn = *(_type *)(_pi + _oo) ? _true : _false; \
                    # 如果 _nn 不为真，则将 _out 设为 _false 并跳出循环
                    if (!_nn) {                                       \
                        _out = _false;                                \
                        break;                                        \
                    }                                                 \
                }                                                     \
            }                                                         \
            # 将 _changed 设为 _out 是否不等于 _in 的结果
            _changed = _out != _in;                                   \
        }                                                             \
    }                                                                 \
    # 如果条件 _mv 不为真，则将 _out 设为 _in
    else {                                                            \
        _out = _in;                                                   \
    }                                                                 \
/* 
   结束当前函数的执行并返回，通常用于错误处理或特定条件下的流程中断
*/
}                                                                     \
break

/* 
   二值侵蚀操作的实现函数，用于处理输入图像的侵蚀操作。
   参数说明：
   - input: 输入图像的数组对象
   - strct: 结构元素的数组对象
   - mask: 可选的掩模数组对象
   - output: 输出图像的数组对象
   - bdr_value: 边界值
   - origins: 结构元素的原点坐标
   - invert: 是否反转操作
   - center_is_true: 中心值是否为真值
   - changed: 用于记录是否有像素值发生改变的整型指针
   - coordinate_list: 可选的坐标列表
*/
int NI_BinaryErosion(PyArrayObject* input, PyArrayObject* strct,
                PyArrayObject* mask, PyArrayObject* output, int bdr_value,
                     npy_intp *origins, int invert, int center_is_true,
                     int* changed, NI_CoordinateList **coordinate_list)
{
    npy_intp struct_size = 0, *offsets = NULL, size, *oo, jj;
    npy_intp ssize, block_size = 0, *current = NULL, border_flag_value;
    int kk, _true, _false, msk_value;
    NI_Iterator ii, io, mi;
    NI_FilterIterator fi;
    npy_bool *ps, out = 0;
    char *pi, *po, *pm = NULL;
    NI_CoordinateBlock *block = NULL;
    NPY_BEGIN_THREADS_DEF;

    /* 初始化结构元素数据指针 */
    ps = (npy_bool*)PyArray_DATA(strct);
    /* 结构元素数组的尺寸 */
    ssize = PyArray_SIZE(strct);
    /* 计算结构元素中为真值的数量 */
    for(jj = 0; jj < ssize; jj++)
        if (ps[jj]) ++struct_size;
    
    /* 如果有掩模数组，则初始化掩模点迭代器 */
    if (mask) {
        if (!NI_InitPointIterator(mask, &mi))
            return 0;
        pm = (void *)PyArray_DATA(mask);
    }

    /* 计算滤波器的偏移量 */
    if (!NI_InitFilterOffsets(input, ps, PyArray_DIMS(strct), origins,
                              NI_EXTEND_CONSTANT, &offsets, &border_flag_value,
                              NULL)) {
        goto exit;
    }

    /* 初始化输入元素迭代器 */
    if (!NI_InitPointIterator(input, &ii))
        goto exit;

    /* 初始化输出元素迭代器 */
    if (!NI_InitPointIterator(output, &io))
        goto exit;

    /* 初始化滤波器迭代器 */
    if (!NI_InitFilterIterator(PyArray_NDIM(input), PyArray_DIMS(strct),
                               struct_size, PyArray_DIMS(input), origins,
                               &fi)) {
        goto exit;
    }

    /* 多线程操作开始 */
    NPY_BEGIN_THREADS;

    /* 获取输入和输出数据指针及尺寸 */
    pi = (void *)PyArray_DATA(input);
    po = (void *)PyArray_DATA(output);
    size = PyArray_SIZE(input);

    /* 如果需要反转操作，则设置边界值和真假值 */
    if (invert) {
        bdr_value = bdr_value ? 0 : 1;
        _true = 0;
        _false = 1;
    } else {
        bdr_value = bdr_value ? 1 : 0;
        _true = 1;
        _false = 0;
    }

    /* 如果需要记录坐标列表 */
    if (coordinate_list) {
        block_size = LIST_SIZE / PyArray_NDIM(input) / sizeof(int);
        if (block_size < 1)
            block_size = 1;
        if (block_size > size)
            block_size = size;
        *coordinate_list = NI_InitCoordinateList(block_size,
                                                 PyArray_NDIM(input));
        /* 内存分配失败处理 */
        if (NPY_UNLIKELY(!*coordinate_list)) {
            NPY_END_THREADS;
            PyErr_NoMemory();
            goto exit;
        }
    }

    /* 元素迭代循环开始 */
    oo = offsets;
    *changed = 0;
    msk_value = 1;
    /* 注意：此处应有循环逻辑，但由于代码片段不完整，具体迭代逻辑未能完全展示 */

exit:
    /* 多线程操作结束 */
    NPY_END_THREADS;
    /* 释放偏移数组的内存 */
    free(offsets);
    /* 检查是否发生 Python 错误 */
    if (PyErr_Occurred()) {
        /* 如果有坐标列表需要释放，则调用相应函数 */
        if (coordinate_list) {
            NI_FreeCoordinateList(*coordinate_list);
            *coordinate_list = NULL;
        }
        return 0;
    } else {
        return 1;
    }
}
# 定义宏，用于处理二维数组中的腐蚀操作，根据不同数据类型和结构体大小生成对应的代码
#define CASE_ERODE_POINT2(_TYPE, _type, _struct_size, _offsets,               \
                          _coordinate_offsets, _pi, _oo, _irank,              \
                          _list1, _list2,                                     \
                          _current_coors1, _current_coors2,                   \
                          _block1, _block2,                                   \
                          _bf_value, _true, _false, _mklist)                  \
case _TYPE:                                                                   \
{                                                                             \
    npy_intp _hh, _kk;                                                        \
    // 遍历结构体大小，处理每一个偏移量
    for (_hh = 0; _hh < _struct_size; _hh++) {                                \
        // 获取当前偏移量
        npy_intp _to = _offsets[_oo + _hh];                                   \
        // 如果偏移量不为预设的特殊值，并且对应位置的值等于真值
        if (_to != _bf_value && *(_type *)(_pi + _to) == (_type)_true) {      \
            // 如果需要生成坐标列表
            if (_mklist) {                                                    \
                // 获取当前坐标偏移量
                npy_intp *_tc = &(_coordinate_offsets[(_oo + _hh) * _irank]); \
                // 如果坐标列表为空或者达到块的大小，添加新的块
                if (_block2 == NULL || _block2->size == _list2->block_size) { \
                    _block2 = NI_CoordinateListAddBlock(_list2);              \
                    // 如果添加失败，释放线程并报内存错误，跳转到退出点
                    if (NPY_UNLIKELY(_block2 == NULL)) {                      \
                        NPY_END_THREADS;                                      \
                        PyErr_NoMemory();                                     \
                        goto exit;                                            \
                    }                                                         \
                    _current_coors2 = _block2->coordinates;                   \
                }                                                             \
                // 根据偏移量计算新的坐标
                for (_kk = 0; _kk < _irank; _kk++) {                          \
                    *_current_coors2++ = _current_coors1[_kk] + _tc[_kk];     \
                }                                                             \
                // 坐标块大小增加
                _block2->size++;                                              \
            }                                                                 \
            // 将处理过的位置值设置为假值
            *(_type *)(_pi + _to) = _false;                                   \
        }                                                                     \
    }                                                                         \
}                                                                             \
break

// 二值侵蚀操作的主函数，处理给定的二维数组，结构数组和掩码，可以进行多次迭代
int NI_BinaryErosion2(PyArrayObject* array, PyArrayObject* strct,
                      PyArrayObject* mask, int niter, npy_intp *origins,
                      int invert, NI_CoordinateList **iclist)
{
    npy_intp struct_size = 0, *offsets = NULL, oo, jj, ssize;
    npy_intp *coordinate_offsets = NULL, size = 0;
    npy_intp *current_coordinates1 = NULL, *current_coordinates2 = NULL;
    npy_intp border_flag_value, current = 0;
    int _true, _false;
    NI_Iterator ii, mi;
    NI_FilterIterator fi, ci;
    npy_bool *ps;
    char *pi, *ibase, *pm = NULL;
    NI_CoordinateBlock *block1 = NULL, *block2 = NULL;
    NI_CoordinateList *list1 = NULL, *list2 = NULL;
    NPY_BEGIN_THREADS_DEF;

    // 获取结构数组的数据指针并计算其大小
    ps = (npy_bool*)PyArray_DATA(strct);
    ssize = PyArray_SIZE(strct);
    for(jj = 0; jj < ssize; jj++)
        if (ps[jj]) ++struct_size;

    /* calculate the filter offsets: */
    // 计算滤波器的偏移量
    if (!NI_InitFilterOffsets(array, ps, PyArray_DIMS(strct), origins,
                              NI_EXTEND_CONSTANT, &offsets,
                              &border_flag_value, &coordinate_offsets)) {
        goto exit;
    }

    /* initialize input element iterator: */
    // 初始化输入元素迭代器
    if (!NI_InitPointIterator(array, &ii))
        goto exit;

    /* initialize filter iterator: */
    // 初始化滤波器迭代器
    if (!NI_InitFilterIterator(PyArray_NDIM(array), PyArray_DIMS(strct),
                               struct_size, PyArray_DIMS(array), origins,
                               &fi)) {
        goto exit;
    }
    if (!NI_InitFilterIterator(PyArray_NDIM(array), PyArray_DIMS(strct),
                               struct_size * PyArray_NDIM(array),
                               PyArray_DIMS(array), origins, &ci)) {
        goto exit;
    }

    /* get data pointers and size: */
    // 获取数据指针和数组大小
    ibase = pi = (void *)PyArray_DATA(array);

    if (invert) {
        _true = 0;
        _false = 1;
    } else {
        _true = 1;
        _false = 0;
    }

    if (mask) {
        /* iterator, data pointer and type of mask array: */
        // 如果有掩码数组，则初始化掩码数组的迭代器
        if (!NI_InitPointIterator(mask, &mi))
            return 0;
        pm = (void *)PyArray_DATA(mask);

        size = PyArray_SIZE(array);

        for(jj = 0; jj < size; jj++) {
            if (*(npy_int8*)pm) {
                *(npy_int8*)pm = -1;
            } else {
                *(npy_int8*)pm = (npy_int8)*(npy_bool*)pi;
                *(npy_bool*)pi = _false;
            }
            NI_ITERATOR_NEXT2(ii, mi,  pi, pm)
        }
        NI_ITERATOR_RESET(ii)
        pi = (void *)PyArray_DATA(array);
    }

    // 初始化两个坐标列表
    list1 = NI_InitCoordinateList((*iclist)->block_size, (*iclist)->rank);
    list2 = NI_InitCoordinateList((*iclist)->block_size, (*iclist)->rank);
    if (!list1 || !list2) {
        PyErr_NoMemory();
        goto exit;
    }
    // 将iclist的块复制到list2中
    if (NI_CoordinateListStealBlocks(list2, *iclist))
        goto exit;

    NPY_BEGIN_THREADS;

    block2 = list2->blocks;
    jj = 0; // 重置jj计数器
    }

    if (mask) {
        NI_ITERATOR_RESET(ii)
        NI_ITERATOR_RESET(mi)
        pi = (void *)PyArray_DATA(array);
        pm = (void *)PyArray_DATA(mask);
        for(jj = 0; jj < size; jj++) {
            int value = *(npy_int8*)pm;
            if (value >= 0)
                *(npy_bool*)pi = value;
            NI_ITERATOR_NEXT2(ii, mi,  pi, pm)
        }
    }

 exit:
    NPY_END_THREADS;
    free(offsets); // 释放偏移量数组
    free(coordinate_offsets); // 释放坐标偏移量数组
    NI_FreeCoordinateList(list1); // 释放坐标列表
    # 释放 list2 所占用的内存空间
    NI_FreeCoordinateList(list2);
    # 检查是否发生了 Python 异常，如果有则返回 0，否则返回 1
    return PyErr_Occurred() ? 0 : 1;
}



#define NI_DISTANCE_EUCLIDIAN  1
#define NI_DISTANCE_CITY_BLOCK 2
#define NI_DISTANCE_CHESSBOARD 3



typedef struct {
    npy_intp *coordinates;  // 保存坐标数组的指针
    npy_intp index;         // 元素索引
    void *next;             // 指向下一个元素的指针
} NI_BorderElement;



int NI_DistanceTransformBruteForce(PyArrayObject* input, int metric,
                                   PyArrayObject *sampling_arr,
                                   PyArrayObject* distances,
                                   PyArrayObject* features)
{
    npy_intp size, jj, min_index = 0;
    int kk;
    NI_BorderElement *border_elements = NULL, *temp;
    NI_Iterator ii, di, fi;  // 迭代器变量
    char *pi, *pd = NULL, *pf = NULL;  // 指向输入、距离、特征数据的指针
    npy_double *sampling = sampling_arr ? (void *)PyArray_DATA(sampling_arr) : NULL;  // 如果存在采样数组，则获取其数据指针
    NPY_BEGIN_THREADS_DEF;  // 定义多线程开始宏



    /* check the output arrays: */
    if (distances) {
        pd = (void *)PyArray_DATA(distances);  // 如果距离数组不为空，则获取其数据指针
        if (!NI_InitPointIterator(distances, &di))  // 如果初始化距离数组的迭代器失败，则跳转到退出标签
            goto exit;
    }

    if (features) {
        pf = (void *)PyArray_DATA(features);  // 如果特征数组不为空，则获取其数据指针
        if (!NI_InitPointIterator(features, &fi))  // 如果初始化特征数组的迭代器失败，则跳转到退出标签
            goto exit;
    }



    size = PyArray_SIZE(input);  // 获取输入数组的大小
    pi = (void *)PyArray_DATA(input);  // 获取输入数组的数据指针

    if (!NI_InitPointIterator(input, &ii))  // 如果初始化输入数组的迭代器失败，则跳转到退出标签
        goto exit;

    for(jj = 0; jj < size; jj++) {
        if (*(npy_int8*)pi < 0) {  // 如果输入数组当前元素小于0
            temp = malloc(sizeof(NI_BorderElement));  // 分配边界元素结构体的内存
            if (NPY_UNLIKELY(!temp)) {  // 如果分配失败，则抛出内存错误并跳转到退出标签
                PyErr_NoMemory();
                goto exit;
            }
            temp->next = border_elements;  // 将新分配的元素作为链表头部
            border_elements = temp;
            temp->index = jj;  // 设置元素的索引
            temp->coordinates = malloc(PyArray_NDIM(input) * sizeof(npy_intp));  // 分配坐标数组的内存
            if (!temp->coordinates) {  // 如果分配失败，则抛出内存错误并跳转到退出标签
                PyErr_NoMemory();
                goto exit;
            }
            for (kk = 0; kk < PyArray_NDIM(input); kk++) {
                temp->coordinates[kk] = ii.coordinates[kk];  // 复制当前迭代器的坐标到新元素的坐标数组中
            }
        }
        NI_ITERATOR_NEXT(ii, pi);  // 移动迭代器到下一个位置
    }



    NPY_BEGIN_THREADS;  // 启动多线程处理

    NI_ITERATOR_RESET(ii);  // 重置输入数组的迭代器
    pi = (void *)PyArray_DATA(input);  // 获取输入数组的数据指针



    switch(metric) {  // 根据指定的度量标准进行分支处理


注：这些注释依据代码的逻辑结构和函数名称提供了对每行代码的简洁解释，不涉及代码整体含义或功能的总结。
    # 当距离度量为欧氏距离时执行以下逻辑
    case NI_DISTANCE_EUCLIDIAN:
        # 遍历输入数据的每个元素
        for(jj = 0; jj < size; jj++) {
            # 如果当前元素大于零
            if (*(npy_int8*)pi > 0) {
                # 初始化距离为双精度浮点数的最大值
                double distance = DBL_MAX;
                # 临时指针指向边界元素链表的头部
                temp = border_elements;
                # 遍历边界元素链表
                while(temp) {
                    # 初始化距离计算变量为0.0
                    double d = 0.0, t;
                    # 遍历输入数据的维度
                    for (kk = 0; kk < PyArray_NDIM(input); kk++) {
                        # 计算当前维度下的坐标差值
                        t = ii.coordinates[kk] - temp->coordinates[kk];
                        # 如果有采样率，乘以对应维度的采样率
                        if (sampling)
                            t *= sampling[kk];
                        # 计算欧氏距离的平方
                        d += t * t;
                    }
                    # 如果计算得到的距离小于当前最小距离，则更新最小距离和最小距离的索引
                    if (d < distance) {
                        distance = d;
                        if (features)
                            min_index = temp->index;
                    }
                    # 移动到链表的下一个元素
                    temp = temp->next;
                }
                # 如果需要计算距离，将距离的平方根存入目标数组
                if (distances)
                    *(npy_double*)pd = sqrt(distance);
                # 如果需要记录最小距离对应的索引，将其存入目标数组
                if (features)
                    *(npy_int32*)pf = min_index;
            } else {
                # 如果当前元素不大于零，则距离设置为0.0
                if (distances)
                    *(npy_double*)pd = 0.0;
                # 如果需要记录特征值，将当前索引存入目标数组
                if (features)
                    *(npy_int32*)pf = jj;
            }
            # 如果需要同时记录特征值和距离，移动到下一个元素位置
            if (features && distances) {
                NI_ITERATOR_NEXT3(ii, di, fi, pi, pd, pf);
            } else if (distances) {
                NI_ITERATOR_NEXT2(ii, di, pi, pd);
            } else {
                NI_ITERATOR_NEXT2(ii, fi, pi, pf);
            }
        }
        # 结束当前 case 分支的执行
        break;
    # 当距离度量为曼哈顿距离时
    case NI_DISTANCE_CITY_BLOCK:
    # 如果使用的距离度量是“棋盘距离”
    case NI_DISTANCE_CHESSBOARD:
        # 遍历所有的像素点
        for(jj = 0; jj < size; jj++) {
            # 如果当前像素点的值大于0
            if (*(npy_int8*)pi > 0) {
                # 初始化最大距离
                unsigned int distance = UINT_MAX;
                # 临时指针指向边界元素
                temp = border_elements;
                # 遍历所有的边界元素
                while(temp) {
                    # 初始化当前距离为0
                    unsigned int d = 0;
                    npy_intp t;
                    # 对每一个维度进行计算
                    for (kk = 0; kk < PyArray_NDIM(input); kk++) {
                        # 计算当前维度上的距离差
                        t = ii.coordinates[kk] - temp->coordinates[kk];
                        if (t < 0)
                            t = -t;
                        # 如果使用的距离度量是“曼哈顿距离”
                        if (metric == NI_DISTANCE_CITY_BLOCK) {
                            # 累加距离
                            d += t;
                        } else {
                            # 否则，更新最大距离
                            if ((unsigned int)t > d)
                                d = t;
                        }
                    }
                    # 如果当前距离小于已记录的最小距离
                    if (d < distance) {
                        # 更新最小距离
                        distance = d;
                        # 如果需要记录特征
                        if (features)
                            # 更新最小距离对应的特征索引
                            min_index = temp->index;
                    }
                    # 移动到下一个边界元素
                    temp = temp->next;
                }
                # 如果需要记录距离
                if (distances)
                    # 将计算出的最小距离存储到输出中
                    *(npy_uint32*)pd = distance;
                # 如果需要记录特征
                if (features)
                    # 将最小距离对应的特征索引存储到输出中
                    *(npy_int32*)pf = min_index;
            } else {
                # 如果当前像素点的值不大于0
                # 如果需要记录距离
                if (distances)
                    # 将距离设置为0
                    *(npy_uint32*)pd = 0;
                # 如果需要记录特征
                if (features)
                    # 将特征索引设置为当前像素点的索引
                    *(npy_int32*)pf = jj;
            }
            # 如果同时需要记录特征和距离
            if (features && distances) {
                # 将迭代器向前移动到下一个像素点
                NI_ITERATOR_NEXT3(ii, di, fi, pi, pd, pf);
            } else if (distances) {
                # 将迭代器向前移动到下一个像素点
                NI_ITERATOR_NEXT2(ii, di, pi, pd);
            } else {
                # 将迭代器向前移动到下一个像素点
                NI_ITERATOR_NEXT2(ii, fi, pi, pf);
            }
        }
        break;
    # 如果距离度量类型不支持
    default:
        # 结束多线程处理
        NPY_END_THREADS;
        # 设置运行时错误信息
        PyErr_SetString(PyExc_RuntimeError, "distance metric not supported");
        # 跳转到函数退出处理步骤
        goto exit;
    }

 exit:
    # 结束多线程处理
    NPY_END_THREADS;
    # 释放边界元素链表的内存
    while (border_elements) {
        temp = border_elements;
        border_elements = border_elements->next;
        free(temp->coordinates);
        free(temp);
    }
    # 检查是否发生了异常，返回相应的结果
    return PyErr_Occurred() ? 0 : 1;
    }
    
    /* 计算距离变换的单次迭代过程 */

    int NI_DistanceTransformOnePass(PyArrayObject *strct,
                                    PyArrayObject* distances,
                                    PyArrayObject *features)
    {
        npy_intp jj, ii, ssize, size, filter_size, mask_value, *oo;
        npy_intp *foffsets = NULL, *foo = NULL, *offsets = NULL;
        npy_bool *ps, *pf = NULL, *footprint = NULL;
        char *pd;
        NI_FilterIterator si, ti;
        NI_Iterator di, fi;
        NPY_BEGIN_THREADS_DEF;

        ssize = PyArray_SIZE(strct);

        /* 我们只使用结构数据的前半部分，因此为过滤函数创建临时结构： */
        footprint = malloc(ssize * sizeof(npy_bool));
        if (!footprint) {
            PyErr_NoMemory();
            goto exit;
        }
        ps = (npy_bool*)PyArray_DATA(strct);
        filter_size = 0;
        for(jj = 0; jj < ssize / 2; jj++) {
            footprint[jj] = ps[jj];
            if (ps[jj])
                ++filter_size;
        }
        for(jj = ssize / 2; jj < ssize; jj++)
            footprint[jj] = 0;
        /* 获取数据和大小 */
        pd = (void *)PyArray_DATA(distances);
        size = PyArray_SIZE(distances);
        if (!NI_InitPointIterator(distances, &di))
            goto exit;
        /* 计算过滤器偏移量： */
        if (!NI_InitFilterOffsets(distances, footprint, PyArray_DIMS(strct), NULL,
                                  NI_EXTEND_CONSTANT, &offsets, &mask_value,
                                  NULL)) {
            goto exit;
        }
        /* 初始化过滤器迭代器： */
        if (!NI_InitFilterIterator(PyArray_NDIM(distances), PyArray_DIMS(strct),
                                   filter_size, PyArray_DIMS(distances), NULL,
                                   &si)) {
            goto exit;
        }

        if (features) {
            npy_intp dummy;
            /* 初始化点迭代器： */
            pf = (void *)PyArray_DATA(features);
            if (!NI_InitPointIterator(features, &fi))
                goto exit;
            /* 计算过滤器偏移量： */
            if (!NI_InitFilterOffsets(features, footprint, PyArray_DIMS(strct),
                                      NULL, NI_EXTEND_CONSTANT, &foffsets, &dummy,
                                      NULL)) {
                goto exit;
            }
            /* 初始化过滤器迭代器： */
            if (!NI_InitFilterIterator(PyArray_NDIM(distances),
                                       PyArray_DIMS(strct), filter_size,
                                       PyArray_DIMS(distances), NULL, &ti)) {
                goto exit;
            }
        }

        NPY_BEGIN_THREADS;
        /* 迭代元素： */
        oo = offsets;
        if (features)
            foo = foffsets;
    // 迭代处理输入数据的每个元素
    for(jj = 0; jj < size; jj++) {
        // 从指针 pd 中读取一个 npy_int32 类型的值
        npy_int32 value = *(npy_int32*)pd;
        // 如果值不为零，则执行以下操作
        if (value != 0) {
            // 将当前值赋给最小值 min，并初始化最小偏移 min_offset
            npy_int32 min = value;
            npy_intp min_offset = 0;
            /* 迭代处理结构元素: */
            // 遍历给定的结构元素数组
            for(ii = 0; ii < filter_size; ii++) {
                // 从偏移数组 oo 中获取当前偏移值
                npy_intp offset = oo[ii];
                npy_int32 tt = -1;
                // 如果偏移值小于 mask_value，则从指针 pd 加上偏移读取一个 npy_int32 类型的值给 tt
                if (offset < mask_value)
                    tt = *(npy_int32*)(pd + offset);
                // 如果 tt 大于等于 0，则执行以下操作
                if (tt >= 0) {
                    // 如果 min 小于 0 或者 tt + 1 小于 min，则更新 min 和 min_offset
                    if ((min < 0) || (tt + 1 < min)) {
                        min = tt + 1;
                        // 如果 features 为真，则更新 min_offset
                        if (features)
                            min_offset = foo[ii];
                    }
                }
            }
            // 将最小值 min 存回指针 pd 所指位置
            *(npy_int32*)pd = min;
            // 如果 features 为真，则将 pf 指针加上 min_offset 的值存回 pf 所指位置
            if (features)
                *(npy_int32*)pf = *(npy_int32*)(pf + min_offset);
        }
        // 如果 features 为真，则执行下一步过滤操作 NI_FILTER_NEXT
        if (features) {
            NI_FILTER_NEXT(ti, fi, foo, pf);
        }
        // 执行下一步过滤操作 NI_FILTER_NEXT
        NI_FILTER_NEXT(si, di, oo, pd);
    }

 exit:
    // 结束线程安全操作
    NPY_END_THREADS;
    // 释放动态分配的内存
    free(offsets);
    free(foffsets);
    free(footprint);
    // 如果发生 Python 异常，则返回 0，否则返回 1
    return PyErr_Occurred() ? 0 : 1;
static void _VoronoiFT(char *pf, npy_intp len, npy_intp *coor, int rank,
                       int d, npy_intp stride, npy_intp cstride,
                       npy_intp **f, npy_intp *g, const npy_double *sampling)
{
    npy_intp l = -1, ii, maxl, idx1, idx2;  // 初始化变量 l, ii, maxl, idx1, idx2
    npy_intp jj;  // 初始化变量 jj

    for(ii = 0; ii < len; ii++)  // 外层循环，遍历 len 次，ii 从 0 到 len-1
        for(jj = 0; jj < rank; jj++)  // 内层循环，遍历 rank 次，jj 从 0 到 rank-1
            // 将二维数组 f 的元素赋值为指定地址处的值
            f[ii][jj] = *(npy_int32*)(pf + ii * stride + cstride * jj);

    for(ii = 0; ii < len; ii++) {
        if (*(npy_int32*)(pf + ii * stride) >= 0) {  // 如果 pf 中特定位置的值大于等于 0
            double fd = f[ii][d];  // 获取 f[ii][d] 的值
            double wR = 0.0;  // 初始化 wR

            for(jj = 0; jj < rank; jj++) {
                if (jj != d) {  // 如果 jj 不等于 d
                    double tw = f[ii][jj] - coor[jj];  // 计算差值 tw
                    if (sampling)
                        tw *= sampling[jj];  // 如果有采样值，对 tw 进行调整
                    wR += tw * tw;  // 计算 wR
                }
            }

            while(l >= 1) {  // while 循环，直到 l 大于等于 1 时退出
                double a, b, c, uR = 0.0, vR = 0.0, f1;
                idx1 = g[l];  // 获取 g[l] 的值
                f1 = f[idx1][d];  // 获取 f[idx1][d] 的值
                idx2 = g[l - 1];  // 获取 g[l-1] 的值
                a = f1 - f[idx2][d];  // 计算 a
                b = fd - f1;  // 计算 b
                if (sampling) {
                    a *= sampling[d];  // 如果有采样值，对 a 和 b 进行调整
                    b *= sampling[d];
                }
                c = a + b;  // 计算 c

                for(jj = 0; jj < rank; jj++) {
                    if (jj != d) {
                        double cc = coor[jj];
                        double tu = f[idx2][jj] - cc;
                        double tv = f[idx1][jj] - cc;
                        if (sampling) {
                            tu *= sampling[jj];
                            tv *= sampling[jj];
                        }
                        uR += tu * tu;  // 计算 uR
                        vR += tv * tv;  // 计算 vR
                    }
                }

                if (c * vR - b * uR - a * wR - a * b * c <= 0.0)  // 如果条件成立
                    break;  // 跳出循环

                --l;  // l 自减
            }

            ++l;  // l 自增
            g[l] = ii;  // 将 ii 赋值给 g[l]
        }
    }

    maxl = l;  // 将 l 的值赋给 maxl
    if (maxl >= 0) {  // 如果 maxl 大于等于 0
        l = 0;  // 初始化 l

        for (ii = 0; ii < len; ii++) {
            double delta1 = 0.0, t;  // 初始化 delta1 和 t

            for(jj = 0; jj < rank; jj++) {
                t = jj == d ? f[g[l]][jj] - ii : f[g[l]][jj] - coor[jj];  // 根据条件计算 t
                if (sampling)
                    t *= sampling[jj];  // 如果有采样值，对 t 进行调整
                delta1 += t * t;  // 计算 delta1
            }

            while (l < maxl) {  // while 循环，直到 l 小于 maxl 时退出
                double delta2 = 0.0;  // 初始化 delta2

                for(jj = 0; jj < rank; jj++) {
                    t = jj == d ? f[g[l + 1]][jj] - ii : f[g[l + 1]][jj] - coor[jj];  // 根据条件计算 t
                    if (sampling)
                        t *= sampling[jj];  // 如果有采样值，对 t 进行调整
                    delta2 += t * t;  // 计算 delta2
                }

                if (delta1 <= delta2)  // 如果条件成立
                    break;  // 跳出循环

                delta1 = delta2;  // 更新 delta1
                ++l;  // l 自增
            }

            idx1 = g[l];  // 获取 g[l] 的值

            for(jj = 0; jj < rank; jj++)
                *(npy_int32*)(pf + ii * stride + jj * cstride) = f[idx1][jj];  // 将 f[idx1][jj] 的值写入 pf 的指定位置
        }
    }
}
/* 
   计算精确的欧几里得特征变换，参考文献：
   C. R. Maurer, Jr., R. Qi, V. Raghavan, "A linear time algorithm for computing
   exact euclidean distance transforms of binary images in arbitrary
   dimensions. IEEE Trans. PAMI 25, 265-270, 2003.
*/

static void _ComputeFT(char *pi, char *pf, npy_intp *ishape,
                       const npy_intp *istrides, const npy_intp *fstrides,
                       int rank, int d, npy_intp *coor, npy_intp **f,
                       npy_intp *g, PyArrayObject *features,
                       const npy_double *sampling)
{
    npy_intp kk;  // 定义循环计数器 kk
    npy_intp jj;  // 定义循环计数器 jj

    if (d == 0) {  // 如果 d 等于 0
        char *tf1 = pf;  // 初始化指针 tf1 指向 pf
        for(jj = 0; jj < ishape[0]; jj++) {  // 循环遍历 ishape 的第一维度
            if (*(npy_int8*)pi) {  // 如果 pi 指向的值被解释为 npy_int8 后非零
                *(npy_int32*)tf1 = -1;  // 将 tf1 解释为 npy_int32，并赋值为 -1
            } else {
                char *tf2 = tf1;  // 初始化指针 tf2 指向 tf1
                *(npy_int32*)tf2 = jj;  // 将 tf2 解释为 npy_int32，并赋值为 jj
                for(kk = 1; kk < rank; kk++) {  // 循环遍历 rank - 1 次
                    tf2 += fstrides[0];  // tf2 偏移 fstrides[0] 的字节数
                    *(npy_int32*)tf2 = coor[kk];  // 将 tf2 解释为 npy_int32，并赋值为 coor[kk]
                }
            }
            pi += istrides[0];  // pi 偏移 istrides[0] 的字节数
            tf1 += fstrides[1];  // tf1 偏移 fstrides[1] 的字节数
        }
        _VoronoiFT(pf, ishape[0], coor, rank, 0, fstrides[1], fstrides[0], f,
                   g, sampling);  // 调用 _VoronoiFT 函数
    } else {
        npy_uint32 axes = 0;  // 定义 axes 变量，并初始化为 0
        char *tf = pf;  // 初始化指针 tf 指向 pf
        npy_intp size = 1;  // 定义 size 变量，并初始化为 1
        NI_Iterator ii;  // 声明 NI_Iterator 结构体变量 ii

        for(jj = 0; jj < ishape[d]; jj++) {  // 循环遍历 ishape 的第 d 维度
            coor[d] = jj;  // 将 coor[d] 赋值为 jj
            _ComputeFT(pi, tf, ishape, istrides, fstrides, rank, d - 1, coor, f,
                       g, features, sampling);  // 递归调用 _ComputeFT 函数
            pi += istrides[d];  // pi 偏移 istrides[d] 的字节数
            tf += fstrides[d + 1];  // tf 偏移 fstrides[d + 1] 的字节数
        }

        for(jj = 0; jj < d; jj++) {  // 循环遍历 d 次
            axes |= (npy_uint32)1 << (jj + 1);  // 更新 axes 的位运算值
            size *= ishape[jj];  // 计算 size 的乘积
        }
        NI_InitPointIterator(features, &ii);  // 初始化 features 的迭代器 ii
        NI_SubspaceIterator(&ii, axes);  // 使用 axes 参数创建子空间迭代器

        tf = pf;  // 指针 tf 指向 pf
        for(jj = 0; jj < size; jj++) {  // 循环遍历 size 次
            for(kk = 0; kk < d; kk++)  // 循环遍历 d 次
                coor[kk] = ii.coordinates[kk];  // 更新 coor[kk] 的值为 ii 的坐标值
            _VoronoiFT(tf, ishape[d], coor, rank, d, fstrides[d + 1],
                       fstrides[0], f, g, sampling);  // 调用 _VoronoiFT 函数
            NI_ITERATOR_NEXT(ii, tf);  // ii 向前移动一步，tf 指针自动偏移
        }
        for(kk = 0; kk < d; kk++)  // 循环遍历 d 次
            coor[kk] = 0;  // 将 coor[kk] 的值设为 0
    }
}
    // 分配指针数组 f，存储 mx 个指针，每个指针指向一个大小为 PyArray_NDIM(input) 的整型数组
    f = malloc(mx * sizeof(npy_intp*));
    // 分配大小为 mx 的整型数组 g，用于存储长度信息
    g = malloc(mx * sizeof(npy_intp));
    // 分配大小为 mx * PyArray_NDIM(input) 的整型数组 tmp，用于临时存储多维数组的索引
    tmp = malloc(mx * PyArray_NDIM(input) * sizeof(npy_intp));
    // 检查内存分配是否成功
    if (!f || !g || !tmp) {
        // 内存分配失败时，引发内存错误异常
        PyErr_NoMemory();
        // 跳转到退出标签 exit
        goto exit;
    }
    // 配置 f 数组，使其每个元素指向 tmp 数组的不同部分，以便于多维数组的索引访问
    for(jj = 0; jj < mx; jj++) {
        f[jj] = tmp + jj * PyArray_NDIM(input);
    }

    /* 第一次调用递归特征变换 */
    // 开始多线程区域
    NPY_BEGIN_THREADS;
    // 调用 _ComputeFT 函数进行特征变换计算
    _ComputeFT(pi, pf, PyArray_DIMS(input), PyArray_STRIDES(input),
               PyArray_STRIDES(features), PyArray_NDIM(input),
               PyArray_NDIM(input) - 1, coor, f, g, features, sampling);
    // 结束多线程区域
    NPY_END_THREADS;

 exit:
    // 释放动态分配的内存
    free(f);
    free(g);
    free(tmp);

    // 根据错误状态返回 0 或 1
    return PyErr_Occurred() ? 0 : 1;
}



# 这是一个单独的右大括号 '}'，用于结束一个代码块或数据结构的定义。
```