# `D:\src\scipysrc\scipy\scipy\ndimage\src\ni_support.h`

```
/*
 * 版权所有 (C) 2003-2005 Peter J. Verveer
 *
 * 在源代码和二进制形式下允许重新分发和使用，无论是否进行修改，只要遵循以下条件：
 *
 * 1. 源代码的重新分发必须保留上述版权声明、此条件列表以及以下免责声明。
 *
 * 2. 在二进制形式的重新分发中，必须在文档和/或提供的其他材料中重复上述版权声明、
 *    此条件列表以及以下免责声明。
 *
 * 3. 未经明确书面许可，不得使用作者的名称来认可或推广基于本软件的产品。
 *
 * 本软件由作者"按原样"提供，任何明示或默示的保证，包括但不限于适销性和特定用途的
 * 隐含保证，均被拒绝。无论出于何种原因，作者都不对任何直接、间接、偶然、特殊、
 * 惩罚性或后果性的损害负责，即使事先已被告知可能发生此类损害。
 */

#ifndef NI_SUPPORT_H
#define NI_SUPPORT_H

/*
 * NO_IMPORT_ARRAY 告诉 numpy 编译单元将重用另一个编译单元中初始化的 numpy API。
 * 调用 import_array() 初始化共享的 numpy API 的编译单元必须通过显式在 ni_support.h
 * 之前包含 nd_image.h 来绕过这一点。
 */
#define NO_IMPORT_ARRAY
#include "nd_image.h"
#undef NO_IMPORT_ARRAY

#include <stdlib.h>
#include <float.h>
#include <limits.h>
#include <assert.h>

/* 不同的边界条件。尽管 Python 代码中不使用镜像条件，但保留 C 代码以防可能添加它。 */
typedef enum {
    NI_EXTEND_FIRST = 0,
    NI_EXTEND_NEAREST = 0,
    NI_EXTEND_WRAP = 1,
    NI_EXTEND_REFLECT = 2,
    NI_EXTEND_MIRROR = 3,
    NI_EXTEND_CONSTANT = 4,
    NI_EXTEND_GRID_WRAP = 5,
    NI_EXTEND_GRID_CONSTANT = 6,
    NI_EXTEND_LAST = NI_EXTEND_GRID_WRAP,
    NI_EXTEND_DEFAULT = NI_EXTEND_MIRROR
} NI_ExtendMode;


/******************************************************************/
/* 迭代器 */
/******************************************************************/

/* 迭代器结构体：*/
typedef struct {
    int rank_m1;                // 数组维度减一
    npy_intp dimensions[NPY_MAXDIMS];   // 数组各维度大小
    npy_intp coordinates[NPY_MAXDIMS];  // 当前迭代器位置的坐标
    npy_intp strides[NPY_MAXDIMS];      // 各维度的步幅
    npy_intp backstrides[NPY_MAXDIMS];  // 逆向步幅
} NI_Iterator;

/* 初始化迭代器以遍历单个数组元素： */
int NI_InitPointIterator(PyArrayObject*, NI_Iterator*);
/* initialize iterations over an arbitrary sub-space: */
/* 初始化对任意子空间的迭代器： */
int NI_SubspaceIterator(NI_Iterator*, npy_uint32);

/* initialize iteration over array lines: */
/* 初始化对数组行的迭代器： */
int NI_LineIterator(NI_Iterator*, int);

/* reset an iterator */
/* 重置迭代器 */
#define NI_ITERATOR_RESET(iterator)              \
{                                                \
    int _ii;                                       \
    for(_ii = 0; _ii <= (iterator).rank_m1; _ii++) \
        (iterator).coordinates[_ii] = 0;             \
}

/* go to the next point in a single array */
/* 移动到单个数组中的下一个点 */
#define NI_ITERATOR_NEXT(iterator, pointer)                         \
{                                                                   \
    int _ii;                                                          \
    for(_ii = (iterator).rank_m1; _ii >= 0; _ii--)                    \
        if ((iterator).coordinates[_ii] < (iterator).dimensions[_ii]) { \
            (iterator).coordinates[_ii]++;                                \
            pointer += (iterator).strides[_ii];                           \
            break;                                                        \
        } else {                                                        \
            (iterator).coordinates[_ii] = 0;                              \
            pointer -= (iterator).backstrides[_ii];                       \
        }                                                               \
}

/* go to the next point in two arrays of the same size */
/* 移动到两个大小相同数组中的下一个点 */
#define NI_ITERATOR_NEXT2(iterator1, iterator2,  pointer1, pointer2)  \
{                                                                     \
    int _ii;                                                            \
    for(_ii = (iterator1).rank_m1; _ii >= 0; _ii--)                     \
        if ((iterator1).coordinates[_ii] < (iterator1).dimensions[_ii]) { \
            (iterator1).coordinates[_ii]++;                                 \
            pointer1 += (iterator1).strides[_ii];                           \
            pointer2 += (iterator2).strides[_ii];                           \
            break;                                                          \
        } else {                                                          \
            (iterator1).coordinates[_ii] = 0;                               \
            pointer1 -= (iterator1).backstrides[_ii];                       \
            pointer2 -= (iterator2).backstrides[_ii];                       \
        }                                                                 \
}

/* go to the next point in three arrays of the same size */
/* 移动到三个大小相同数组中的下一个点 */
#define NI_ITERATOR_NEXT3(iterator1, iterator2,  iterator3,           \
                                                    pointer1, pointer2, pointer3)               \
{                                                                     \
    int _ii;                                                            \
    // 从最高维度开始遍历迭代器1，直到最低维度
    for(_ii = (iterator1).rank_m1; _ii >= 0; _ii--)                     \
        // 如果当前维度的坐标小于该维度的尺寸
        if ((iterator1).coordinates[_ii] < (iterator1).dimensions[_ii]) { \
            // 增加当前维度的坐标
            (iterator1).coordinates[_ii]++;                                 \
            // 移动指针1、指针2、指针3，分别根据迭代器1、迭代器2、迭代器3的步幅
            pointer1 += (iterator1).strides[_ii];                           \
            pointer2 += (iterator2).strides[_ii];                           \
            pointer3 += (iterator3).strides[_ii];                           \
            // 跳出循环
            break;                                                          \
        } else {                                                          \
            // 如果当前维度的坐标已达到最大值，则重置为0
            (iterator1).coordinates[_ii] = 0;                               \
            // 向前移动指针1、指针2、指针3，根据迭代器1、迭代器2、迭代器3的反向步幅
            pointer1 -= (iterator1).backstrides[_ii];                       \
            pointer2 -= (iterator2).backstrides[_ii];                       \
            pointer3 -= (iterator3).backstrides[_ii];                       \
        }                                                                 \
/* 
   从一个单一数组的任意点开始遍历
   宏定义，用于将迭代器指向指定位置
*/
#define NI_ITERATOR_GOTO(iterator, destination, base, pointer) \
{                                                              \
    int _ii;                                                     \
    pointer = base;                                              \
    // 逆向遍历迭代器的秩，计算指针的新位置
    for(_ii = (iterator).rank_m1; _ii >= 0; _ii--) {             \
        pointer += destination[_ii] * (iterator).strides[_ii];     \
        // 更新迭代器的坐标为目标位置的坐标
        (iterator).coordinates[_ii] = destination[_ii];            \
    }                                                            \
}

/******************************************************************/
/* Line buffers */
/******************************************************************/

/* 线缓冲区结构体： */
typedef struct {
    double *buffer_data;
    npy_intp buffer_lines, line_length, line_stride;
    npy_intp size1, size2, array_lines, next_line;
    NI_Iterator iterator;
    char* array_data;
    enum NPY_TYPES array_type;
    NI_ExtendMode extend_mode;
    double extend_value;
} NI_LineBuffer;

/* 获取正在处理的下一行： */
#define NI_GET_LINE(_buffer, _line)                                      \
    ((_buffer).buffer_data + (_line) * ((_buffer).line_length +            \
                                        (_buffer).size1 + (_buffer).size2))

/* 分配线缓冲区数据的内存空间 */
int NI_AllocateLineBuffer(PyArrayObject*, int, npy_intp, npy_intp,
                           npy_intp*, npy_intp, double**);

/* 初始化线缓冲区 */
int NI_InitLineBuffer(PyArrayObject*, int, npy_intp, npy_intp, npy_intp,
                      double*, NI_ExtendMode, double, NI_LineBuffer*);

/* 扩展内存中的一行，以实现边界条件： */
int NI_ExtendLine(double*, npy_intp, npy_intp, npy_intp, NI_ExtendMode, double);

/* 将数组中的一行复制到缓冲区： */
int NI_ArrayToLineBuffer(NI_LineBuffer*, npy_intp*, int*);

/* 将缓冲区中的一行复制回数组： */
int NI_LineBufferToArray(NI_LineBuffer*);

/******************************************************************/
/* Multi-dimensional filter support functions */
/******************************************************************/

/* 过滤器迭代器结构体： */
typedef struct {
    npy_intp strides[NPY_MAXDIMS], backstrides[NPY_MAXDIMS];
    npy_intp bound1[NPY_MAXDIMS], bound2[NPY_MAXDIMS];
} NI_FilterIterator;

/* 初始化过滤器迭代器： */
int NI_InitFilterIterator(int, npy_intp*, npy_intp, npy_intp*,
                          npy_intp*, NI_FilterIterator*);

/* 计算过滤器点的偏移量，用于数组的所有边界区域和内部区域： */
int NI_InitFilterOffsets(PyArrayObject*, npy_bool*, npy_intp*,
                         npy_intp*, NI_ExtendMode, npy_intp**,
                         npy_intp*, npy_intp**);
/* Move to the next point in two arrays, possibly adjusting pointers based on boundary conditions */

#define NI_FILTER_NEXT(iteratorf, iterator1, pointerf, pointer1)  \
{                                                                 \
    int _ii;                                                        \
    for(_ii = (iterator1).rank_m1; _ii >= 0; _ii--) {               \
        npy_intp _pp = (iterator1).coordinates[_ii];              \
        // Check if current coordinate exceeds dimensions; adjust if necessary
        if (_pp < (iterator1).dimensions[_ii]) {                      \
            // Check if current coordinate is out of filter bounds
            if (_pp < (iteratorf).bound1[_ii] ||                        \
                _pp >= (iteratorf).bound2[_ii])                        \
                pointerf += (iteratorf).strides[_ii];                     \
            // Move to the next coordinate in iterator1
            (iterator1).coordinates[_ii]++;                             \
            pointer1 += (iterator1).strides[_ii];                       \
            break;                                                      \
        } else {                                                      \
            // Reset coordinate and adjust pointers when exceeding dimensions
            (iterator1).coordinates[_ii] = 0;                           \
            pointer1 -= (iterator1).backstrides[_ii];                   \
            pointerf -= (iteratorf).backstrides[_ii];                   \
        }                                                             \
    }                                                               \
}

/* Move to the next point in three arrays, adjusting pointers based on boundary conditions */

#define NI_FILTER_NEXT2(iteratorf, iterator1, iterator2,    \
                                                pointerf, pointer1, pointer2)       \
{                                                           \
    int _ii;                                                  \
    for(_ii = (iterator1).rank_m1; _ii >= 0; _ii--) {         \
        npy_intp _pp = (iterator1).coordinates[_ii];        \
        // Check if current coordinate exceeds dimensions; adjust if necessary
        if (_pp < (iterator1).dimensions[_ii]) {                \
            // Check if current coordinate is out of filter bounds
            if (_pp < (iteratorf).bound1[_ii] ||                  \
                _pp >= (iteratorf).bound2[_ii])                        \
                pointerf += (iteratorf).strides[_ii];               \
            // Move to the next coordinate in iterator1 and adjust pointers for iterator2
            (iterator1).coordinates[_ii]++;                       \
            pointer1 += (iterator1).strides[_ii];                 \
            pointer2 += (iterator2).strides[_ii];                 \
            break;                                                \
        } else {                                                \
            // Reset coordinate and adjust pointers for iterator1 and iterator2 when exceeding dimensions
            (iterator1).coordinates[_ii] = 0;                     \
            pointer1 -= (iterator1).backstrides[_ii];             \
            pointer2 -= (iterator2).backstrides[_ii];             \
            pointerf -= (iteratorf).backstrides[_ii];             \
        }                                                       \
    }                                                           \
}
    }  # 这是一个代码块的结束标记，表示一个代码块的闭合
/* 
   将三个迭代器中的指针移动到下一个位置，可能会在进入数组的不同区域时改变到滤波器偏移量的指针：
   - iteratorf: 滤波器迭代器
   - iterator1, iterator2, iterator3: 三个数据数组的迭代器
   - pointerf, pointer1, pointer2, pointer3: 指向相应数组数据的指针
*/
#define NI_FILTER_NEXT3(iteratorf, iterator1, iterator2, iterator3,  \
                                                pointerf, pointer1, pointer2, pointer3)      \
{                                                                    \
    int _ii;                                                           \
    for(_ii = (iterator1).rank_m1; _ii >= 0; _ii--) {                  \
        npy_intp _pp = (iterator1).coordinates[_ii];                 \
        if (_pp < (iterator1).dimensions[_ii]) {                         \
            if (_pp < (iteratorf).bound1[_ii] ||                           \
                                                                         _pp >= (iteratorf).bound2[_ii]) \
                pointerf += (iteratorf).strides[_ii];                        \
            (iterator1).coordinates[_ii]++;                                \
            pointer1 += (iterator1).strides[_ii];                          \
            pointer2 += (iterator2).strides[_ii];                          \
            pointer3 += (iterator3).strides[_ii];                          \
            break;                                                         \
        } else {                                                         \
            (iterator1).coordinates[_ii] = 0;                              \
            pointer1 -= (iterator1).backstrides[_ii];                      \
            pointer2 -= (iterator2).backstrides[_ii];                      \
            pointer3 -= (iterator3).backstrides[_ii];                      \
            pointerf -= (iteratorf).backstrides[_ii];                      \
        }                                                                \
    }                                                                  \
}

/* 
   根据给定的坐标移动到滤波器偏移量的指针：
   - iteratorf: 滤波器迭代器
   - iterator: 数据数组的迭代器
   - fbase: 滤波器偏移量的基础指针
   - pointerf: 指向滤波器偏移量的指针，将被移动到给定坐标的位置
*/
#define NI_FILTER_GOTO(iteratorf, iterator, fbase, pointerf) \
{                                                            \
    int _ii;                                                   \
    npy_intp _jj;                                             \
    pointerf = fbase;                                          \
    for(_ii = iterator.rank_m1; _ii >= 0; _ii--) {             \
        # 从最高维度开始向最低维度迭代，_ii 是迭代器的当前维度索引
        npy_intp _pp = iterator.coordinates[_ii];             \
        # 获取当前维度上的坐标值 _pp
        npy_intp b1 = (iteratorf).bound1[_ii];                \
        # 获取迭代器在当前维度上的 lower bound b1
        npy_intp b2 = (iteratorf).bound2[_ii];                \
        # 获取迭代器在当前维度上的 upper bound b2
        if (_pp < b1) {                                          \
                # 如果 _pp 小于 b1，则 _jj 被设置为 _pp
                _jj = _pp;                                           \
        } else if (_pp > b2 && b2 >= b1) {                       \
                # 如果 _pp 大于 b2 且 b2 大于等于 b1，则 _jj 被设置为 _pp + b1 - b2
                _jj = _pp + b1 - b2;                                 \
        } else {                                                 \
                # 否则，_jj 被设置为 b1
                _jj = b1;                                            \
        }                                                        \
        pointerf += (iteratorf).strides[_ii] * _jj;              \
        # 根据当前维度上的步长 strides[_ii] 和 _jj 更新指针位置 pointerf
    }                                                          \
# 结构体定义：表示一块坐标数据的结构体
typedef struct {
    # 指向坐标数组的指针
    npy_intp *coordinates;
    # 坐标数组的大小
    int size;
    # 指向下一个坐标块的指针
    void *next;
} NI_CoordinateBlock;

# 结构体定义：表示一系列坐标块的列表结构
typedef struct {
    # 坐标块的大小
    int block_size;
    # 坐标的维度
    int rank;
    # 指向坐标块列表的指针
    void *blocks;
} NI_CoordinateList;

# 函数声明：初始化坐标列表结构
NI_CoordinateList* NI_InitCoordinateList(int, int);

# 函数声明：将坐标列表结构中的坐标块转移到另一个坐标列表结构中
int NI_CoordinateListStealBlocks(NI_CoordinateList*, NI_CoordinateList*);

# 函数声明：在坐标列表结构中添加一个新的坐标块
NI_CoordinateBlock* NI_CoordinateListAddBlock(NI_CoordinateList*);

# 函数声明：从坐标列表结构中删除一个坐标块
NI_CoordinateBlock* NI_CoordinateListDeleteBlock(NI_CoordinateList*);

# 函数声明：释放坐标列表结构及其包含的所有资源
void NI_FreeCoordinateList(NI_CoordinateList*);

#endif
```