# `.\numpy\numpy\_core\src\npysort\mergesort.cpp`

```py
/*
 * The purpose of this module is to add faster sort functions
 * that are type-specific.  This is done by altering the
 * function table for the builtin descriptors.
 *
 * These sorting functions are copied almost directly from numarray
 * with a few modifications (complex comparisons compare the imaginary
 * part if the real parts are equal, for example), and the names
 * are changed.
 *
 * The original sorting code is due to Charles R. Harris who wrote
 * it for numarray.
 */

/*
 * Quick sort is usually the fastest, but the worst case scenario can
 * be slower than the merge and heap sorts.  The merge sort requires
 * extra memory and so for large arrays may not be useful.
 *
 * The merge sort is *stable*, meaning that equal components
 * are unmoved from their entry versions, so it can be used to
 * implement lexicographic sorting on multiple keys.
 *
 * The heap sort is included for completeness.
 */

#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include "npy_sort.h"               // Include header for sorting functions
#include "npysort_common.h"         // Include header for common sorting utilities
#include "numpy_tag.h"              // Include header for numpy tags (type descriptors)

#include <cstdlib>                  // Standard library for memory allocation

#define NOT_USED NPY_UNUSED(unused) // Macro to indicate unused variables
#define PYA_QS_STACK 100            // Constant for quicksort stack size
#define SMALL_QUICKSORT 15          // Threshold for small quicksort
#define SMALL_MERGESORT 20          // Threshold for small mergesort
#define SMALL_STRING 16             // Threshold for small string sorting

/*
 *****************************************************************************
 **                            NUMERIC SORTS                                **
 *****************************************************************************
 */

template <typename Tag, typename type>
static void
mergesort0_(type *pl, type *pr, type *pw)
{
    type vp, *pi, *pj, *pk, *pm;

    if (pr - pl > SMALL_MERGESORT) {
        /* merge sort */
        pm = pl + ((pr - pl) >> 1);                 // Calculate middle point
        mergesort0_<Tag>(pl, pm, pw);               // Recursively sort left half
        mergesort0_<Tag>(pm, pr, pw);               // Recursively sort right half
        for (pi = pw, pj = pl; pj < pm;) {          // Copy elements to workspace
            *pi++ = *pj++;
        }
        pi = pw + (pm - pl);                        // Set insertion point in workspace
        pj = pw;
        pk = pl;
        while (pj < pi && pm < pr) {                // Merge sorted halves
            if (Tag::less(*pm, *pj)) {              // Compare using custom tag less function
                *pk++ = *pm++;
            }
            else {
                *pk++ = *pj++;
            }
        }
        while (pj < pi) {                           // Copy remaining elements from workspace
            *pk++ = *pj++;
        }
    }
    else {
        /* insertion sort */
        for (pi = pl + 1; pi < pr; ++pi) {          // Iterate through elements
            vp = *pi;
            pj = pi;
            pk = pi - 1;
            while (pj > pl && Tag::less(vp, *pk)) { // Perform insertion sort
                *pj-- = *pk--;
            }
            *pj = vp;
        }
    }
}

template <typename Tag, typename type>
NPY_NO_EXPORT int
mergesort_(type *start, npy_intp num)
{
    type *pl, *pr, *pw;

    pl = start;
    pr = pl + num;
    pw = (type *)malloc((num / 2) * sizeof(type));  // Allocate workspace
    if (pw == NULL) {
        return -NPY_ENOMEM;                       // Return error if allocation fails
    }
    mergesort0_<Tag>(pl, pr, pw);                 // Call mergesort function

    free(pw);                                      // Free allocated workspace
    return 0;                                      // Return success
}

template <typename Tag, typename type>
static void
amergesort0_(npy_intp *pl, npy_intp *pr, type *v, npy_intp *pw)
{
    type vp;
    npy_intp vi, *pi, *pj, *pk, *pm;
    if (pr - pl > SMALL_MERGESORT) {
        /* 如果待排序的子数组长度大于预设的小数组长度，使用归并排序 */

        // 计算中间位置
        pm = pl + ((pr - pl) >> 1);

        // 对左半部分进行归并排序
        amergesort0_<Tag>(pl, pm, v, pw);

        // 对右半部分进行归并排序
        amergesort0_<Tag>(pm, pr, v, pw);

        // 合并两个有序子数组
        for (pi = pw, pj = pl; pj < pm;) {
            *pi++ = *pj++;
        }

        // 设置合并后的起始位置
        pi = pw + (pm - pl);
        pj = pw;
        pk = pl;

        // 合并左右两个子数组
        while (pj < pi && pm < pr) {
            if (Tag::less(v[*pm], v[*pj])) {
                *pk++ = *pm++;
            }
            else {
                *pk++ = *pj++;
            }
        }

        // 处理剩余的元素
        while (pj < pi) {
            *pk++ = *pj++;
        }
    }
    else {
        /* 如果待排序的子数组长度不大于预设的小数组长度，使用插入排序 */

        // 插入排序
        for (pi = pl + 1; pi < pr; ++pi) {
            vi = *pi;
            vp = v[vi];
            pj = pi;
            pk = pi - 1;

            // 寻找合适的插入位置
            while (pj > pl && Tag::less(vp, v[*pk])) {
                *pj-- = *pk--;
            }

            // 插入元素
            *pj = vi;
        }
    }
}

template <typename Tag, typename type>
NPY_NO_EXPORT int
amergesort_(type *v, npy_intp *tosort, npy_intp num)
{
    npy_intp *pl, *pr, *pw;

    pl = tosort;                    // 初始化指向排序数组的起始位置
    pr = pl + num;                  // 初始化指向排序数组的结束位置的下一个位置
    pw = (npy_intp *)malloc((num / 2) * sizeof(npy_intp));   // 分配临时空间，用于归并排序中的工作数组
    if (pw == NULL) {               // 如果分配失败，则返回内存不足错误码
        return -NPY_ENOMEM;
    }
    amergesort0_<Tag>(pl, pr, v, pw);   // 调用归并排序的实现函数
    free(pw);                       // 释放临时工作数组的内存空间

    return 0;                       // 返回排序成功
}

/*
 
 *****************************************************************************
 **                             STRING SORTS                                **
 *****************************************************************************
 */

template <typename Tag, typename type>
static void
mergesort0_(type *pl, type *pr, type *pw, type *vp, size_t len)
{
    type *pi, *pj, *pk, *pm;

    if ((size_t)(pr - pl) > SMALL_MERGESORT * len) {   // 如果数组长度大于指定值，执行归并排序
        /* merge sort */
        pm = pl + (((pr - pl) / len) >> 1) * len;      // 计算中间位置并取整作为分割点
        mergesort0_<Tag>(pl, pm, pw, vp, len);         // 递归调用归并排序左半部分
        mergesort0_<Tag>(pm, pr, pw, vp, len);         // 递归调用归并排序右半部分
        Tag::copy(pw, pl, pm - pl);                   // 复制左半部分到临时工作数组
        pi = pw + (pm - pl);
        pj = pw;
        pk = pl;
        while (pj < pi && pm < pr) {                   // 归并两个有序数组
            if (Tag::less(pm, pj, len)) {              // 如果右半部分元素小于左半部分元素
                Tag::copy(pk, pm, len);                // 将右半部分元素复制到原数组
                pm += len;
                pk += len;
            }
            else {
                Tag::copy(pk, pj, len);                // 将左半部分元素复制到原数组
                pj += len;
                pk += len;
            }
        }
        Tag::copy(pk, pj, pi - pj);                    // 处理剩余的元素
    }
    else {
        /* insertion sort */                          // 如果数组长度小于等于指定值，执行插入排序
        for (pi = pl + len; pi < pr; pi += len) {      // 从第二个元素开始，依次将元素插入已排序的序列中
            Tag::copy(vp, pi, len);                   // 备份当前元素
            pj = pi;
            pk = pi - len;
            while (pj > pl && Tag::less(vp, pk, len)) {   // 向前比较并移动元素，保持序列有序
                Tag::copy(pj, pk, len);
                pj -= len;
                pk -= len;
            }
            Tag::copy(pj, vp, len);                   // 将备份的元素插入到正确的位置
        }
    }
}

template <typename Tag, typename type>
static int
string_mergesort_(type *start, npy_intp num, void *varr)
{
    PyArrayObject *arr = (PyArrayObject *)varr;
    size_t elsize = PyArray_ITEMSIZE(arr);
    size_t len = elsize / sizeof(type);
    type *pl, *pr, *pw, *vp;
    int err = 0;

    /* Items that have zero size don't make sense to sort */
    if (elsize == 0) {                              // 如果元素大小为0，直接返回排序成功
        return 0;
    }

    pl = start;                                      // 初始化指向排序数组的起始位置
    pr = pl + num * len;                             // 初始化指向排序数组的结束位置的下一个位置
    pw = (type *)malloc((num / 2) * elsize);          // 分配临时空间，用于归并排序中的工作数组
    if (pw == NULL) {                                // 如果分配失败，则返回内存不足错误码
        err = -NPY_ENOMEM;
        goto fail_0;
    }
    vp = (type *)malloc(elsize);                     // 分配临时空间，用于备份当前元素
    if (vp == NULL) {                                // 如果分配失败，则返回内存不足错误码
        err = -NPY_ENOMEM;
        goto fail_1;
    }
    mergesort0_<Tag>(pl, pr, pw, vp, len);           // 调用归并排序的实现函数

    free(vp);                                        // 释放临时备份元素的内存空间
fail_1:
    free(pw);                                        // 释放临时工作数组的内存空间
fail_0:
    return err;                                      // 返回排序的结果
}

template <typename Tag, typename type>
static void
amergesort0_(npy_intp *pl, npy_intp *pr, type *v, npy_intp *pw, size_t len)
{
    type *vp;
    npy_intp vi, *pi, *pj, *pk, *pm;
    if (pr - pl > SMALL_MERGESORT) {
        /* 如果子数组长度大于SMALL_MERGESORT，执行归并排序 */

        // 计算中间点
        pm = pl + ((pr - pl) >> 1);

        // 递归调用归并排序，对左半部分进行排序
        amergesort0_<Tag>(pl, pm, v, pw, len);
        
        // 递归调用归并排序，对右半部分进行排序
        amergesort0_<Tag>(pm, pr, v, pw, len);

        // 将左右两部分合并到临时数组pw中
        for (pi = pw, pj = pl; pj < pm;) {
            *pi++ = *pj++;
        }

        // 初始化指针pi指向合并后的起始位置
        pi = pw + (pm - pl);
        pj = pw;
        pk = pl;

        // 归并左右两部分，根据排序规则决定元素存放位置
        while (pj < pi && pm < pr) {
            if (Tag::less(v + (*pm) * len, v + (*pj) * len, len)) {
                *pk++ = *pm++;
            }
            else {
                *pk++ = *pj++;
            }
        }

        // 处理剩余元素
        while (pj < pi) {
            *pk++ = *pj++;
        }
    }
    else {
        /* 如果子数组长度不大于SMALL_MERGESORT，执行插入排序 */

        // 使用插入排序对当前子数组进行排序
        for (pi = pl + 1; pi < pr; ++pi) {
            vi = *pi;  // 当前待插入的值
            vp = v + vi * len;  // 对应的数据起始地址
            pj = pi;  // 待比较位置
            pk = pi - 1;  // 前一个位置

            // 向前比较并移动元素，直到找到插入位置
            while (pj > pl && Tag::less(vp, v + (*pk) * len, len)) {
                *pj-- = *pk--;
            }
            
            // 插入当前值到正确位置
            *pj = vi;
        }
    }


这段代码实现了一个通用的归并排序算法，其中根据子数组长度选择使用归并排序或插入排序。
}

template <typename Tag, typename type>
static int
string_amergesort_(type *v, npy_intp *tosort, npy_intp num, void *varr)
{
    PyArrayObject *arr = (PyArrayObject *)varr;  // 将void指针转换为PyArrayObject类型
    size_t elsize = PyArray_ITEMSIZE(arr);  // 计算数组元素的大小
    size_t len = elsize / sizeof(type);  // 计算数组中元素的个数
    npy_intp *pl, *pr, *pw;  // 定义指向npy_intp类型的指针

    /* Items that have zero size don't make sense to sort */
    if (elsize == 0) {
        return 0;  // 如果元素大小为0，返回0
    }

    pl = tosort;  // 初始化排序起始位置为tosort
    pr = pl + num;  // 初始化排序结束位置为tosort + num
    pw = (npy_intp *)malloc((num / 2) * sizeof(npy_intp));  // 分配内存给pw
    if (pw == NULL) {
        return -NPY_ENOMEM;  // 如果内存分配失败，返回内存不足错误码
    }
    amergesort0_<Tag>(pl, pr, v, pw, len);  // 调用模板函数进行排序
    free(pw);  // 释放pw指向的内存空间

    return 0;  // 返回0表示成功
}

/*
 *****************************************************************************
 **                             GENERIC SORT                                **
 *****************************************************************************
 */

static void
npy_mergesort0(char *pl, char *pr, char *pw, char *vp, npy_intp elsize,
               PyArray_CompareFunc *cmp, PyArrayObject *arr)
{
    char *pi, *pj, *pk, *pm;  // 定义指向字符的指针

    if (pr - pl > SMALL_MERGESORT * elsize) {
        /* merge sort */
        pm = pl + (((pr - pl) / elsize) >> 1) * elsize;  // 计算中间点pm
        npy_mergesort0(pl, pm, pw, vp, elsize, cmp, arr);  // 递归调用归并排序左半部分
        npy_mergesort0(pm, pr, pw, vp, elsize, cmp, arr);  // 递归调用归并排序右半部分
        GENERIC_COPY(pw, pl, pm - pl);  // 复制左半部分到pw中
        pi = pw + (pm - pl);  // 初始化pi为pw + (pm - pl)
        pj = pw;  // 初始化pj为pw
        pk = pl;  // 初始化pk为pl
        while (pj < pi && pm < pr) {
            if (cmp(pm, pj, arr) < 0) {  // 如果pm < pj，则复制pm到pk位置
                GENERIC_COPY(pk, pm, elsize);
                pm += elsize;
                pk += elsize;
            }
            else {  // 否则，复制pj到pk位置
                GENERIC_COPY(pk, pj, elsize);
                pj += elsize;
                pk += elsize;
            }
        }
        GENERIC_COPY(pk, pj, pi - pj);  // 复制剩余的元素到pk位置
    }
    else {
        /* insertion sort */
        for (pi = pl + elsize; pi < pr; pi += elsize) {  // 插入排序
            GENERIC_COPY(vp, pi, elsize);
            pj = pi;
            pk = pi - elsize;
            while (pj > pl && cmp(vp, pk, arr) < 0) {
                GENERIC_COPY(pj, pk, elsize);
                pj -= elsize;
                pk -= elsize;
            }
            GENERIC_COPY(pj, vp, elsize);
        }
    }
}

NPY_NO_EXPORT int
npy_mergesort(void *start, npy_intp num, void *varr)
{
    PyArrayObject *arr = (PyArrayObject *)varr;  // 将void指针转换为PyArrayObject类型
    npy_intp elsize = PyArray_ITEMSIZE(arr);  // 计算数组元素的大小
    PyArray_CompareFunc *cmp = PyDataType_GetArrFuncs(PyArray_DESCR(arr))->compare;  // 获取比较函数
    char *pl = (char *)start;  // 初始化数组起始位置
    char *pr = pl + num * elsize;  // 初始化数组结束位置
    char *pw;
    char *vp;
    int err = -NPY_ENOMEM;

    /* Items that have zero size don't make sense to sort */
    if (elsize == 0) {
        return 0;  // 如果元素大小为0，返回0
    }

    pw = (char *)malloc((num >> 1) * elsize);  // 分配内存给pw
    vp = (char *)malloc(elsize);  // 分配内存给vp

    if (pw != NULL && vp != NULL) {
        npy_mergesort0(pl, pr, pw, vp, elsize, cmp, arr);  // 调用归并排序函数
        err = 0;  // 表示排序成功
    }

    free(vp);  // 释放vp指向的内存空间
    free(pw);  // 释放pw指向的内存空间

    return err;  // 返回错误码或者0表示成功
}

static void
/* 
 * 使用合并排序算法对数组进行排序，其中包含以下参数：
 * - pl: 数组的左边界指针
 * - pr: 数组的右边界指针
 * - v: 指向数据的指针
 * - pw: 临时数组的指针，用于存储排序过程中的中间结果
 * - elsize: 元素大小
 * - cmp: 比较函数指针，用于比较数组元素
 * - arr: NumPy 数组对象指针，包含排序数据的描述信息
 */
npy_amergesort0(npy_intp *pl, npy_intp *pr, char *v, npy_intp *pw,
                npy_intp elsize, PyArray_CompareFunc *cmp, PyArrayObject *arr)
{
    char *vp;
    npy_intp vi, *pi, *pj, *pk, *pm;

    if (pr - pl > SMALL_MERGESORT) {
        /* 使用合并排序 */
        pm = pl + ((pr - pl) >> 1);
        npy_amergesort0(pl, pm, v, pw, elsize, cmp, arr);
        npy_amergesort0(pm, pr, v, pw, elsize, cmp, arr);
        for (pi = pw, pj = pl; pj < pm;) {
            *pi++ = *pj++;
        }
        pi = pw + (pm - pl);
        pj = pw;
        pk = pl;
        while (pj < pi && pm < pr) {
            if (cmp(v + (*pm) * elsize, v + (*pj) * elsize, arr) < 0) {
                *pk++ = *pm++;
            }
            else {
                *pk++ = *pj++;
            }
        }
        while (pj < pi) {
            *pk++ = *pj++;
        }
    }
    else {
        /* 使用插入排序 */
        for (pi = pl + 1; pi < pr; ++pi) {
            vi = *pi;
            vp = v + vi * elsize;
            pj = pi;
            pk = pi - 1;
            while (pj > pl && cmp(vp, v + (*pk) * elsize, arr) < 0) {
                *pj-- = *pk--;
            }
            *pj = vi;
        }
    }
}

/* 
 * 使用合并排序算法对数组进行排序的入口函数。
 * 其中包含以下参数：
 * - v: 指向数组的指针
 * - tosort: 指向要排序的元素的指针数组
 * - num: 要排序的元素数量
 * - varr: 指向 NumPy 数组对象的指针
 * 返回值：
 * - 0：排序成功
 * - -NPY_ENOMEM：内存分配失败
 */
NPY_NO_EXPORT int
npy_amergesort(void *v, npy_intp *tosort, npy_intp num, void *varr)
{
    PyArrayObject *arr = (PyArrayObject *)varr;
    npy_intp elsize = PyArray_ITEMSIZE(arr);
    PyArray_CompareFunc *cmp = PyDataType_GetArrFuncs(PyArray_DESCR(arr))->compare;
    npy_intp *pl, *pr, *pw;

    /* 如果元素大小为0，则没有意义进行排序 */
    if (elsize == 0) {
        return 0;
    }

    pl = tosort;
    pr = pl + num;
    pw = (npy_intp *)malloc((num >> 1) * sizeof(npy_intp));
    if (pw == NULL) {
        return -NPY_ENOMEM;
    }
    npy_amergesort0(pl, pr, (char *)v, pw, elsize, cmp, arr);
    free(pw);

    return 0;
}

/***************************************
 * C > C++ 调度
 ***************************************/

/* 
 * 对布尔类型数组使用合并排序的函数
 * - start: 指向布尔类型数组的指针
 * - num: 要排序的元素数量
 * - varr: 指向 NumPy 数组对象的指针（未使用）
 */
NPY_NO_EXPORT int
mergesort_bool(void *start, npy_intp num, void *NPY_UNUSED(varr))
{
    return mergesort_<npy::bool_tag>((npy_bool *)start, num);
}

/* 
 * 对字节类型数组使用合并排序的函数
 * - start: 指向字节类型数组的指针
 * - num: 要排序的元素数量
 * - varr: 指向 NumPy 数组对象的指针（未使用）
 */
NPY_NO_EXPORT int
mergesort_byte(void *start, npy_intp num, void *NPY_UNUSED(varr))
{
    return mergesort_<npy::byte_tag>((npy_byte *)start, num);
}

/* 
 * 对无符号字节类型数组使用合并排序的函数
 * - start: 指向无符号字节类型数组的指针
 * - num: 要排序的元素数量
 * - varr: 指向 NumPy 数组对象的指针（未使用）
 */
NPY_NO_EXPORT int
mergesort_ubyte(void *start, npy_intp num, void *NPY_UNUSED(varr))
{
    return mergesort_<npy::ubyte_tag>((npy_ubyte *)start, num);
}

/* 
 * 对短整型数组使用合并排序的函数
 * - start: 指向短整型数组的指针
 * - num: 要排序的元素数量
 * - varr: 指向 NumPy 数组对象的指针（未使用）
 */
NPY_NO_EXPORT int
mergesort_short(void *start, npy_intp num, void *NPY_UNUSED(varr))
{
    return mergesort_<npy::short_tag>((npy_short *)start, num);
}

/* 
 * 对无符号短整型数组使用合并排序的函数
 * - start: 指向无符号短整型数组的指针
 * - num: 要排序的元素数量
 * - varr: 指向 NumPy 数组对象的指针（未使用）
 */
NPY_NO_EXPORT int
mergesort_ushort(void *start, npy_intp num, void *NPY_UNUSED(varr))
{
    return mergesort_<npy::ushort_tag>((npy_ushort *)start, num);
}

/* 
 * 对整型数组使用合并排序的函数
 * - start: 指向整型数组的指针
 * - num: 要排序的元素数量
 * - varr: 指向 NumPy 数组对象的指针（未使用）
 */
NPY_NO_EXPORT int
mergesort_int(void *start, npy_intp num, void *NPY_UNUSED(varr))
{
    return mergesort_<npy::int_tag>((npy_int *)start, num);
}

/* 
 * 对无符号整型数组使用合并排序的函数
 * - start: 指向无符号整型数组的指针
 * - num: 要排序的元素数量
 * - varr: 指向 NumPy 数组对象的指针（未使用）
 */
NPY_NO_EXPORT int
mergesort_uint(void *start, npy_intp num, void *NPY_UNUSED(varr)))
{
    // mergesort_uint 函数尚未完成
    return 0;
}
    // 调用 mergesort_<npy::uint_tag> 函数，对给定的数组进行归并排序，并返回排序后的结果。
    return mergesort_<npy::uint_tag>((npy_uint *)start, num);
NPY_NO_EXPORT int
mergesort_long(void *start, npy_intp num, void *NPY_UNUSED(varr))
{
    // 调用模板函数 mergesort_，使用 npy::long_tag 进行排序，返回排序结果
    return mergesort_<npy::long_tag>((npy_long *)start, num);
}

NPY_NO_EXPORT int
mergesort_ulong(void *start, npy_intp num, void *NPY_UNUSED(varr))
{
    // 调用模板函数 mergesort_，使用 npy::ulong_tag 进行排序，返回排序结果
    return mergesort_<npy::ulong_tag>((npy_ulong *)start, num);
}

NPY_NO_EXPORT int
mergesort_longlong(void *start, npy_intp num, void *NPY_UNUSED(varr))
{
    // 调用模板函数 mergesort_，使用 npy::longlong_tag 进行排序，返回排序结果
    return mergesort_<npy::longlong_tag>((npy_longlong *)start, num);
}

NPY_NO_EXPORT int
mergesort_ulonglong(void *start, npy_intp num, void *NPY_UNUSED(varr))
{
    // 调用模板函数 mergesort_，使用 npy::ulonglong_tag 进行排序，返回排序结果
    return mergesort_<npy::ulonglong_tag>((npy_ulonglong *)start, num);
}

NPY_NO_EXPORT int
mergesort_half(void *start, npy_intp num, void *NPY_UNUSED(varr))
{
    // 调用模板函数 mergesort_，使用 npy::half_tag 进行排序，返回排序结果
    return mergesort_<npy::half_tag>((npy_half *)start, num);
}

NPY_NO_EXPORT int
mergesort_float(void *start, npy_intp num, void *NPY_UNUSED(varr))
{
    // 调用模板函数 mergesort_，使用 npy::float_tag 进行排序，返回排序结果
    return mergesort_<npy::float_tag>((npy_float *)start, num);
}

NPY_NO_EXPORT int
mergesort_double(void *start, npy_intp num, void *NPY_UNUSED(varr))
{
    // 调用模板函数 mergesort_，使用 npy::double_tag 进行排序，返回排序结果
    return mergesort_<npy::double_tag>((npy_double *)start, num);
}

NPY_NO_EXPORT int
mergesort_longdouble(void *start, npy_intp num, void *NPY_UNUSED(varr))
{
    // 调用模板函数 mergesort_，使用 npy::longdouble_tag 进行排序，返回排序结果
    return mergesort_<npy::longdouble_tag>((npy_longdouble *)start, num);
}

NPY_NO_EXPORT int
mergesort_cfloat(void *start, npy_intp num, void *NPY_UNUSED(varr))
{
    // 调用模板函数 mergesort_，使用 npy::cfloat_tag 进行排序，返回排序结果
    return mergesort_<npy::cfloat_tag>((npy_cfloat *)start, num);
}

NPY_NO_EXPORT int
mergesort_cdouble(void *start, npy_intp num, void *NPY_UNUSED(varr))
{
    // 调用模板函数 mergesort_，使用 npy::cdouble_tag 进行排序，返回排序结果
    return mergesort_<npy::cdouble_tag>((npy_cdouble *)start, num);
}

NPY_NO_EXPORT int
mergesort_clongdouble(void *start, npy_intp num, void *NPY_UNUSED(varr))
{
    // 调用模板函数 mergesort_，使用 npy::clongdouble_tag 进行排序，返回排序结果
    return mergesort_<npy::clongdouble_tag>((npy_clongdouble *)start, num);
}

NPY_NO_EXPORT int
mergesort_datetime(void *start, npy_intp num, void *NPY_UNUSED(varr))
{
    // 调用模板函数 mergesort_，使用 npy::datetime_tag 进行排序，返回排序结果
    return mergesort_<npy::datetime_tag>((npy_datetime *)start, num);
}

NPY_NO_EXPORT int
mergesort_timedelta(void *start, npy_intp num, void *NPY_UNUSED(varr))
{
    // 调用模板函数 mergesort_，使用 npy::timedelta_tag 进行排序，返回排序结果
    return mergesort_<npy::timedelta_tag>((npy_timedelta *)start, num);
}

NPY_NO_EXPORT int
amergesort_bool(void *start, npy_intp *tosort, npy_intp num,
                void *NPY_UNUSED(varr))
{
    // 调用模板函数 amergesort_，使用 npy::bool_tag 进行排序，返回排序结果
    return amergesort_<npy::bool_tag>((npy_bool *)start, tosort, num);
}

NPY_NO_EXPORT int
amergesort_byte(void *start, npy_intp *tosort, npy_intp num,
                void *NPY_UNUSED(varr))
{
    // 调用模板函数 amergesort_，使用 npy::byte_tag 进行排序，返回排序结果
    return amergesort_<npy::byte_tag>((npy_byte *)start, tosort, num);
}

NPY_NO_EXPORT int
amergesort_ubyte(void *start, npy_intp *tosort, npy_intp num,
                 void *NPY_UNUSED(varr))
{
    // 调用模板函数 amergesort_，使用 npy::ubyte_tag 进行排序，返回排序结果
    return amergesort_<npy::ubyte_tag>((npy_ubyte *)start, tosort, num);
}

NPY_NO_EXPORT int
amergesort_short(void *start, npy_intp *tosort, npy_intp num,
                 void *NPY_UNUSED(varr))
{
    // 调用模板函数 amergesort_，使用 npy::short_tag 进行排序，返回排序结果
    return amergesort_<npy::short_tag>((npy_short *)start, tosort, num);
}

NPY_NO_EXPORT int
amergesort_ushort(void *start, npy_intp *tosort, npy_intp num,
                  void *NPY_UNUSED(varr))
{
    // 调用模板函数 amergesort_，使用 npy::ushort_tag 进行排序，返回排序结果
    return amergesort_<npy::ushort_tag>((npy_ushort *)start, tosort, num);
}


以上是给定代码的注释。
    # 调用amergesort_<npy::ushort_tag>函数，传入参数并返回结果
    return amergesort_<npy::ushort_tag>((npy_ushort *)start, tosort, num);
NPY_NO_EXPORT int
amergesort_int(void *start, npy_intp *tosort, npy_intp num,
               void *NPY_UNUSED(varr))
{
    // 调用模板函数 amergesort_，以处理整数类型的排序，返回排序完成的状态
    return amergesort_<npy::int_tag>((npy_int *)start, tosort, num);
}

NPY_NO_EXPORT int
amergesort_uint(void *start, npy_intp *tosort, npy_intp num,
                void *NPY_UNUSED(varr))
{
    // 调用模板函数 amergesort_，以处理无符号整数类型的排序，返回排序完成的状态
    return amergesort_<npy::uint_tag>((npy_uint *)start, tosort, num);
}

NPY_NO_EXPORT int
amergesort_long(void *start, npy_intp *tosort, npy_intp num,
                void *NPY_UNUSED(varr))
{
    // 调用模板函数 amergesort_，以处理长整型的排序，返回排序完成的状态
    return amergesort_<npy::long_tag>((npy_long *)start, tosort, num);
}

NPY_NO_EXPORT int
amergesort_ulong(void *start, npy_intp *tosort, npy_intp num,
                 void *NPY_UNUSED(varr))
{
    // 调用模板函数 amergesort_，以处理无符号长整型的排序，返回排序完成的状态
    return amergesort_<npy::ulong_tag>((npy_ulong *)start, tosort, num);
}

NPY_NO_EXPORT int
amergesort_longlong(void *start, npy_intp *tosort, npy_intp num,
                    void *NPY_UNUSED(varr))
{
    // 调用模板函数 amergesort_，以处理长长整型的排序，返回排序完成的状态
    return amergesort_<npy::longlong_tag>((npy_longlong *)start, tosort, num);
}

NPY_NO_EXPORT int
amergesort_ulonglong(void *start, npy_intp *tosort, npy_intp num,
                     void *NPY_UNUSED(varr))
{
    // 调用模板函数 amergesort_，以处理无符号长长整型的排序，返回排序完成的状态
    return amergesort_<npy::ulonglong_tag>((npy_ulonglong *)start, tosort,
                                           num);
}

NPY_NO_EXPORT int
amergesort_half(void *start, npy_intp *tosort, npy_intp num,
                void *NPY_UNUSED(varr))
{
    // 调用模板函数 amergesort_，以处理半精度浮点数的排序，返回排序完成的状态
    return amergesort_<npy::half_tag>((npy_half *)start, tosort, num);
}

NPY_NO_EXPORT int
amergesort_float(void *start, npy_intp *tosort, npy_intp num,
                 void *NPY_UNUSED(varr))
{
    // 调用模板函数 amergesort_，以处理单精度浮点数的排序，返回排序完成的状态
    return amergesort_<npy::float_tag>((npy_float *)start, tosort, num);
}

NPY_NO_EXPORT int
amergesort_double(void *start, npy_intp *tosort, npy_intp num,
                  void *NPY_UNUSED(varr))
{
    // 调用模板函数 amergesort_，以处理双精度浮点数的排序，返回排序完成的状态
    return amergesort_<npy::double_tag>((npy_double *)start, tosort, num);
}

NPY_NO_EXPORT int
amergesort_longdouble(void *start, npy_intp *tosort, npy_intp num,
                      void *NPY_UNUSED(varr))
{
    // 调用模板函数 amergesort_，以处理长双精度浮点数的排序，返回排序完成的状态
    return amergesort_<npy::longdouble_tag>((npy_longdouble *)start, tosort,
                                            num);
}

NPY_NO_EXPORT int
amergesort_cfloat(void *start, npy_intp *tosort, npy_intp num,
                  void *NPY_UNUSED(varr))
{
    // 调用模板函数 amergesort_，以处理复数浮点数（单精度）的排序，返回排序完成的状态
    return amergesort_<npy::cfloat_tag>((npy_cfloat *)start, tosort, num);
}

NPY_NO_EXPORT int
amergesort_cdouble(void *start, npy_intp *tosort, npy_intp num,
                   void *NPY_UNUSED(varr))
{
    // 调用模板函数 amergesort_，以处理复数浮点数（双精度）的排序，返回排序完成的状态
    return amergesort_<npy::cdouble_tag>((npy_cdouble *)start, tosort, num);
}

NPY_NO_EXPORT int
amergesort_clongdouble(void *start, npy_intp *tosort, npy_intp num,
                       void *NPY_UNUSED(varr))
{
    // 调用模板函数 amergesort_，以处理复数浮点数（长双精度）的排序，返回排序完成的状态
    return amergesort_<npy::clongdouble_tag>((npy_clongdouble *)start, tosort,
                                             num);
}

NPY_NO_EXPORT int
amergesort_datetime(void *start, npy_intp *tosort, npy_intp num,
                    void *NPY_UNUSED(varr))
{
    // 调用模板函数 amergesort_，以处理日期时间类型的排序，返回排序完成的状态
    return amergesort_<npy::datetime_tag>((npy_datetime *)start, tosort, num);
}
# 定义一个不导出的函数，使用 amergesort 算法对 timedelta 类型的数据进行排序
NPY_NO_EXPORT int
amergesort_timedelta(void *start, npy_intp *tosort, npy_intp num,
                     void *NPY_UNUSED(varr))
{
    // 调用 amergesort_ 函数，对应 timedelta 类型的排序算法
    return amergesort_<npy::timedelta_tag>((npy_timedelta *)start, tosort,
                                           num);
}

# 定义一个不导出的函数，使用 mergesort 算法对 string 类型的数据进行排序
NPY_NO_EXPORT int
mergesort_string(void *start, npy_intp num, void *varr)
{
    // 调用 string_mergesort_ 函数，对应 string 类型的 mergesort 排序算法
    return string_mergesort_<npy::string_tag>((npy_char *)start, num, varr);
}

# 定义一个不导出的函数，使用 mergesort 算法对 unicode 类型的数据进行排序
NPY_NO_EXPORT int
mergesort_unicode(void *start, npy_intp num, void *varr)
{
    // 调用 string_mergesort_ 函数，对应 unicode 类型的 mergesort 排序算法
    return string_mergesort_<npy::unicode_tag>((npy_ucs4 *)start, num, varr);
}

# 定义一个不导出的函数，使用 amergesort 算法对 string 类型的数据进行排序
NPY_NO_EXPORT int
amergesort_string(void *v, npy_intp *tosort, npy_intp num, void *varr)
{
    // 调用 string_amergesort_ 函数，对应 string 类型的 amergesort 排序算法
    return string_amergesort_<npy::string_tag>((npy_char *)v, tosort, num,
                                               varr);
}

# 定义一个不导出的函数，使用 amergesort 算法对 unicode 类型的数据进行排序
NPY_NO_EXPORT int
amergesort_unicode(void *v, npy_intp *tosort, npy_intp num, void *varr)
{
    // 调用 string_amergesort_ 函数，对应 unicode 类型的 amergesort 排序算法
    return string_amergesort_<npy::unicode_tag>((npy_ucs4 *)v, tosort, num,
                                                varr);
}
```