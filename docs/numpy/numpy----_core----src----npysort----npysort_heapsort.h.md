# `.\numpy\numpy\_core\src\npysort\npysort_heapsort.h`

```py
#ifndef __NPY_SORT_HEAPSORT_H__
#define __NPY_SORT_HEAPSORT_H__

#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include "npy_sort.h"
#include "npysort_common.h"
#include "numpy_tag.h"

#include <cstdlib>

/*
 *****************************************************************************
 **                            NUMERIC SORTS                                **
 *****************************************************************************
 */

// 定义 heapsort_ 模板函数，用于对类型为 type 的数组进行堆排序
template <typename Tag, typename type>
inline NPY_NO_EXPORT
int heapsort_(type *start, npy_intp n)
{
    type tmp, *a;
    npy_intp i, j, l;

    /* The array needs to be offset by one for heapsort indexing */
    a = start - 1;

    // 建立最大堆
    for (l = n >> 1; l > 0; --l) {
        tmp = a[l];
        for (i = l, j = l << 1; j <= n;) {
            // 如果右子节点存在且大于左子节点，则选择右子节点
            if (j < n && Tag::less(a[j], a[j + 1])) {
                j += 1;
            }
            // 如果当前节点小于子节点，则交换
            if (Tag::less(tmp, a[j])) {
                a[i] = a[j];
                i = j;
                j += j;
            }
            else {
                break;
            }
        }
        a[i] = tmp;
    }

    // 堆排序
    for (; n > 1;) {
        tmp = a[n];
        a[n] = a[1];
        n -= 1;
        for (i = 1, j = 2; j <= n;) {
            // 如果右子节点存在且大于左子节点，则选择右子节点
            if (j < n && Tag::less(a[j], a[j + 1])) {
                j++;
            }
            // 如果根节点小于子节点，则交换
            if (Tag::less(tmp, a[j])) {
                a[i] = a[j];
                i = j;
                j += j;
            }
            else {
                break;
            }
        }
        a[i] = tmp;
    }

    return 0;
}

// 定义 aheapsort_ 模板函数，用于对类型为 type 的数组进行间接堆排序
template <typename Tag, typename type>
inline NPY_NO_EXPORT
int aheapsort_(type *vv, npy_intp *tosort, npy_intp n)
{
    type *v = vv;
    npy_intp *a, i, j, l, tmp;
    /* The arrays need to be offset by one for heapsort indexing */
    a = tosort - 1;

    // 建立最大堆
    for (l = n >> 1; l > 0; --l) {
        tmp = a[l];
        for (i = l, j = l << 1; j <= n;) {
            // 如果右子节点存在且大于左子节点，则选择右子节点
            if (j < n && Tag::less(v[a[j]], v[a[j + 1]])) {
                j += 1;
            }
            // 如果当前节点小于子节点，则交换
            if (Tag::less(v[tmp], v[a[j]])) {
                a[i] = a[j];
                i = j;
                j += j;
            }
            else {
                break;
            }
        }
        a[i] = tmp;
    }

    // 堆排序
    for (; n > 1;) {
        tmp = a[n];
        a[n] = a[1];
        n -= 1;
        for (i = 1, j = 2; j <= n;) {
            // 如果右子节点存在且大于左子节点，则选择右子节点
            if (j < n && Tag::less(v[a[j]], v[a[j + 1]])) {
                j++;
            }
            // 如果根节点小于子节点，则交换
            if (Tag::less(v[tmp], v[a[j]])) {
                a[i] = a[j];
                i = j;
                j += j;
            }
            else {
                break;
            }
        }
        a[i] = tmp;
    }

    return 0;
}

/*
 *****************************************************************************
 **                            STRING SORTS                                 **
 *****************************************************************************
 */

template <typename Tag, typename type>
inline NPY_NO_EXPORT
// 对字符串类型数组进行堆排序，使用 Tag 和 type 参数
int string_heapsort_(type *start, npy_intp n, void *varr)
{
    // 将 void 指针 varr 转换为 PyArrayObject 指针
    PyArrayObject *arr = (PyArrayObject *)varr;
    // 计算每个元素的大小，并确定数组长度 len
    size_t len = PyArray_ITEMSIZE(arr) / sizeof(type);
    // 如果数组长度为 0，则不需要排序，直接返回
    if (len == 0) {
        return 0;  /* no need for sorting if strings are empty */
    }

    // 分配临时存储空间 tmp
    type *tmp = (type *)malloc(PyArray_ITEMSIZE(arr));
    // 将 a 指针设为 start 的前一个位置，为堆排序做准备
    type *a = (type *)start - len;
    npy_intp i, j, l;

    // 如果分配失败，返回内存不足错误
    if (tmp == NULL) {
        return -NPY_ENOMEM;
    }

    // 建立最大堆
    for (l = n >> 1; l > 0; --l) {
        // 拷贝 a[l * len] 的内容到 tmp
        Tag::copy(tmp, a + l * len, len);
        for (i = l, j = l << 1; j <= n;) {
            // 比较 a[j * len] 和 a[(j + 1) * len] 的大小，选择较大的一个
            if (j < n && Tag::less(a + j * len, a + (j + 1) * len, len))
                j += 1;
            // 如果 tmp 小于 a[j * len]，则交换它们
            if (Tag::less(tmp, a + j * len, len)) {
                Tag::copy(a + i * len, a + j * len, len);
                i = j;
                j += j;
            }
            else {
                break;
            }
        }
        // 将 tmp 拷贝到最终位置
        Tag::copy(a + i * len, tmp, len);
    }

    // 排序堆
    for (; n > 1;) {
        // 拷贝 a[n * len] 到 tmp
        Tag::copy(tmp, a + n * len, len);
        // 将 a[len] 拷贝到 a[n * len]
        Tag::copy(a + n * len, a + len, len);
        n -= 1;
        for (i = 1, j = 2; j <= n;) {
            // 比较 a[j * len] 和 a[(j + 1) * len] 的大小，选择较大的一个
            if (j < n && Tag::less(a + j * len, a + (j + 1) * len, len))
                j++;
            // 如果 tmp 小于 a[j * len]，则交换它们
            if (Tag::less(tmp, a + j * len, len)) {
                Tag::copy(a + i * len, a + j * len, len);
                i = j;
                j += j;
            }
            else {
                break;
            }
        }
        // 将 tmp 拷贝到最终位置
        Tag::copy(a + i * len, tmp, len);
    }

    // 释放临时存储空间 tmp
    free(tmp);
    return 0;
}

// 对字符串类型数组进行堆排序，使用 Tag 和 type 参数
template <typename Tag, typename type>
inline NPY_NO_EXPORT
int string_aheapsort_(type *vv, npy_intp *tosort, npy_intp n, void *varr)
{
    // 将 void 指针 varr 转换为 PyArrayObject 指针
    type *v = vv;
    PyArrayObject *arr = (PyArrayObject *)varr;
    // 计算每个元素的大小，并确定数组长度 len
    size_t len = PyArray_ITEMSIZE(arr) / sizeof(type);
    npy_intp *a, i, j, l, tmp;

    // 将 tosort 指针向前偏移一个位置，为堆排序做准备
    a = tosort - 1;

    // 建立最大堆
    for (l = n >> 1; l > 0; --l) {
        tmp = a[l];
        for (i = l, j = l << 1; j <= n;) {
            // 比较 v[a[j] * len] 和 v[a[j + 1] * len] 的大小，选择较大的一个
            if (j < n && Tag::less(v + a[j] * len, v + a[j + 1] * len, len))
                j += 1;
            // 如果 v[tmp * len] 小于 v[a[j] * len]，则交换它们
            if (Tag::less(v + tmp * len, v + a[j] * len, len)) {
                a[i] = a[j];
                i = j;
                j += j;
            }
            else {
                break;
            }
        }
        // 将 tmp 拷贝到最终位置
        a[i] = tmp;
    }

    // 排序堆
    for (; n > 1;) {
        tmp = a[n];
        a[n] = a[1];
        n -= 1;
        for (i = 1, j = 2; j <= n;) {
            // 比较 v[a[j] * len] 和 v[a[j + 1] * len] 的大小，选择较大的一个
            if (j < n && Tag::less(v + a[j] * len, v + a[j + 1] * len, len))
                j++;
            // 如果 v[tmp * len] 小于 v[a[j] * len]，则交换它们
            if (Tag::less(v + tmp * len, v + a[j] * len, len)) {
                a[i] = a[j];
                i = j;
                j += j;
            }
            else {
                break;
            }
        }
        // 将 tmp 拷贝到最终位置
        a[i] = tmp;
    }

    return 0;
}

#endif
```