# `.\numpy\numpy\random\include\aligned_malloc.h`

```
#ifndef _RANDOMDGEN__ALIGNED_MALLOC_H_
#define _RANDOMDGEN__ALIGNED_MALLOC_H_

#include <Python.h>
#include "numpy/npy_common.h"

#define NPY_MEMALIGN 16 /* 内存对齐要求，为 SSE2 设置为16，AVX 设置为32，Xeon Phi 设置为64 */

// 重新分配对齐内存的函数，支持按需分配
static inline void *PyArray_realloc_aligned(void *p, size_t n)
{
    void *p1, **p2, *base;
    size_t old_offs, offs = NPY_MEMALIGN - 1 + sizeof(void *);
    
    // 如果指针 p 不为空，进行重新分配操作
    if (NPY_UNLIKELY(p != NULL))
    {
        // 获取原始分配的基地址
        base = *(((void **)p) - 1);
        // 尝试重新分配内存，以满足新的大小要求
        if (NPY_UNLIKELY((p1 = PyMem_Realloc(base, n + offs)) == NULL))
            return NULL;
        // 如果重新分配后的地址和原地址相同，则直接返回原指针 p
        if (NPY_LIKELY(p1 == base))
            return p;
        // 计算新的对齐地址并进行数据的移动
        p2 = (void **)(((Py_uintptr_t)(p1) + offs) & ~(NPY_MEMALIGN - 1));
        old_offs = (size_t)((Py_uintptr_t)p - (Py_uintptr_t)base);
        memmove((void *)p2, ((char *)p1) + old_offs, n);
    }
    else
    {
        // 如果 p 为空，直接分配新的对齐内存
        if (NPY_UNLIKELY((p1 = PyMem_Malloc(n + offs)) == NULL))
            return NULL;
        // 计算新的对齐地址
        p2 = (void **)(((Py_uintptr_t)(p1) + offs) & ~(NPY_MEMALIGN - 1));
    }
    // 记录分配的基地址，并返回对齐后的指针
    *(p2 - 1) = p1;
    return (void *)p2;
}

// 分配对齐内存的函数，直接调用 PyArray_realloc_aligned，并传入空指针 p
static inline void *PyArray_malloc_aligned(size_t n)
{
    return PyArray_realloc_aligned(NULL, n);
}

// 分配并清零对齐内存的函数，调用 PyArray_realloc_aligned，并在分配后使用 memset 清零
static inline void *PyArray_calloc_aligned(size_t n, size_t s)
{
    void *p;
    // 分配 n*s 大小的对齐内存
    if (NPY_UNLIKELY((p = PyArray_realloc_aligned(NULL, n * s)) == NULL))
        return NULL;
    // 清零分配的内存
    memset(p, 0, n * s);
    return p;
}

// 释放对齐内存的函数，通过基地址释放内存
static inline void PyArray_free_aligned(void *p)
{
    void *base = *(((void **)p) - 1);
    PyMem_Free(base);
}

#endif
```