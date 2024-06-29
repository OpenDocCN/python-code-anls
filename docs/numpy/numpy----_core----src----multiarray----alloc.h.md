# `.\numpy\numpy\_core\src\multiarray\alloc.h`

```py
#ifndef NUMPY_CORE_SRC_MULTIARRAY_ALLOC_H_   // 如果未定义 NUMPY_CORE_SRC_MULTIARRAY_ALLOC_H_ 则执行下面的代码
#define NUMPY_CORE_SRC_MULTIARRAY_ALLOC_H_   // 定义 NUMPY_CORE_SRC_MULTIARRAY_ALLOC_H_

#define NPY_NO_DEPRECATED_API NPY_API_VERSION  // 设置 NPY_NO_DEPRECATED_API 为 NPY_API_VERSION
#define _MULTIARRAYMODULE  // 定义 _MULTIARRAYMODULE
#include "numpy/ndarraytypes.h"  // 引入 "numpy/ndarraytypes.h"

#define NPY_TRACE_DOMAIN 389047  // 设置 NPY_TRACE_DOMAIN 为 389047
#define MEM_HANDLER_CAPSULE_NAME "mem_handler"  // 设置 MEM_HANDLER_CAPSULE_NAME 为 "mem_handler"

NPY_NO_EXPORT PyObject *   // NPY_NO_EXPORT修饰的PyObject指针类型的函数
_get_madvise_hugepage(PyObject *NPY_UNUSED(self), PyObject *NPY_UNUSED(args));  // 获取_madvise_hugepage函数声明

NPY_NO_EXPORT PyObject *  // NPY_NO_EXPORT修饰的PyObject指针类型的函数
_set_madvise_hugepage(PyObject *NPY_UNUSED(self), PyObject *enabled_obj);  // 设置_madvise_hugepage函数声明

NPY_NO_EXPORT void *  // NPY_NO_EXPORT修饰的void指针类型的函数
PyDataMem_UserNEW(npy_uintp sz, PyObject *mem_handler);  // PyDataMem_UserNEW函数声明

NPY_NO_EXPORT void *   // NPY_NO_EXPORT修饰的void指针类型的函数
PyDataMem_UserNEW_ZEROED(size_t nmemb, size_t size, PyObject *mem_handler);  // PyDataMem_UserNEW_ZEROED函数声明

NPY_NO_EXPORT void   // NPY_NO_EXPORT修饰的void类型的函数
PyDataMem_UserFREE(void * p, npy_uintp sd, PyObject *mem_handler);  // PyDataMem_UserFREE函数声明

NPY_NO_EXPORT void *   // NPY_NO_EXPORT修饰的void指针类型的函数
PyDataMem_UserRENEW(void *ptr, size_t size, PyObject *mem_handler);  // PyDataMem_UserRENEW函数声明

NPY_NO_EXPORT void *   // NPY_NO_EXPORT修饰的void指针类型的函数
npy_alloc_cache_dim(npy_uintp sz);  // npy_alloc_cache_dim函数声明

NPY_NO_EXPORT void   // NPY_NO_EXPORT修饰的void类型的函数
npy_free_cache_dim(void * p, npy_uintp sd);  // npy_free_cache_dim函数声明

static inline void  // 内联函数void类型
npy_free_cache_dim_obj(PyArray_Dims dims)  // npy_free_cache_dim_obj函数声明
{
    npy_free_cache_dim(dims.ptr, dims.len);  // 调用npy_free_cache_dim函数
}

static inline void  // 内联函数void类型
npy_free_cache_dim_array(PyArrayObject * arr)  // npy_free_cache_dim_array函数声明
{
    npy_free_cache_dim(PyArray_DIMS(arr), PyArray_NDIM(arr));  // 调用npy_free_cache_dim函数
}

extern PyDataMem_Handler default_handler;  // 声明外部变量default_handler，类型为PyDataMem_Handler
extern PyObject *current_handler;  // 声明外部变量current_handler，类型为PyObject指针

NPY_NO_EXPORT PyObject *  // NPY_NO_EXPORT修饰的PyObject指针类型的函数
get_handler_name(PyObject *NPY_UNUSED(self), PyObject *obj);  // get_handler_name函数声明

NPY_NO_EXPORT PyObject *  // NPY_NO_EXPORT修饰的PyObject指针类型的函数
get_handler_version(PyObject *NPY_UNUSED(self), PyObject *obj);  // get_handler_version函数声明

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_ALLOC_H_ */   // 如果未定义 NUMPY_CORE_SRC_MULTIARRAY_ALLOC_H_ 则结束```
```