# `.\numpy\numpy\_core\include\numpy\npy_no_deprecated_api.h`

```
/*
 * This include file is provided for inclusion in Cython *.pyd files where
 * one would like to define the NPY_NO_DEPRECATED_API macro. It can be
 * included by
 *
 * cdef extern from "npy_no_deprecated_api.h": pass
 *
 */

#ifndef NPY_NO_DEPRECATED_API
/* 检查是否已经包含了旧版 API 相关的头文件，如果是则报错 */
#if defined(NUMPY_CORE_INCLUDE_NUMPY_NDARRAYTYPES_H_) || \
    defined(NUMPY_CORE_INCLUDE_NUMPY_NPY_DEPRECATED_API_H) || \
    defined(NUMPY_CORE_INCLUDE_NUMPY_OLD_DEFINES_H_)
#error "npy_no_deprecated_api.h" must be first among numpy includes.
#else
/* 定义 NPY_NO_DEPRECATED_API 宏为当前 NPY_API_VERSION，以禁用过时的 API */
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#endif

#endif  /* NPY_NO_DEPRECATED_API */
```