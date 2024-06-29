# `.\numpy\numpy\_core\include\numpy\npy_1_7_deprecated_api.h`

```py
#ifndef NPY_DEPRECATED_INCLUDES
#error "Should never include npy_*_*_deprecated_api directly."
#endif


// 如果未定义 NPY_DEPRECATED_INCLUDES 宏，则产生编译错误，要求不直接包含 npy_*_*_deprecated_api 文件
#ifndef NUMPY_CORE_INCLUDE_NUMPY_NPY_1_7_DEPRECATED_API_H_
#define NUMPY_CORE_INCLUDE_NUMPY_NPY_1_7_DEPRECATED_API_H_


// 如果未定义 NUMPY_CORE_INCLUDE_NUMPY_NPY_1_7_DEPRECATED_API_H_ 宏，则定义该宏，避免重复包含本文件


/* Emit a warning if the user did not specifically request the old API */
#ifndef NPY_NO_DEPRECATED_API
#if defined(_WIN32)
#define _WARN___STR2__(x) #x
#define _WARN___STR1__(x) _WARN___STR2__(x)
#define _WARN___LOC__ __FILE__ "(" _WARN___STR1__(__LINE__) ") : Warning Msg: "
#pragma message(_WARN___LOC__"Using deprecated NumPy API, disable it with " \
                         "#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION")
#else
#warning "Using deprecated NumPy API, disable it with " \
         "#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION"
#endif
#endif


// 如果未定义 NPY_NO_DEPRECATED_API 宏，则生成编译警告，提示用户使用了已弃用的 NumPy API，并给出禁用方法


/*
 * This header exists to collect all dangerous/deprecated NumPy API
 * as of NumPy 1.7.
 *
 * This is an attempt to remove bad API, the proliferation of macros,
 * and namespace pollution currently produced by the NumPy headers.
 */


// 本头文件用于收集 NumPy 1.7 中所有危险/已弃用的 API。
// 旨在清除糟糕的 API、宏的过度使用以及当前由 NumPy 头文件引起的命名空间污染。


/* These array flags are deprecated as of NumPy 1.7 */
#define NPY_CONTIGUOUS NPY_ARRAY_C_CONTIGUOUS
#define NPY_FORTRAN NPY_ARRAY_F_CONTIGUOUS


// 这些数组标志在 NumPy 1.7 中已弃用
#define NPY_CONTIGUOUS NPY_ARRAY_C_CONTIGUOUS  // NPY_CONTIGUOUS 等同于 NPY_ARRAY_C_CONTIGUOUS
#define NPY_FORTRAN NPY_ARRAY_F_CONTIGUOUS      // NPY_FORTRAN 等同于 NPY_ARRAY_F_CONTIGUOUS


/*
 * The consistent NPY_ARRAY_* names which don't pollute the NPY_*
 * namespace were added in NumPy 1.7.
 *
 * These versions of the carray flags are deprecated, but
 * probably should only be removed after two releases instead of one.
 */
#define NPY_C_CONTIGUOUS   NPY_ARRAY_C_CONTIGUOUS
#define NPY_F_CONTIGUOUS   NPY_ARRAY_F_CONTIGUOUS
#define NPY_OWNDATA        NPY_ARRAY_OWNDATA
#define NPY_FORCECAST      NPY_ARRAY_FORCECAST
#define NPY_ENSURECOPY     NPY_ARRAY_ENSURECOPY
#define NPY_ENSUREARRAY    NPY_ARRAY_ENSUREARRAY
#define NPY_ELEMENTSTRIDES NPY_ARRAY_ELEMENTSTRIDES
#define NPY_ALIGNED        NPY_ARRAY_ALIGNED
#define NPY_NOTSWAPPED     NPY_ARRAY_NOTSWAPPED
#define NPY_WRITEABLE      NPY_ARRAY_WRITEABLE
#define NPY_BEHAVED        NPY_ARRAY_BEHAVED
#define NPY_BEHAVED_NS     NPY_ARRAY_BEHAVED_NS
#define NPY_CARRAY         NPY_ARRAY_CARRAY
#define NPY_CARRAY_RO      NPY_ARRAY_CARRAY_RO
#define NPY_FARRAY         NPY_ARRAY_FARRAY
#define NPY_FARRAY_RO      NPY_ARRAY_FARRAY_RO
#define NPY_DEFAULT        NPY_ARRAY_DEFAULT
#define NPY_IN_ARRAY       NPY_ARRAY_IN_ARRAY
#define NPY_OUT_ARRAY      NPY_ARRAY_OUT_ARRAY
#define NPY_INOUT_ARRAY    NPY_ARRAY_INOUT_ARRAY
#define NPY_IN_FARRAY      NPY_ARRAY_IN_FARRAY
#define NPY_OUT_FARRAY     NPY_ARRAY_OUT_FARRAY
#define NPY_INOUT_FARRAY   NPY_ARRAY_INOUT_FARRAY
#define NPY_UPDATE_ALL     NPY_ARRAY_UPDATE_ALL


// 在 NumPy 1.7 中添加了一致的 NPY_ARRAY_* 名称，避免了 NPY_* 命名空间的污染。
// 这些 carray 标志的版本已弃用，但可能应该在两个版本发布后移除，而不是一个版本。


/* This way of accessing the default type is deprecated as of NumPy 1.7 */
#define PyArray_DEFAULT NPY_DEFAULT_TYPE


// 在 NumPy 1.7 中已弃用这种访问默认类型的方式
#define PyArray_DEFAULT NPY_DEFAULT_TYPE  // PyArray_DEFAULT 等同于 NPY_DEFAULT_TYPE


/*
 * Deprecated as of NumPy 1.7, this kind of shortcut doesn't
 * belong in the public API.
 */
#define NPY_AO PyArrayObject


// 在 NumPy 1.7 中已弃用此类快捷方式，不应出现在公共 API 中。
#define NPY_AO PyArrayObject  // NPY_AO 等同于 PyArrayObject


/*
 * Deprecated as of NumPy 1.7, an all-lowercase macro doesn't
 * belong in the public API.
 */
#define fortran fortran_


// 在 NumPy 1.7 中已弃用全部小写的宏，不应出现在公共 API 中。
#define fortran fortran_  // fortran 等同于 fortran_
/*
 * NumPy 1.7 版本后已弃用，因为它是一个污染命名空间的宏定义。
 */
#define FORTRAN_IF PyArray_FORTRAN_IF

/* NumPy 1.7 版本后已弃用，datetime64 类型现在使用 c_metadata 替代 */
#define NPY_METADATA_DTSTR "__timeunit__"

/*
 * NumPy 1.7 版本后已弃用。
 * 原因：
 *  - 这些是用于 datetime 的，但没有 datetime 的 "namespace"。
 *  - 它们只是将 NPY_STR_<x> 转换为 "<x>"，这只是将简单的东西变成了间接引用。
 */
#define NPY_STR_Y "Y"
#define NPY_STR_M "M"
#define NPY_STR_W "W"
#define NPY_STR_D "D"
#define NPY_STR_h "h"
#define NPY_STR_m "m"
#define NPY_STR_s "s"
#define NPY_STR_ms "ms"
#define NPY_STR_us "us"
#define NPY_STR_ns "ns"
#define NPY_STR_ps "ps"
#define NPY_STR_fs "fs"
#define NPY_STR_as "as"

#endif  /* NUMPY_CORE_INCLUDE_NUMPY_NPY_1_7_DEPRECATED_API_H_ */


注释：
```