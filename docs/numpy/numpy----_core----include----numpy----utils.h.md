# `.\numpy\numpy\_core\include\numpy\utils.h`

```py
#ifndef NUMPY_CORE_INCLUDE_NUMPY_UTILS_H_
#define NUMPY_CORE_INCLUDE_NUMPY_UTILS_H_

#ifndef __COMP_NPY_UNUSED
    #if defined(__GNUC__)
        // 如果是 GCC 编译器，则定义 __COMP_NPY_UNUSED 为未使用属性
        #define __COMP_NPY_UNUSED __attribute__ ((__unused__))
    #elif defined(__ICC)
        // 如果是 Intel 编译器，则定义 __COMP_NPY_UNUSED 为未使用属性
        #define __COMP_NPY_UNUSED __attribute__ ((__unused__))
    #elif defined(__clang__)
        // 如果是 Clang 编译器，则定义 __COMP_NPY_UNUSED 为未使用属性
        #define __COMP_NPY_UNUSED __attribute__ ((unused))
    #else
        // 其他情况下，__COMP_NPY_UNUSED 不做特殊处理
        #define __COMP_NPY_UNUSED
    #endif
#endif

#if defined(__GNUC__) || defined(__ICC) || defined(__clang__)
    // 如果是 GCC、Intel 编译器或者 Clang 编译器，则定义 NPY_DECL_ALIGNED(x) 为按 x 对齐
    #define NPY_DECL_ALIGNED(x) __attribute__ ((aligned (x)))
#elif defined(_MSC_VER)
    // 如果是 MSVC 编译器，则定义 NPY_DECL_ALIGNED(x) 为按 x 对齐
    #define NPY_DECL_ALIGNED(x) __declspec(align(x))
#else
    // 其他情况下，不做特殊处理
    #define NPY_DECL_ALIGNED(x)
#endif

/* Use this to tag a variable as not used. It will remove unused variable
 * warning on support platforms (see __COM_NPY_UNUSED) and mangle the variable
 * to avoid accidental use */
// 使用此宏标记未使用的变量，以消除在支持的平台上的未使用变量警告（参见 __COM_NPY_UNUSED），并混淆变量以避免意外使用
#define NPY_UNUSED(x) __NPY_UNUSED_TAGGED ## x __COMP_NPY_UNUSED
#define NPY_EXPAND(x) x

#define NPY_STRINGIFY(x) #x
#define NPY_TOSTRING(x) NPY_STRINGIFY(x)

#define NPY_CAT__(a, b) a ## b
#define NPY_CAT_(a, b) NPY_CAT__(a, b)
#define NPY_CAT(a, b) NPY_CAT_(a, b)

#endif  /* NUMPY_CORE_INCLUDE_NUMPY_UTILS_H_ */
```