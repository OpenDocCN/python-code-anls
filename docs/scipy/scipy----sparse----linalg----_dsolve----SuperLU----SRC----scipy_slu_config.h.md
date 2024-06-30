# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_dsolve\SuperLU\SRC\scipy_slu_config.h`

```
#ifndef SCIPY_SLU_CONFIG_H
#define SCIPY_SLU_CONFIG_H

#include <stdlib.h>

/*
 * Support routines
 */

// 定义终止函数，用于打印错误消息并中止程序
void superlu_python_module_abort(char *msg);

// 定义内存分配函数，用于分配指定大小的内存块并返回指针
void *superlu_python_module_malloc(size_t size);

// 定义内存释放函数，用于释放之前分配的内存块
void superlu_python_module_free(void *ptr);

// 定义宏，将终止函数绑定到 USER_ABORT 宏
#define USER_ABORT  superlu_python_module_abort

// 定义宏，将内存分配函数绑定到 USER_MALLOC 宏
#define USER_MALLOC superlu_python_module_malloc

// 定义宏，将内存释放函数绑定到 USER_FREE 宏
#define USER_FREE   superlu_python_module_free

// 定义宏，指定开启 SCIPY_FIX 功能
#define SCIPY_FIX 1

/*
 * Fortran configuration
 */

// 如果定义了 NO_APPEND_FORTRAN 宏
#if defined(NO_APPEND_FORTRAN)
    // 如果定义了 UPPERCASE_FORTRAN 宏
    #if defined(UPPERCASE_FORTRAN)
        // 定义 UpCase 宏为 1，表示使用大写字母的 Fortran 函数名
        #define UpCase 1
    // 如果未定义 UPPERCASE_FORTRAN 宏
    #else
        // 定义 NoChange 宏为 1，表示保持 Fortran 函数名不变
        #define NoChange 1
    #endif
// 如果未定义 NO_APPEND_FORTRAN 宏
#else
    // 如果定义了 UPPERCASE_FORTRAN 宏
    #if defined(UPPERCASE_FORTRAN)
        // 抛出编译错误，因为不支持大写字母和斜杠结尾的 Fortran 函数名
        #error Uppercase and trailing slash in Fortran names not supported
    // 如果未定义 UPPERCASE_FORTRAN 宏
    #else
        // 定义 Add_ 宏为 1，表示添加下划线到 Fortran 函数名末尾
        #define Add_ 1
    #endif
#endif

#endif
```