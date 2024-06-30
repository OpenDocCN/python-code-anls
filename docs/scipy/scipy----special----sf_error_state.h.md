# `D:\src\scipysrc\scipy\scipy\special\sf_error_state.h`

```
#pragma once
// 当前文件的预处理命令，表示只包含一次该头文件

#include "scipy_dll.h"
// 包含自定义的 scipy_dll.h 头文件

#include "special/error.h"
// 包含自定义的 special/error.h 头文件

#ifdef __cplusplus
extern "C" {
#endif
// 如果是 C++ 编译环境，则按 C 语言的方式编译

    typedef enum {
        SF_ERROR_IGNORE = 0,  /* Ignore errors */
        SF_ERROR_WARN,        /* Warn on errors */
        SF_ERROR_RAISE        /* Raise on errors */
    } sf_action_t;
    // 定义枚举类型 sf_action_t，包括 SF_ERROR_IGNORE、SF_ERROR_WARN、SF_ERROR_RAISE 三种错误处理方式

    SCIPY_DLL void scipy_sf_error_set_action(sf_error_t code, sf_action_t action);
    // 声明 scipy_sf_error_set_action 函数，用于设置特定错误码的处理方式

    SCIPY_DLL sf_action_t scipy_sf_error_get_action(sf_error_t code);
    // 声明 scipy_sf_error_get_action 函数，用于获取特定错误码的处理方式

#ifdef __cplusplus
}
#endif
// 如果是 C++ 编译环境，结束以 C 语言方式编译的部分
```