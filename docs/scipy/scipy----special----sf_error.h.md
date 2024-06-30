# `D:\src\scipysrc\scipy\scipy\special\sf_error.h`

```
#pragma once

# 告诉编译器只包含此文件一次，防止多重包含


#include "sf_error_state.h"

# 包含名为 "sf_error_state.h" 的头文件，引入错误状态相关的定义和声明


#ifdef __cplusplus
extern "C" {
#endif

# 如果是 C++ 环境，则使用 extern "C" 将后续的函数声明放在 C 语言代码块中处理


extern const char *sf_error_messages[];

# 声明一个全局的常量字符指针数组 sf_error_messages，用于存储错误消息字符串的地址


void sf_error(const char *func_name, sf_error_t code, const char *fmt, ...);

# 声明一个函数 sf_error，接受函数名、错误代码、格式化字符串等参数，用于处理和报告错误


void sf_error_check_fpe(const char *func_name);

# 声明一个函数 sf_error_check_fpe，接受函数名参数，用于检查和处理浮点异常错误


#ifdef __cplusplus
}
#endif

# 如果是 C++ 环境，则结束 extern "C" 块
```