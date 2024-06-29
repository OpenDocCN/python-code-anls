# `.\numpy\numpy\_core\src\npymath\npy_math.c`

```
/*
 * vim:syntax=c
 * This file is compiled into the npy_math library with externally visible
 * symbols, and the static and inline specifiers utilized in the npy_math
 * function definitions are switched off.
 */
// 设置 C 语法高亮，指定这个文件被编译为 npy_math 库，其中包含外部可见的符号
// 在 npy_math 函数定义中，关闭了 static 和 inline 修饰符的使用
#define NPY_INLINE_MATH 0
// 包含 npy_math_internal.h 头文件
#include "npy_math_internal.h"
```