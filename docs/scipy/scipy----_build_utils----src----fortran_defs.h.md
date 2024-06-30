# `D:\src\scipysrc\scipy\scipy\_build_utils\src\fortran_defs.h`

```
/*
 * Handle different Fortran conventions.
 */

#if defined(NO_APPEND_FORTRAN)
#if defined(UPPERCASE_FORTRAN)
#define F_FUNC(f,F) F
#else
#define F_FUNC(f,F) f
#endif
#else
#if defined(UPPERCASE_FORTRAN)
// 定义一个宏，根据 Fortran 惯例将函数名转换为大写形式
#define F_FUNC(f,F) F##_
#else
// 定义一个宏，根据 Fortran 惯例将函数名转换为带下划线后缀的形式
#define F_FUNC(f,F) f##_
#endif
#endif


这段代码主要是一系列预处理指令，用于根据不同的编译宏定义（NO_APPEND_FORTRAN 和 UPPERCASE_FORTRAN），定义宏 F_FUNC，用于在 Fortran 中处理函数名的转换。
```