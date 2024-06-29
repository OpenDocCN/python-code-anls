# `.\numpy\numpy\_core\src\umath\clip.h`

```py
#ifndef _NPY_UMATH_CLIP_H_
#define _NPY_UMATH_CLIP_H_

#ifdef __cplusplus
extern "C" {
#endif

// 声明函数 BOOL_clip，用于对布尔类型数据进行截取
NPY_NO_EXPORT void
BOOL_clip(char **args, npy_intp const *dimensions, npy_intp const *steps,
          void *NPY_UNUSED(func));

// 声明函数 BYTE_clip，用于对字节类型数据进行截取
NPY_NO_EXPORT void
BYTE_clip(char **args, npy_intp const *dimensions, npy_intp const *steps,
          void *NPY_UNUSED(func));

// 声明函数 UBYTE_clip，用于对无符号字节类型数据进行截取
NPY_NO_EXPORT void
UBYTE_clip(char **args, npy_intp const *dimensions, npy_intp const *steps,
           void *NPY_UNUSED(func));

// 声明函数 SHORT_clip，用于对短整型数据进行截取
NPY_NO_EXPORT void
SHORT_clip(char **args, npy_intp const *dimensions, npy_intp const *steps,
           void *NPY_UNUSED(func));

// 声明函数 USHORT_clip，用于对无符号短整型数据进行截取
NPY_NO_EXPORT void
USHORT_clip(char **args, npy_intp const *dimensions, npy_intp const *steps,
            void *NPY_UNUSED(func));

// 声明函数 INT_clip，用于对整型数据进行截取
NPY_NO_EXPORT void
INT_clip(char **args, npy_intp const *dimensions, npy_intp const *steps,
         void *NPY_UNUSED(func));

// 声明函数 UINT_clip，用于对无符号整型数据进行截取
NPY_NO_EXPORT void
UINT_clip(char **args, npy_intp const *dimensions, npy_intp const *steps,
          void *NPY_UNUSED(func));

// 声明函数 LONG_clip，用于对长整型数据进行截取
NPY_NO_EXPORT void
LONG_clip(char **args, npy_intp const *dimensions, npy_intp const *steps,
          void *NPY_UNUSED(func));

// 声明函数 ULONG_clip，用于对无符号长整型数据进行截取
NPY_NO_EXPORT void
ULONG_clip(char **args, npy_intp const *dimensions, npy_intp const *steps,
           void *NPY_UNUSED(func));

// 声明函数 LONGLONG_clip，用于对长长整型数据进行截取
NPY_NO_EXPORT void
LONGLONG_clip(char **args, npy_intp const *dimensions, npy_intp const *steps,
              void *NPY_UNUSED(func));

// 声明函数 ULONGLONG_clip，用于对无符号长长整型数据进行截取
NPY_NO_EXPORT void
ULONGLONG_clip(char **args, npy_intp const *dimensions, npy_intp const *steps,
               void *NPY_UNUSED(func));

// 声明函数 HALF_clip，用于对半精度浮点型数据进行截取
NPY_NO_EXPORT void
HALF_clip(char **args, npy_intp const *dimensions, npy_intp const *steps,
          void *NPY_UNUSED(func));

// 声明函数 FLOAT_clip，用于对单精度浮点型数据进行截取
NPY_NO_EXPORT void
FLOAT_clip(char **args, npy_intp const *dimensions, npy_intp const *steps,
           void *NPY_UNUSED(func));

// 声明函数 DOUBLE_clip，用于对双精度浮点型数据进行截取
NPY_NO_EXPORT void
DOUBLE_clip(char **args, npy_intp const *dimensions, npy_intp const *steps,
            void *NPY_UNUSED(func));

// 声明函数 LONGDOUBLE_clip，用于对长双精度浮点型数据进行截取
NPY_NO_EXPORT void
LONGDOUBLE_clip(char **args, npy_intp const *dimensions, npy_intp const *steps,
                void *NPY_UNUSED(func));

// 声明函数 CFLOAT_clip，用于对复数单精度浮点型数据进行截取
NPY_NO_EXPORT void
CFLOAT_clip(char **args, npy_intp const *dimensions, npy_intp const *steps,
            void *NPY_UNUSED(func));

// 声明函数 CDOUBLE_clip，用于对复数双精度浮点型数据进行截取
NPY_NO_EXPORT void
CDOUBLE_clip(char **args, npy_intp const *dimensions, npy_intp const *steps,
             void *NPY_UNUSED(func));

// 声明函数 CLONGDOUBLE_clip，用于对复数长双精度浮点型数据进行截取
NPY_NO_EXPORT void
CLONGDOUBLE_clip(char **args, npy_intp const *dimensions,
                 npy_intp const *steps, void *NPY_UNUSED(func));

// 声明函数 DATETIME_clip，用于对日期时间类型数据进行截取
NPY_NO_EXPORT void
DATETIME_clip(char **args, npy_intp const *dimensions, npy_intp const *steps,
              void *NPY_UNUSED(func));

// 声明函数 TIMEDELTA_clip，用于对时间间隔类型数据进行截取
NPY_NO_EXPORT void
TIMEDELTA_clip(char **args, npy_intp const *dimensions, npy_intp const *steps,
               void *NPY_UNUSED(func));

#ifdef __cplusplus
}
#endif

#endif
```