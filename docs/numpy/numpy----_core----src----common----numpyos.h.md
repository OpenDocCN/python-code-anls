# `.\numpy\numpy\_core\src\common\numpyos.h`

```
#ifndef NUMPY_CORE_SRC_COMMON_NPY_NUMPYOS_H_
#define NUMPY_CORE_SRC_COMMON_NPY_NUMPYOS_H_

#ifdef __cplusplus
extern "C" {
#endif

// 定义了以下函数的声明，这些函数实现在 NumPy 源码中，用于处理 ASCII 字符串和数字转换

// 将 double 类型的 val 格式化成 ASCII 字符串，存储到 buffer 中，格式由 format 指定，decimal 是小数位数
NPY_NO_EXPORT char*
NumPyOS_ascii_formatd(char *buffer, size_t buf_size,
                      const char *format,
                      double val, int decimal);

// 将 float 类型的 val 格式化成 ASCII 字符串，存储到 buffer 中，格式由 format 指定，decimal 是小数位数
NPY_NO_EXPORT char*
NumPyOS_ascii_formatf(char *buffer, size_t buf_size,
                      const char *format,
                      float val, int decimal);

// 将 long double 类型的 val 格式化成 ASCII 字符串，存储到 buffer 中，格式由 format 指定，decimal 是小数位数
NPY_NO_EXPORT char*
NumPyOS_ascii_formatl(char *buffer, size_t buf_size,
                      const char *format,
                      long double val, int decimal);

// 将 ASCII 字符串 s 转换为 double 类型的数值
NPY_NO_EXPORT double
NumPyOS_ascii_strtod(const char *s, char** endptr);

// 将 ASCII 字符串 s 转换为 long double 类型的数值
NPY_NO_EXPORT long double
NumPyOS_ascii_strtold(const char *s, char** endptr);

// 从文件 fp 中读取 double 类型的数值，存储到 value 中，返回成功读取的标志
NPY_NO_EXPORT int
NumPyOS_ascii_ftolf(FILE *fp, double *value);

// 从文件 fp 中读取 long double 类型的数值，存储到 value 中，返回成功读取的标志
NPY_NO_EXPORT int
NumPyOS_ascii_ftoLf(FILE *fp, long double *value);

// 检查字符 c 是否为空白字符
NPY_NO_EXPORT int
NumPyOS_ascii_isspace(int c);

// 检查字符 c 是否为字母
NPY_NO_EXPORT int
NumPyOS_ascii_isalpha(char c);

// 检查字符 c 是否为数字
NPY_NO_EXPORT int
NumPyOS_ascii_isdigit(char c);

// 检查字符 c 是否为字母或数字
NPY_NO_EXPORT int
NumPyOS_ascii_isalnum(char c);

// 检查字符 c 是否为小写字母
NPY_NO_EXPORT int
NumPyOS_ascii_islower(char c);

// 检查字符 c 是否为大写字母
NPY_NO_EXPORT int
NumPyOS_ascii_isupper(char c);

// 将大写字母转换为小写字母
NPY_NO_EXPORT int
NumPyOS_ascii_tolower(char c);

/* 将字符串 str 转换为指定 base 进制的长长整型数值 */
NPY_NO_EXPORT npy_longlong
NumPyOS_strtoll(const char *str, char **endptr, int base);

/* 将字符串 str 转换为指定 base 进制的无符号长长整型数值 */
NPY_NO_EXPORT npy_ulonglong
NumPyOS_strtoull(const char *str, char **endptr, int base);

#ifdef __cplusplus
}
#endif

#endif  /* NUMPY_CORE_SRC_COMMON_NPY_NUMPYOS_H_ */
```