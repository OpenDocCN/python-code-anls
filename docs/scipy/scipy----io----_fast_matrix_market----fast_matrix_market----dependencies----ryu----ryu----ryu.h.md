# `D:\src\scipysrc\scipy\scipy\io\_fast_matrix_market\fast_matrix_market\dependencies\ryu\ryu\ryu.h`

```
// 如果未定义 RYU_H，则定义 RYU_H，以避免多次包含
#ifndef RYU_H
#define RYU_H

// 如果是 C++ 环境，则使用 extern "C"，以便能够在 C++ 中使用 C 函数
#ifdef __cplusplus
extern "C" {
#endif

// 包含标准整数类型头文件，以便使用 uint32_t 类型
#include <inttypes.h>

// 函数声明：将双精度浮点数转换为字符串表示，使用缓冲区来存储结果
int d2s_buffered_n(double f, char* result);
void d2s_buffered(double f, char* result);
char* d2s(double f);

// 函数声明：将单精度浮点数转换为字符串表示，使用缓冲区来存储结果
int f2s_buffered_n(float f, char* result);
void f2s_buffered(float f, char* result);
char* f2s(float f);

// 函数声明：将双精度浮点数转换为固定精度小数点表示的字符串，使用缓冲区来存储结果
int d2fixed_buffered_n(double d, uint32_t precision, char* result);
void d2fixed_buffered(double d, uint32_t precision, char* result);
char* d2fixed(double d, uint32_t precision);

// 函数声明：将双精度浮点数转换为科学计数法表示的字符串，使用缓冲区来存储结果
int d2exp_buffered_n(double d, uint32_t precision, char* result);
void d2exp_buffered(double d, uint32_t precision, char* result);
char* d2exp(double d, uint32_t precision);

// 如果是 C++ 环境，结束 extern "C" 块
#ifdef __cplusplus
}
#endif

// 结束条件编译指令，结束对 RYU_H 的定义
#endif // RYU_H
```