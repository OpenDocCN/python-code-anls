# `bitsandbytes\csrc\cpu_ops.h`

```py
// 防止头文件重复包含
#ifndef BITSANDBYTES_CPU_OPS_H
#define BITSANDBYTES_CPU_OPS_H

// 包含必要的头文件
#include <iostream>
#include <stdio.h>

// 声明量化函数，将浮点数数组量化为无符号字符数组
void quantize_cpu(float *code, float *A, float *absmax, unsigned char *out, long long blocksize, long long n);
// 声明反量化函数，将无符号字符数组反量化为浮点数数组
void dequantize_cpu(float *code, unsigned char *A, float *absmax, float *out, long long blocksize, long long n);

// 结束头文件防止重复包含
#endif
```