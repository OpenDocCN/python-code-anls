# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\q8gemm_sparse\8x4c1x4-packed-sse2.h`

```
#pragma once
// 预处理命令，表示本头文件只包含一次

#define MR 8
// 定义常量 MR，值为 8

#define COL_BLOCK_SIZE 4
// 定义常量 COL_BLOCK_SIZE，值为 4

#define PACKED_A_BLOCK_SIZE COL_BLOCK_SIZE*MR
// 定义常量 PACKED_A_BLOCK_SIZE，其值为 COL_BLOCK_SIZE 乘以 MR 的结果
```