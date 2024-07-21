# `.\pytorch\aten\src\ATen\native\quantized\cpu\qnnpack\src\q8gemm_sparse\8x4c1x4-dq-packedA-sse2.c`

```py
// 定义宏 KERNEL_NAME 为 pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA_w32__sse2
#define KERNEL_NAME pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA_w32__sse2
// 定义宏 W_INDEX_DTYPE 为 uint32_t
#define W_INDEX_DTYPE uint32_t
// 包含头文件 "8x4c1x4-dq-packedA-sse2.h"
#include "8x4c1x4-dq-packedA-sse2.h"
// 取消宏定义 KERNEL_NAME
#undef KERNEL_NAME
// 取消宏定义 W_INDEX_DTYPE
#undef W_INDEX_DTYPE

// 定义宏 KERNEL_NAME 为 pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA_w16__sse2
#define KERNEL_NAME pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA_w16__sse2
// 定义宏 W_INDEX_DTYPE 为 uint16_t
#define W_INDEX_DTYPE uint16_t
// 包含头文件 "8x4c1x4-dq-packedA-sse2.h"
#include "8x4c1x4-dq-packedA-sse2.h"
// 取消宏定义 KERNEL_NAME
#undef KERNEL_NAME
// 取消宏定义 W_INDEX_DTYPE
#undef W_INDEX_DTYPE

// 定义宏 KERNEL_NAME 为 pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA_w8__sse2
#define KERNEL_NAME pytorch_q8gemm_dq_sparse_1x4_ukernel_8x4_packedA_w8__sse2
// 定义宏 W_INDEX_DTYPE 为 uint8_t
#define W_INDEX_DTYPE uint8_t
// 包含头文件 "8x4c1x4-dq-packedA-sse2.h"
#include "8x4c1x4-dq-packedA-sse2.h"
// 取消宏定义 KERNEL_NAME
#undef KERNEL_NAME
// 取消宏定义 W_INDEX_DTYPE
#undef W_INDEX_DTYPE
```