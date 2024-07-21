# `.\pytorch\aten\src\ATen\cuda\CUDASparse.h`

```py
#pragma once
// 用于确保头文件只被包含一次

#include <ATen/cuda/CUDAContext.h>
// 包含 CUDA 的上下文头文件

#if defined(USE_ROCM)
#include <hipsparse/hipsparse-version.h>
// 如果是 ROCm 平台，包含 hipsparse 版本信息头文件
#define HIPSPARSE_VERSION ((hipsparseVersionMajor*100000) + (hipsparseVersionMinor*100) + hipsparseVersionPatch)
#endif

// cuSparse Generic API added in CUDA 10.1
// Windows support added in CUDA 11.0
#if defined(CUDART_VERSION) && defined(CUSPARSE_VERSION) && ((CUSPARSE_VERSION >= 10300) || (CUSPARSE_VERSION >= 11000 && defined(_WIN32)))
// 如果定义了 CUDART_VERSION 和 CUSPARSE_VERSION，且满足条件则定义为 1，否则为 0
#define AT_USE_CUSPARSE_GENERIC_API() 1
#else
#define AT_USE_CUSPARSE_GENERIC_API() 0
#endif

// cuSparse Generic API descriptor pointers were changed to const in CUDA 12.0
#if defined(CUDART_VERSION) && defined(CUSPARSE_VERSION) && \
    (CUSPARSE_VERSION < 12000)
// 如果定义了 CUDART_VERSION 和 CUSPARSE_VERSION，并且 CUSPARSE_VERSION 小于 12000，则定义为 1，否则为 0
#define AT_USE_CUSPARSE_NON_CONST_DESCRIPTORS() 1
#else
#define AT_USE_CUSPARSE_NON_CONST_DESCRIPTORS() 0
#endif

#if defined(CUDART_VERSION) && defined(CUSPARSE_VERSION) && \
    (CUSPARSE_VERSION >= 12000)
// 如果定义了 CUDART_VERSION 和 CUSPARSE_VERSION，并且 CUSPARSE_VERSION 大于等于 12000，则定义为 1，否则为 0
#define AT_USE_CUSPARSE_CONST_DESCRIPTORS() 1
#else
#define AT_USE_CUSPARSE_CONST_DESCRIPTORS() 0
#endif

#if defined(USE_ROCM)
// hipSparse const API added in v2.4.0
#if HIPSPARSE_VERSION >= 200400
// 如果定义了 USE_ROCM，并且 HIPSPARSE_VERSION 大于等于 200400，则定义为 1，否则根据条件定义为 0
#define AT_USE_HIPSPARSE_CONST_DESCRIPTORS() 1
#define AT_USE_HIPSPARSE_NON_CONST_DESCRIPTORS() 0
#define AT_USE_HIPSPARSE_GENERIC_API() 1
#else
#define AT_USE_HIPSPARSE_CONST_DESCRIPTORS() 0
#define AT_USE_HIPSPARSE_NON_CONST_DESCRIPTORS() 1
#define AT_USE_HIPSPARSE_GENERIC_API() 1
#endif
#else // USE_ROCM
#define AT_USE_HIPSPARSE_CONST_DESCRIPTORS() 0
#define AT_USE_HIPSPARSE_NON_CONST_DESCRIPTORS() 0
#define AT_USE_HIPSPARSE_GENERIC_API() 0
#endif // USE_ROCM

// cuSparse Generic API spsv function was added in CUDA 11.3.0
#if defined(CUDART_VERSION) && defined(CUSPARSE_VERSION) && (CUSPARSE_VERSION >= 11500)
// 如果定义了 CUDART_VERSION 和 CUSPARSE_VERSION，并且 CUSPARSE_VERSION 大于等于 11500，则定义为 1，否则为 0
#define AT_USE_CUSPARSE_GENERIC_SPSV() 1
#else
#define AT_USE_CUSPARSE_GENERIC_SPSV() 0
#endif

// cuSparse Generic API spsm function was added in CUDA 11.3.1
#if defined(CUDART_VERSION) && defined(CUSPARSE_VERSION) && (CUSPARSE_VERSION >= 11600)
// 如果定义了 CUDART_VERSION 和 CUSPARSE_VERSION，并且 CUSPARSE_VERSION 大于等于 11600，则定义为 1，否则为 0
#define AT_USE_CUSPARSE_GENERIC_SPSM() 1
#else
#define AT_USE_CUSPARSE_GENERIC_SPSM() 0
#endif

// cuSparse Generic API sddmm function was added in CUDA 11.2.1 (cuSparse version 11400)
#if defined(CUDART_VERSION) && defined(CUSPARSE_VERSION) && (CUSPARSE_VERSION >= 11400)
// 如果定义了 CUDART_VERSION 和 CUSPARSE_VERSION，并且 CUSPARSE_VERSION 大于等于 11400，则定义为 1，否则为 0
#define AT_USE_CUSPARSE_GENERIC_SDDMM() 1
#else
#define AT_USE_CUSPARSE_GENERIC_SDDMM() 0
#endif

// BSR triangular solve functions were added in hipSPARSE 1.11.2 (ROCm 4.5.0)
#if defined(CUDART_VERSION) || defined(USE_ROCM)
// 如果定义了 CUDART_VERSION 或者 USE_ROCM，则定义为 1，否则为 0
#define AT_USE_HIPSPARSE_TRIANGULAR_SOLVE() 1
#else
#define AT_USE_HIPSPARSE_TRIANGULAR_SOLVE() 0
#endif
```