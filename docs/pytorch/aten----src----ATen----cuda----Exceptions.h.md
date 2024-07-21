# `.\pytorch\aten\src\ATen\cuda\Exceptions.h`

```
#pragma once
    # 使用宏定义 TORCH_CHECK 检查 __err 是否等于 CUBLAS_STATUS_SUCCESS，
    # 如果不等于，则抛出带有错误信息的异常
    TORCH_CHECK(__err == CUBLAS_STATUS_SUCCESS,                 \
                "CUDA error: ",                                 \
                at::cuda::blas::_cublasGetErrorEnum(__err),     \
                " when calling `" #EXPR "`");                   \
  } while (0)
// 定义一个函数原型，用于获取 cuSPARSE 库中的错误信息
const char *cusparseGetErrorString(cusparseStatus_t status);

// 定义一个宏 TORCH_CUDASPARSE_CHECK，用于检查 cuSPARSE 函数调用的返回状态，如果不是成功状态则输出错误信息
#define TORCH_CUDASPARSE_CHECK(EXPR)                            \
  do {                                                          \
    cusparseStatus_t __err = EXPR;                              \
    TORCH_CHECK(__err == CUSPARSE_STATUS_SUCCESS,               \
                "CUDA error: ",                                 \
                cusparseGetErrorString(__err),                  \
                " when calling `" #EXPR "`");                   \
  } while (0)

// 当 CUDA 运行时版本定义存在时，进入命名空间 at::cuda::solver
#ifdef CUDART_VERSION

namespace at::cuda::solver {
// 定义一个函数原型，用于获取 cuSOLVER 库中的错误信息
C10_EXPORT const char* cusolverGetErrorMessage(cusolverStatus_t status);

// 定义一个字符串常量，提供使用其他支持的线性代数库的建议信息
constexpr const char* _cusolver_backend_suggestion =            \
  "If you keep seeing this error, you may use "                 \
  "`torch.backends.cuda.preferred_linalg_library()` to try "    \
  "linear algebra operators with other supported backends. "    \
  "See https://pytorch.org/docs/stable/backends.html#torch.backends.cuda.preferred_linalg_library";

} // namespace at::cuda::solver

// 根据 CUDA 版本和 cuSOLVER 返回的错误码进行检查和错误处理
#define TORCH_CUSOLVER_CHECK(EXPR)                                      \
  do {                                                                  \
    cusolverStatus_t __err = EXPR;                                      \
    if ((CUDA_VERSION < 11500 &&                                        \
         __err == CUSOLVER_STATUS_EXECUTION_FAILED) ||                  \
        (CUDA_VERSION >= 11500 &&                                       \
         __err == CUSOLVER_STATUS_INVALID_VALUE)) {                     \
      TORCH_CHECK_LINALG(                                               \
          false,                                                        \
          "cusolver error: ",                                           \
          at::cuda::solver::cusolverGetErrorMessage(__err),             \
          ", when calling `" #EXPR "`",                                 \
          ". This error may appear if the input matrix contains NaN. ", \
          at::cuda::solver::_cusolver_backend_suggestion);              \
    } else {                                                            \
      TORCH_CHECK(                                                      \
          __err == CUSOLVER_STATUS_SUCCESS,                             \
          "cusolver error: ",                                           \
          at::cuda::solver::cusolverGetErrorMessage(__err),             \
          ", when calling `" #EXPR "`. ",                               \
          at::cuda::solver::_cusolver_backend_suggestion);              \
    }                                                                   \
  } while (0)

#else
#ifdef TORCH_CUSOLVER_CHECK(EXPR) EXPR
#endif

#ifdef AT_CUDA_CHECK(EXPR) C10_CUDA_CHECK(EXPR)
    # 如果 NVRTC 编译过程中出现错误
    if (__err != NVRTC_SUCCESS) {
      # 如果错误码不是7（NVRTC_ERROR_BUILTIN_OPERATION_FAILURE）
      if (static_cast<int>(__err) != 7) {
        # 抛出包含具体错误信息的异常，使用全局上下文获取错误字符串
        AT_ERROR("CUDA NVRTC error: ", at::globalContext().getNVRTC().nvrtcGetErrorString(__err));
      } else {
        # 如果错误码是7，表示内置操作失败，抛出相应异常
        AT_ERROR("CUDA NVRTC error: NVRTC_ERROR_BUILTIN_OPERATION_FAILURE");
      }
    }
  } while (0)
```