# `.\pytorch\aten\src\ATen\cuda\nvrtc_stub\ATenNVRTC.h`

```py
#pragma once
// 防止头文件被多次包含

#include <ATen/cuda/ATenCUDAGeneral.h>
// 引入 ATen CUDA 通用头文件

#include <cuda.h>
// 引入 CUDA 运行时头文件

#include <nvrtc.h>
// 引入 NVRTC 头文件

namespace at { namespace cuda {

// ATen 使用 NVRTC 和 Driver API 注意事项

// NOTE [ USE OF NVRTC AND DRIVER API ]
//
// ATen 不直接链接 libnvrtc 或 libcuda，因为它们要求安装 libcuda，
// 但我们希望 GPU 构建在没有安装驱动的机器上也能工作，只要 CUDA 没有初始化。
//
// 在 torch 中，通常使用 CUDA 运行时库，它们可以在没有安装驱动的情况下安装，
// 但有时我们需要使用 Driver API（例如，加载 JIT 编译的代码）。
// 为了实现这一点，我们延迟链接 libcaffe2_nvrtc，该库提供了 at::cuda::NVRTC 结构，
// 包含我们需要的所有 API 函数指针。
//
// 直接调用任何 nvrtc* 或 cu* 函数是错误的。
// 可以使用以下方式调用：
//   detail::getCUDAHooks().nvrtc().cuLoadModule(...)
// 或者
//   globalContext().getNVRTC().cuLoadModule(...)
//
// 如果缺少某个函数，请将其添加到 ATen/cuda/nvrtc_stub/ATenNVRTC.h 中的列表中，
// 并相应地编辑 ATen/cuda/detail/LazyNVRTC.cpp（例如，通过一个存根宏）。

#if !defined(USE_ROCM)

#define AT_FORALL_NVRTC_BASE(_)                  \
  _(nvrtcVersion)                                \
  _(nvrtcAddNameExpression)                      \
  _(nvrtcCreateProgram)                          \
  _(nvrtcDestroyProgram)                         \
  _(nvrtcGetPTXSize)                             \
  _(nvrtcGetPTX)                                 \
  _(nvrtcCompileProgram)                         \
  _(nvrtcGetErrorString)                         \
  _(nvrtcGetProgramLogSize)                      \
  _(nvrtcGetProgramLog)                          \
  _(nvrtcGetLoweredName)                         \
  _(cuModuleLoadData)                            \
  _(cuModuleLoadDataEx)                          \
  _(cuModuleGetFunction)                         \
  _(cuOccupancyMaxActiveBlocksPerMultiprocessor) \
  _(cuGetErrorString)                            \
  _(cuLaunchKernel)                              \
  _(cuLaunchCooperativeKernel)                   \
  _(cuCtxGetCurrent)                             \
  _(cuCtxSetCurrent)                             \
  _(cuModuleUnload)                              \
  _(cuDevicePrimaryCtxGetState)                  \
  _(cuDevicePrimaryCtxRetain)                    \
  _(cuLinkCreate)                                \
  _(cuLinkAddData)                               \
  _(cuLinkComplete)                              \
  _(cuFuncSetAttribute)                          \
  _(cuFuncGetAttribute)                          \

#if defined(CUDA_VERSION) && CUDA_VERSION >= 12000
#define AT_FORALL_NVRTC_EXTENDED(_)              \
  AT_FORALL_NVRTC_BASE(_)                        \
  _(cuTensorMapEncodeTiled)
#else
#define AT_FORALL_NVRTC_EXTENDED(_)              \
  AT_FORALL_NVRTC_BASE(_)
#endif

#if defined(CUDA_VERSION) && CUDA_VERSION >= 11010
`
# 定义了一个宏 AT_FORALL_NVRTC，根据条件不同扩展宏内容
#define AT_FORALL_NVRTC(_) \
  AT_FORALL_NVRTC_EXTENDED(_)  \
  _(nvrtcGetCUBINSize)     \
  _(nvrtcGetCUBIN)
#else
# 如果不满足上述条件，则只使用扩展宏内容
#define AT_FORALL_NVRTC(_) \
  AT_FORALL_NVRTC_EXTENDED(_)
#endif

#else

// 注意 [ ATen NVRTC Stub and HIP ]
//
// ATen的NVRTC存根库，caffe2_nvrtc，动态加载NVRTC和驱动程序API。尽管前者尚不支持HIP，但后者支持且需要（例如，在CUDAHooks::getDeviceWithPrimaryContext()中由tensor.pin_memory()使用）。
//
// 下面的宏从上面完整列表中剔除了HIP上不支持的某些操作。
//
// HIP没有
//   cuGetErrorString  （映射到不起作用的hipGetErrorString___）
//
// 从ROCm 3.5开始，HIP将hipOccupancyMaxActiveBlocksPerMultiprocessor重命名为hipModuleOccupancyMaxActiveBlocksPerMultiprocessor。
#if TORCH_HIP_VERSION < 305
# 如果HIP版本低于305，则使用hipOccupancyMaxActiveBlocksPerMultiprocessor
#define HIPOCCUPANCYMAXACTIVEBLOCKSPERMULTIPROCESSOR hipOccupancyMaxActiveBlocksPerMultiprocessor
#else
# 否则使用cuOccupancyMaxActiveBlocksPerMultiprocessor
#define HIPOCCUPANCYMAXACTIVEBLOCKSPERMULTIPROCESSOR cuOccupancyMaxActiveBlocksPerMultiprocessor
#endif

# 定义宏 AT_FORALL_NVRTC，列出NVRTC相关的各种API函数
#define AT_FORALL_NVRTC(_)                        \
  _(nvrtcVersion)                                 \
  _(nvrtcCreateProgram)                           \
  _(nvrtcAddNameExpression)                       \
  _(nvrtcDestroyProgram)                          \
  _(nvrtcGetPTXSize)                              \
  _(nvrtcGetPTX)                                  \
  _(cuModuleLoadData)                             \
  _(cuModuleGetFunction)                          \
  _(HIPOCCUPANCYMAXACTIVEBLOCKSPERMULTIPROCESSOR) \
  _(nvrtcGetErrorString)                          \
  _(nvrtcGetProgramLogSize)                       \
  _(nvrtcGetProgramLog)                           \
  _(cuLaunchKernel)                               \
  _(nvrtcCompileProgram)                          \
  _(cuCtxGetCurrent)                              \
  _(nvrtcGetLoweredName)                          \
  _(cuModuleUnload)                               \
  _(cuDevicePrimaryCtxGetState)

#endif

# 声明一个名为NVRTC的结构体，其成员是所有定义的NVRTC API函数指针
extern "C" typedef struct NVRTC {
# 通过宏展开，为结构体定义成员变量
#define CREATE_MEMBER(name) decltype(&name) name;
  AT_FORALL_NVRTC(CREATE_MEMBER)
#undef CREATE_MEMBER
} NVRTC;

# extern "C"声明一个C语言风格的函数load_nvrtc()，返回类型为NVRTC指针
extern "C" TORCH_CUDA_CPP_API NVRTC* load_nvrtc();
}} // at::cuda
```