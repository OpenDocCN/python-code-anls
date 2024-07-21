# `.\pytorch\c10\cuda\driver_api.h`

```
#pragma once
// 包含 CUDA 的头文件
#include <cuda.h>
// 禁用未版本化的函数定义
#define NVML_NO_UNVERSIONED_FUNC_DEFS
// 包含 NVML 的头文件
#include <nvml.h>

// 定义宏 C10_CUDA_DRIVER_CHECK，用于检查 CUDA 驱动函数的返回结果
#define C10_CUDA_DRIVER_CHECK(EXPR)                                        \
  do {                                                                     \
    // 声明 CUresult 类型的错误变量 __err，并执行 EXPR 表达式
    CUresult __err = EXPR;                                                 \
    // 如果返回的结果不是 CUDA_SUCCESS
    if (__err != CUDA_SUCCESS) {                                           \
      const char* err_str;                                                 \
      // 调用 cuGetErrorString_ 函数获取错误信息并存储在 err_str 中
      CUresult get_error_str_err C10_UNUSED =                              \
          c10::cuda::DriverAPI::get()->cuGetErrorString_(__err, &err_str); \
      // 如果获取错误信息失败，则抛出错误信息 "CUDA driver error: unknown error"
      if (get_error_str_err != CUDA_SUCCESS) {                             \
        AT_ERROR("CUDA driver error: unknown error");                      \
      } else {                                                             \
        // 否则，抛出包含具体错误信息的异常 "CUDA driver error: <err_str>"
        AT_ERROR("CUDA driver error: ", err_str);                          \
      }                                                                    \
    }                                                                      \
  } while (0)

// 定义宏 C10_LIBCUDA_DRIVER_API，包含 CUDA 驱动 API 的函数列表
#define C10_LIBCUDA_DRIVER_API(_)   \
  _(cuMemAddressReserve)            \
  _(cuMemRelease)                   \
  _(cuMemMap)                       \
  _(cuMemAddressFree)               \
  _(cuMemSetAccess)                 \
  _(cuMemUnmap)                     \
  _(cuMemCreate)                    \
  _(cuMemGetAllocationGranularity)  \
  _(cuMemExportToShareableHandle)   \
  _(cuMemImportFromShareableHandle) \
  _(cuGetErrorString)

// 定义宏 C10_NVML_DRIVER_API，包含 NVML 驱动 API 的函数列表
#define C10_NVML_DRIVER_API(_)           \
  _(nvmlInit_v2)                         \
  _(nvmlDeviceGetHandleByPciBusId_v2)    \
  _(nvmlDeviceGetNvLinkRemoteDeviceType) \
  _(nvmlDeviceGetNvLinkRemotePciInfo_v2) \
  _(nvmlDeviceGetComputeRunningProcesses)

// 命名空间 c10::cuda 中的结构体 DriverAPI
namespace c10::cuda {

struct DriverAPI {
  // 定义宏 CREATE_MEMBER，用于声明 CUDA 和 NVML 驱动函数指针成员
#define CREATE_MEMBER(name) decltype(&name) name##_;
  C10_LIBCUDA_DRIVER_API(CREATE_MEMBER)    // 声明 CUDA 驱动函数指针成员
  C10_NVML_DRIVER_API(CREATE_MEMBER)       // 声明 NVML 驱动函数指针成员
#undef CREATE_MEMBER

  // 静态成员函数 get()，返回 DriverAPI 的实例指针
  static DriverAPI* get();

  // 静态成员函数 get_nvml_handle()，返回 NVML 的句柄
  static void* get_nvml_handle();
};

} // namespace c10::cuda
```