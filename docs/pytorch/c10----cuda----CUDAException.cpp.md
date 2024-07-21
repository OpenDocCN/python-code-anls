# `.\pytorch\c10\cuda\CUDAException.cpp`

```
// 引入必要的头文件，包括 CUDA 异常处理和相关工具
#include <c10/cuda/CUDAException.h>

#include <c10/cuda/CUDADeviceAssertionHost.h>
#include <c10/util/Exception.h>
#include <cuda_runtime.h>

#include <string>

// 定义命名空间 c10::cuda，包含 CUDA 相关的实现细节
namespace c10::cuda {

// CUDA 错误检查的实现函数，检查 CUDA 错误和设备断言失败情况
void c10_cuda_check_implementation(
    const int32_t err,                  // CUDA 错误码
    const char* filename,               // 发生错误的文件名
    const char* function_name,          // 发生错误的函数名
    const int line_number,              // 发生错误的行号
    const bool include_device_assertions // 是否包含设备断言
) {
  // 将错误码转换为 cudaError_t 类型
  const auto cuda_error = static_cast<cudaError_t>(err);
  // 检查是否有 CUDA 核启动失败，如果需要包含设备断言
  const auto cuda_kernel_failure = include_device_assertions
      ? c10::cuda::CUDAKernelLaunchRegistry::get_singleton_ref().has_failed()
      : false;

  // 如果 CUDA 错误码为 cudaSuccess 并且没有 CUDA 核启动失败，则直接返回
  if (C10_LIKELY(cuda_error == cudaSuccess && !cuda_kernel_failure)) {
    return;
  }

  // 获取并忽略 CUDA 最后的错误
  auto error_unused C10_UNUSED = cudaGetLastError();
  (void)error_unused;

  // 准备错误信息字符串
  std::string check_message;
#ifndef STRIP_ERROR_MESSAGES
  // 添加 CUDA 错误信息到错误消息中
  check_message.append("CUDA error: ");
  check_message.append(cudaGetErrorString(cuda_error));
  // 添加 CUDA 检查后缀
  check_message.append(c10::cuda::get_cuda_check_suffix());
  check_message.append("\n");
  // 如果需要包含设备断言信息，则添加设备端断言信息到错误消息
  if (include_device_assertions) {
    check_message.append(c10_retrieve_device_side_assertion_info());
  } else {
    // 否则，说明为初始化 DSA 处理程序时出现错误
    check_message.append(
        "Device-side assertions were explicitly omitted for this error check; the error probably arose while initializing the DSA handlers.");
  }
#endif

  // 使用 TORCH_CHECK 抛出带有错误消息的异常
  TORCH_CHECK(false, check_message);
}

} // namespace c10::cuda
```