# `.\pytorch\c10\cuda\impl\CUDATest.cpp`

```
// 包含 CUDA 库的测试文件，用于确认 CUDA 库是否可用

#include <c10/cuda/CUDAException.h>     // 包含 CUDA 异常处理的头文件
#include <c10/cuda/impl/CUDATest.h>    // 包含 CUDA 测试的头文件

#include <cuda_runtime.h>              // 包含 CUDA 运行时的头文件

namespace c10::cuda::impl {

// 检查系统是否存在 CUDA GPU
bool has_cuda_gpu() {
  int count = 0;                      // 初始化 GPU 数量为 0
  C10_CUDA_IGNORE_ERROR(cudaGetDeviceCount(&count));  // 忽略 CUDA 错误并获取 GPU 数量

  return count != 0;                  // 返回 GPU 数量是否不为 0
}

// 执行 CUDA 测试，返回当前 CUDA 设备编号
int c10_cuda_test() {
  int r = 0;                          // 初始化返回值为 0
  if (has_cuda_gpu()) {               // 如果系统存在 CUDA GPU
    C10_CUDA_CHECK(cudaGetDevice(&r));  // 获取当前 CUDA 设备的编号，并检查 CUDA 错误
  }
  return r;                           // 返回 CUDA 设备的编号
}

// 此函数不对外部导出
int c10_cuda_private_test() {
  return 2;                           // 返回固定值 2，用于私有 CUDA 测试
}

} // namespace c10::cuda::impl
```