# `.\pytorch\aten\src\ATen\test\cuda_device_test.cpp`

```
# 包含 Google Test 框架的头文件，用于进行单元测试
#include <gtest/gtest.h>

# 包含 CUDA 相关的头文件，用于 CUDA 上下文和设备操作
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDADevice.h>

# 定义名为 CudaDeviceTest 的测试案例类，测试 CUDA 设备相关功能
TEST(CudaDeviceTest, getDeviceFromPtr_fails_with_host_memory) {
  # 如果 CUDA 不可用，则退出测试
  if (!at::cuda::is_available()) {
    return;
  }

  # 创建一个整型变量 dummy，用于测试
  int dummy = 0;

  # 使用断言 ASSERT_THROW 测试获取指针所在设备的函数在使用主机内存时是否抛出异常
  ASSERT_THROW(at::cuda::getDeviceFromPtr(&dummy), c10::Error);
}
```