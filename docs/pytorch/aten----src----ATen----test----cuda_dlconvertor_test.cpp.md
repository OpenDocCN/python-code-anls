# `.\pytorch\aten\src\ATen\test\cuda_dlconvertor_test.cpp`

```py
// 包含 CUDA 相关头文件
#include <cuda.h>
#include <cuda_runtime.h>

// 包含 Google 测试框架头文件
#include <gtest/gtest.h>

// 包含 ATen（PyTorch C++ 前端）头文件
#include <ATen/ATen.h>
#include <ATen/DLConvertor.h>
#include <ATen/cuda/CUDAConfig.h>
#include <ATen/cuda/CUDAContext.h>

// 使用 ATen 命名空间
using namespace at;

// 定义测试用例 TestDlconvertor，测试 CUDA 下的 DLConvertor 功能
TEST(TestDlconvertor, TestDlconvertorCUDA) {
  // 设置随机种子
  manual_seed(123);

  // 创建一个在 CUDA 上的随机张量 a
  Tensor a = rand({3, 4}, at::kCUDA);
  // 将 Tensor a 转换为 DLManagedTensor 结构
  DLManagedTensor* dlMTensor = toDLPack(a);

  // 从 DLManagedTensor 结构恢复张量 b
  Tensor b = fromDLPack(dlMTensor);

  // 断言张量 a 和 b 在数值上相等
  ASSERT_TRUE(a.equal(b));
}

// 定义测试用例 TestDlconvertorNoStridesCUDA，测试 CUDA 下的 DLConvertor 功能（不包含步长）
TEST(TestDlconvertor, TestDlconvertorNoStridesCUDA) {
  // 设置随机种子
  manual_seed(123);

  // 创建一个在 CUDA 上的随机张量 a
  Tensor a = rand({3, 4}, at::kCUDA);
  // 将 Tensor a 转换为 DLManagedTensor 结构
  DLManagedTensor* dlMTensor = toDLPack(a);
  // 将 DLManagedTensor 结构的步长设置为 nullptr
  dlMTensor->dl_tensor.strides = nullptr;

  // 从 DLManagedTensor 结构恢复张量 b
  Tensor b = fromDLPack(dlMTensor);

  // 断言张量 a 和 b 在数值上相等
  ASSERT_TRUE(a.equal(b));
}

// 定义测试用例 TestDlconvertorCUDAHIP，测试是否正确区分 CUDA 和 HIP（AMD ROCm）下的 DLConvertor 功能
TEST(TestDlconvertor, TestDlconvertorCUDAHIP) {
  // 如果 CUDA 不可用，则返回
  if (!at::cuda::is_available())
    return;

  // 设置随机种子
  manual_seed(123);

  // 创建一个在 CUDA 上的随机张量 a
  Tensor a = rand({3, 4}, at::kCUDA);
  // 将 Tensor a 转换为 DLManagedTensor 结构
  DLManagedTensor* dlMTensor = toDLPack(a);

  // 根据编译时是否启用 HIP（AMD ROCm），进行断言
#if AT_ROCM_ENABLED()
  // 如果启用了 ROCm，断言 DLManagedTensor 的设备类型为 kDLROCM
  ASSERT_TRUE(dlMTensor->dl_tensor.device.device_type == DLDeviceType::kDLROCM);
#else
  // 如果未启用 ROCm，断言 DLManagedTensor 的设备类型为 kDLCUDA
  ASSERT_TRUE(dlMTensor->dl_tensor.device.device_type == DLDeviceType::kDLCUDA);
#endif

  // 从 DLManagedTensor 结构恢复张量 b
  Tensor b = fromDLPack(dlMTensor);

  // 断言张量 a 和 b 在数值上相等
  ASSERT_TRUE(a.equal(b));
}
```