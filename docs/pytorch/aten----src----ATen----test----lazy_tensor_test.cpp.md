# `.\pytorch\aten\src\ATen\test\lazy_tensor_test.cpp`

```py
// 包含 Google Test 框架的头文件
#include <gtest/gtest.h>

// 包含 ATen 库的头文件
#include <ATen/ATen.h>

// 定义一个函数 LazyTensorTest，用于测试惰性张量的创建
void LazyTensorTest(c10::DispatchKey dispatch_key, at::DeviceType device_type) {
  // 创建一个未定义的张量实现对象，使用指定的分发键、数据类型和设备
  auto tensor_impl =
      c10::make_intrusive<c10::TensorImpl, c10::UndefinedTensorImpl>(
          dispatch_key,
          caffe2::TypeMeta::Make<float>(),
          at::Device(device_type, 0));
  // 创建一个张量对象 t，移动所有权到 t
  at::Tensor t(std::move(tensor_impl));
  // 断言张量 t 的设备与预期设备相匹配
  ASSERT_TRUE(t.device() == at::Device(device_type, 0));
}

// 定义一个 Google Test 测试用例 XlaTensorTest.TestNoStorage
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(XlaTensorTest, TestNoStorage) {
  // 调用 LazyTensorTest 函数测试 XLA 分发键和设备类型
  LazyTensorTest(at::DispatchKey::XLA, at::DeviceType::XLA);
}

// 定义一个 Google Test 测试用例 LazyTensorTest.TestNoStorage
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(LazyTensorTest, TestNoStorage) {
  // 调用 LazyTensorTest 函数测试 Lazy 分发键和设备类型
  LazyTensorTest(at::DispatchKey::Lazy, at::DeviceType::Lazy);
}
```