# `.\pytorch\c10\test\core\DeviceGuard_test.cpp`

```
// 包含 Google Test 框架的头文件
#include <gtest/gtest.h>

// 包含 C10 库的相关头文件
#include <c10/core/DeviceGuard.h>
#include <c10/core/impl/FakeGuardImpl.h>

// 使用 C10 和 C10::impl 命名空间
using namespace c10;
using namespace c10::impl;

// 在这里进行的测试大部分由 InlineDeviceGuard_test 涵盖，但是有一些 DeviceGuard 特定的功能必须进行测试。

// -- DeviceGuard -------------------------------------------------------

// DeviceGuard 类的测试用例
TEST(DeviceGuard, ResetDeviceDifferentDeviceType) {
  // 创建 CUDA 和 HIP 设备的虚拟实现
  FakeGuardImpl<DeviceType::CUDA> cuda_impl;
  FakeGuardImpl<DeviceType::HIP> hip_impl;
  
  // 设置 CUDA 和 HIP 设备的索引
  FakeGuardImpl<DeviceType::CUDA>::setDeviceIndex(0);
  FakeGuardImpl<DeviceType::HIP>::setDeviceIndex(0);
  
  // 创建 DeviceGuard 对象，将当前设备设置为 CUDA 设备
  DeviceGuard g(Device(DeviceType::CUDA, 1), &cuda_impl);
  
  // 重置设备为 HIP 设备
  g.reset_device(Device(DeviceType::HIP, 2), &hip_impl);
  
  // 断言当前 CUDA 设备索引未改变
  ASSERT_EQ(FakeGuardImpl<DeviceType::CUDA>::getDeviceIndex(), 0);
  // 断言当前 HIP 设备索引为设定的值 2
  ASSERT_EQ(FakeGuardImpl<DeviceType::HIP>::getDeviceIndex(), 2);
  // 断言当前 DeviceGuard 对象的设备为 HIP 设备
  ASSERT_EQ(g.current_device(), Device(DeviceType::HIP, 2));
  // 断言 DeviceGuard 对象的原始设备为 HIP 设备
  ASSERT_EQ(g.original_device(), Device(DeviceType::HIP, 0));
}

// -- OptionalDeviceGuard -----------------------------------------------

// OptionalDeviceGuard 类的测试用例
TEST(OptionalDeviceGuard, ResetDeviceDifferentDeviceType) {
  // 创建 CUDA 和 HIP 设备的虚拟实现
  FakeGuardImpl<DeviceType::CUDA> cuda_impl;
  FakeGuardImpl<DeviceType::HIP> hip_impl;
  
  // 设置 CUDA 和 HIP 设备的索引
  FakeGuardImpl<DeviceType::CUDA>::setDeviceIndex(0);
  FakeGuardImpl<DeviceType::HIP>::setDeviceIndex(0);
  
  // 创建 OptionalDeviceGuard 对象
  OptionalDeviceGuard g;
  
  // 重置设备为 CUDA 设备
  g.reset_device(Device(DeviceType::CUDA, 1), &cuda_impl);
  
  // 重置设备为 HIP 设备
  g.reset_device(Device(DeviceType::HIP, 2), &hip_impl);
  
  // 断言当前 CUDA 设备索引未改变
  ASSERT_EQ(FakeGuardImpl<DeviceType::CUDA>::getDeviceIndex(), 0);
  // 断言当前 HIP 设备索引为设定的值 2
  ASSERT_EQ(FakeGuardImpl<DeviceType::HIP>::getDeviceIndex(), 2);
  // 断言当前 OptionalDeviceGuard 对象的当前设备为 HIP 设备（用 make_optional 包装）
  ASSERT_EQ(g.current_device(), make_optional(Device(DeviceType::HIP, 2)));
  // 断言 OptionalDeviceGuard 对象的原始设备为 HIP 设备（用 make_optional 包装）
  ASSERT_EQ(g.original_device(), make_optional(Device(DeviceType::HIP, 0)));
}
```