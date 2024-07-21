# `.\pytorch\aten\src\ATen\test\xpu_device_test.cpp`

```
#include <gtest/gtest.h>  // 引入 Google Test 框架的头文件

#include <ATen/xpu/XPUContext.h>  // 引入 ATen XPU 相关的头文件
#include <ATen/xpu/XPUDevice.h>   // 引入 ATen XPU 设备相关的头文件
#include <torch/torch.h>          // 引入 PyTorch 的头文件

TEST(XpuDeviceTest, getDeviceProperties) {
  EXPECT_EQ(at::xpu::is_available(), torch::xpu::is_available());  // 检查是否 XPU 可用

  if (!at::xpu::is_available()) {  // 如果 XPU 不可用，则直接返回
    return;
  }

  c10::xpu::DeviceProp* cur_device_prop = at::xpu::getCurrentDeviceProperties();  // 获取当前设备属性
  c10::xpu::DeviceProp* device_prop = at::xpu::getDeviceProperties(0);            // 获取指定设备属性

  EXPECT_EQ(cur_device_prop->name, device_prop->name);                // 检查设备名称是否一致
  EXPECT_EQ(cur_device_prop->platform_name, device_prop->platform_name);  // 检查平台名称是否一致
  EXPECT_EQ(cur_device_prop->gpu_eu_count, device_prop->gpu_eu_count);  // 检查 GPU EU 数量是否一致
}

TEST(XpuDeviceTest, getDeviceFromPtr) {
  if (!at::xpu::is_available()) {  // 如果 XPU 不可用，则直接返回
    return;
  }

  sycl::device& raw_device = at::xpu::get_raw_device(0);  // 获取原始设备对象
  void* ptr = sycl::malloc_device(8, raw_device, at::xpu::get_device_context());  // 在设备上分配内存

  at::Device device = at::xpu::getDeviceFromPtr(ptr);  // 根据指针获取设备对象
  sycl::free(ptr, at::xpu::get_device_context());      // 释放设备上的内存
  EXPECT_EQ(device.index(), 0);                       // 检查设备索引是否正确
  EXPECT_EQ(device.type(), at::kXPU);                  // 检查设备类型是否为 XPU

  int dummy = 0;
  ASSERT_THROW(at::xpu::getDeviceFromPtr(&dummy), c10::Error);  // 预期抛出异常，因为指针无效
}

TEST(XpuDeviceTest, getGlobalIdxFromDevice) {
  if (!at::xpu::is_available()) {  // 如果 XPU 不可用，则直接返回
    return;
  }

  int target_device = 0;
  auto global_index = at::xpu::getGlobalIdxFromDevice(target_device);  // 获取全局设备索引
  auto devices = sycl::device::get_devices();                          // 获取所有设备列表
  EXPECT_EQ(devices[global_index], at::xpu::get_raw_device(target_device));  // 检查全局索引对应的设备是否正确

  void* ptr = sycl::malloc_device(8, devices[global_index], at::xpu::get_device_context());  // 在设备上分配内存
  at::Device device = at::xpu::getDeviceFromPtr(ptr);  // 根据指针获取设备对象
  sycl::free(ptr, at::xpu::get_device_context());      // 释放设备上的内存
  EXPECT_EQ(device.index(), target_device);            // 检查设备索引是否正确
  EXPECT_EQ(device.type(), at::kXPU);                  // 检查设备类型是否为 XPU

  if (at::xpu::device_count() == 1) {  // 如果只有一个设备，直接返回
    return;
  }
  // 测试最后一个设备
  target_device = at::xpu::device_count() - 1;
  global_index = at::xpu::getGlobalIdxFromDevice(target_device);  // 获取全局设备索引
  EXPECT_EQ(devices[global_index], at::xpu::get_raw_device(target_device));  // 检查全局索引对应的设备是否正确

  target_device = at::xpu::device_count();
  ASSERT_THROW(at::xpu::getGlobalIdxFromDevice(target_device), c10::Error);  // 预期抛出异常，因为设备索引越界
}
```