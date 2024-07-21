# `.\pytorch\c10\test\core\Device_test.cpp`

```py
// 包含 Google Test 框架的头文件
#include <gtest/gtest.h>

// 包含 PyTorch C++ 库中与设备相关的头文件
#include <c10/core/Device.h>
#include <c10/util/Exception.h>

// -- Device -------------------------------------------------------

// 结构体，用于测试预期设备结果
struct ExpectedDeviceTestResult {
  std::string device_string;  // 设备字符串表示
  c10::DeviceType device_type;  // 设备类型
  c10::DeviceIndex device_index;  // 设备索引
};

// 设备测试用例，测试设备的基本构造
TEST(DeviceTest, BasicConstruction) {
  // 有效设备列表，包括设备字符串、设备类型和设备索引
  std::vector<ExpectedDeviceTestResult> valid_devices = {
      {"cpu", c10::DeviceType::CPU, -1},
      {"cuda", c10::DeviceType::CUDA, -1},
      {"cpu:0", c10::DeviceType::CPU, 0},
      {"cuda:0", c10::DeviceType::CUDA, 0},
      {"cuda:1", c10::DeviceType::CUDA, 1},
  };

  // 无效设备字符串列表，用于测试异常情况
  std::vector<std::string> invalid_device_strings = {
      "cpu:x",
      "cpu:foo",
      "cuda:cuda",
      "cuda:",
      "cpu:0:0",
      "cpu:0:",
      "cpu:-1",
      "::",
      ":",
      "cpu:00",
      "cpu:01"
  };

  // 遍历测试有效设备列表
  for (auto& ds : valid_devices) {
    // 创建设备对象并进行断言验证
    c10::Device d(ds.device_string);
    ASSERT_EQ(d.type(), ds.device_type)
        << "Device String: " << ds.device_string;
    ASSERT_EQ(d.index(), ds.device_index)
        << "Device String: " << ds.device_string;
  }

  // Lambda 函数，用于创建设备对象并验证抛出异常
  auto make_device = [](const std::string& ds) { return c10::Device(ds); };

  // 遍历测试无效设备字符串列表
  for (auto& ds : invalid_device_strings) {
    // 预期抛出异常，并记录设备字符串信息
    EXPECT_THROW(make_device(ds), c10::Error) << "Device String: " << ds;
  }
}
```