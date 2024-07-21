# `.\pytorch\test\cpp\lazy\test_backend_device.cpp`

```
#include <gtest/gtest.h>  // 包含 Google Test 框架的头文件

#include <sstream>  // 包含用于字符串流操作的头文件

#include <c10/core/Device.h>  // 包含 PyTorch 的设备相关头文件
#include <torch/csrc/lazy/backend/backend_device.h>  // 包含 PyTorch 惰性模块的设备后端头文件
#include <torch/csrc/lazy/ts_backend/ts_backend_impl.h>  // 包含 PyTorch 惰性模块的后端实现头文件
#include <torch/torch.h>  // 包含 PyTorch 的主头文件

namespace torch {
namespace lazy {

TEST(BackendDeviceTest, BackendDeviceType) {  // 定义 Google Test 测试用例 BackendDeviceType
  auto type = BackendDeviceType();  // 创建 BackendDeviceType 对象

  EXPECT_EQ(type.type, 0);  // 断言类型字段为0
  EXPECT_STREQ(type.toString().c_str(), "Unknown");  // 断言类型转换为字符串后为"Unknown"
}

TEST(BackendDeviceTest, Basic1) {  // 定义 Google Test 测试用例 Basic1
  auto device = BackendDevice();  // 创建 BackendDevice 对象

  EXPECT_EQ(device.ordinal(), 0);  // 断言设备序数为0
  if (std::getenv("LTC_TS_CUDA") != nullptr) {  // 检查环境变量是否设置为使用 CUDA
    EXPECT_EQ(device.type(), 1);  // 如果使用 CUDA，断言设备类型为1
    EXPECT_STREQ(device.toString().c_str(), "CUDA0");  // 断言设备类型转换为字符串后为"CUDA0"
  } else {
    EXPECT_EQ(device.type(), 0);  // 如果未使用 CUDA，断言设备类型为0
    EXPECT_STREQ(device.toString().c_str(), "CPU0");  // 断言设备类型转换为字符串后为"CPU0"
  }
}

TEST(BackendDeviceTest, Basic2) {  // 定义 Google Test 测试用例 Basic2
  auto type = std::make_shared<BackendDeviceType>();  // 创建 BackendDeviceType 的 shared_ptr
  type->type = 1;  // 设置类型字段为1
  auto device = BackendDevice(std::move(type), 1);  // 创建 BackendDevice 对象

  EXPECT_EQ(device.type(), 1);  // 断言设备类型为1
  EXPECT_EQ(device.ordinal(), 1);  // 断言设备序数为1
  EXPECT_STREQ(device.toString().c_str(), "Unknown1");  // 断言设备类型转换为字符串后为"Unknown1"
}

TEST(BackendDeviceTest, Basic3) {  // 定义 Google Test 测试用例 Basic3
  struct TestType : public BackendDeviceType {  // 定义 TestType 结构体继承自 BackendDeviceType
    std::string toString() const override {  // 重写 toString 方法
      return "Test";  // 返回固定字符串 "Test"
    }
  };

  auto device = BackendDevice(std::make_shared<TestType>(), 1);  // 使用 TestType 创建 BackendDevice 对象

  EXPECT_EQ(device.type(), 0);  // 断言设备类型为0
  EXPECT_EQ(device.ordinal(), 1);  // 断言设备序数为1
  EXPECT_STREQ(device.toString().c_str(), "Test1");  // 断言设备类型转换为字符串后为"Test1"
}

TEST(BackendDeviceTest, Basic4) {  // 定义 Google Test 测试用例 Basic4
  // 由于 getBackend()->GetDefaultDeviceType() 返回常量指针，调用设置函数看起来不太合理。
  auto default_type = getBackend()->GetDefaultDeviceType();  // 获取默认设备类型
  auto default_ordinal = getBackend()->GetDefaultDeviceOrdinal();  // 获取默认设备序数
  const_cast<BackendImplInterface*>(getBackend())  // 强制转换去掉常量属性
      ->SetDefaultDeviceType(static_cast<int8_t>(c10::kCUDA));  // 设置默认设备类型为 CUDA
  const_cast<BackendImplInterface*>(getBackend())->SetDefaultDeviceOrdinal(1);  // 设置默认设备序数为1

  auto device = BackendDevice();  // 创建 BackendDevice 对象

  EXPECT_EQ(device.type(), 1);  // 断言设备类型为1
  EXPECT_EQ(device.ordinal(), 1);  // 断言设备序数为1
  EXPECT_STREQ(device.toString().c_str(), "CUDA1");  // 断言设备类型转换为字符串后为"CUDA1"

  const_cast<BackendImplInterface*>(getBackend())  // 恢复默认设备类型
      ->SetDefaultDeviceType(default_type->type);
  const_cast<BackendImplInterface*>(getBackend())  // 恢复默认设备序数
      ->SetDefaultDeviceOrdinal(default_ordinal);
}

TEST(BackendDeviceTest, Compare) {  // 定义 Google Test 测试用例 Compare
  auto type = std::make_shared<BackendDeviceType>();  // 创建 BackendDeviceType 的 shared_ptr
  type->type = 1;  // 设置类型字段为1

  auto device1 = BackendDevice(std::make_shared<BackendDeviceType>(), 1);  // 创建多个 BackendDevice 对象
  auto device2 = BackendDevice(std::move(type), 0);
  auto device3 = BackendDevice(std::make_shared<BackendDeviceType>(), 2);
  auto device4 = BackendDevice(std::make_shared<BackendDeviceType>(), 1);

  EXPECT_NE(device1, device2);  // 断言不相等性
  EXPECT_NE(device1, device3);  // 断言不相等性
  EXPECT_EQ(device1, device4);  // 断言相等性
  EXPECT_LT(device1, device2);  // 断言小于关系
  EXPECT_LT(device1, device3);  // 断言小于关系
}

TEST(BackendDeviceTest, Ostream) {  // 定义 Google Test 测试用例 Ostream
  auto device = BackendDevice();  // 创建 BackendDevice 对象
  std::stringstream ss;  // 创建字符串流对象
  ss << device;  // 将设备对象输出到字符串流中

  EXPECT_EQ(device.toString(), ss.str());  // 断言设备对象的字符串表示与字符串流内容相等
}

}  // namespace lazy
}  // namespace torch
TEST(BackendDeviceTest, FromAten) {
  // 创建一个 CPU 设备对象
  auto device = c10::Device(c10::kCPU);
  // 期望抛出 c10::Error 异常，因为没有实现从 Aten 设备到后端设备的转换
  EXPECT_THROW(atenDeviceToBackendDevice(device), c10::Error);

  // 修改设备为 Lazy 类型
  device = c10::Device(c10::kLazy);
#ifndef FBCODE_CAFFE2
  // 在非 FBCODE_CAFFE2 环境中，执行从 Aten 设备到后端设备的转换
  auto backend_device = atenDeviceToBackendDevice(device);
#else
  // 在 FBCODE_CAFFE2 环境中，Lazy Tensor 被禁用，因此期望抛出异常
  // 目前直到在 TensorImpl 中解决非虚拟方法（如 sizes）的问题为止
  EXPECT_THROW(atenDeviceToBackendDevice(device), c10::Error);
#endif // FBCODE_CAFFE2
}

TEST(BackendDeviceTest, ToAten) {
  // 创建一个空的 BackendDevice 对象
  auto device = backendDeviceToAtenDevice(BackendDevice());
  // 期望设备类型为 Lazy
  EXPECT_EQ(device.type(), c10::kLazy);
  // 期望设备具有索引
  EXPECT_TRUE(device.has_index());
  // 期望设备索引为 0
  EXPECT_EQ(device.index(), 0);
}

// TODO(alanwaketan): Update the following test once we have TorchScript backend
// upstreamed.
TEST(BackendDeviceTest, GetBackendDevice1) {
  // 创建一个形状为空的随机张量 tensor
  auto tensor = torch::rand({0, 1, 3, 0});
  // 期望 GetBackendDevice 返回 false
  EXPECT_FALSE(GetBackendDevice(tensor));
}

TEST(BackendDeviceTest, GetBackendDevice2) {
  // 创建两个形状为空的随机张量 tensor1 和 tensor2
  auto tensor1 = torch::rand({0, 1, 3, 0});
  auto tensor2 = torch::rand({0, 1, 3, 0});
  // TODO(alanwaketan): Cover the test case for GetBackendDevice().
  // 期望 GetBackendDevice 返回 false
  EXPECT_FALSE(GetBackendDevice(tensor1, tensor2));
}

} // namespace lazy
} // namespace torch
```