# `.\pytorch\test\cpp\api\tensor_options.cpp`

```
// 引入 Google Test 框架的头文件
#include <gtest/gtest.h>
// 引入 Torch 的测试支持文件
#include <test/cpp/api/support.h>

// 引入 Torch 主要头文件
#include <torch/torch.h>

// 引入标准库头文件
#include <string>
#include <vector>

// 使用 at 命名空间
using namespace at;
// 使用 torch::test 命名空间
using namespace torch::test;

// 定义宏，用于确保选项的设备、索引、数据类型和布局符合预期
#define REQUIRE_OPTIONS(device_, index_, type_, layout_)                  \
  ASSERT_EQ(options.device().type(), Device((device_), (index_)).type()); \
  ASSERT_TRUE(                                                            \
      options.device().index() == Device((device_), (index_)).index());   \
  ASSERT_EQ(options.dtype(), (type_));                                    \
  ASSERT_TRUE(options.layout() == (layout_))

// 定义宏，用于确保张量选项的设备、索引、标量类型和布局符合预期
#define REQUIRE_TENSOR_OPTIONS(device_, index_, type_, layout_)            \
  ASSERT_EQ(tensor.device().type(), Device((device_), (index_)).type());   \
  ASSERT_EQ(tensor.device().index(), Device((device_), (index_)).index()); \
  ASSERT_EQ(tensor.scalar_type(), (type_));                                \
  ASSERT_TRUE(tensor.options().layout() == (layout_))

// 测试用例，验证默认张量选项是否正确
TEST(TensorOptionsTest, DefaultsToTheRightValues) {
  TensorOptions options;
  REQUIRE_OPTIONS(kCPU, -1, kFloat, kStrided);
}

// 测试用例，验证工具函数返回正确的张量选项
TEST(TensorOptionsTest, UtilityFunctionsReturnTheRightTensorOptions) {
  auto options = dtype(kInt);
  REQUIRE_OPTIONS(kCPU, -1, kInt, kStrided);

  options = layout(kSparse);
  REQUIRE_OPTIONS(kCPU, -1, kFloat, kSparse);

  options = device({kCUDA, 1});
  REQUIRE_OPTIONS(kCUDA, 1, kFloat, kStrided);

  options = device_index(1);
  REQUIRE_OPTIONS(kCUDA, 1, kFloat, kStrided);

  options = dtype(kByte).layout(kSparse).device(kCUDA, 2).device_index(3);
  REQUIRE_OPTIONS(kCUDA, 3, kByte, kSparse);
}

// 测试用例，验证从 CPU 类型构造张量选项是否正确
TEST(TensorOptionsTest, ConstructsWellFromCPUTypes) {
  TensorOptions options;
  REQUIRE_OPTIONS(kCPU, -1, kFloat, kStrided);

  options = TensorOptions({kCPU, 0});
  REQUIRE_OPTIONS(kCPU, 0, kFloat, kStrided);

  options = TensorOptions("cpu:0");
  REQUIRE_OPTIONS(kCPU, 0, kFloat, kStrided);

  options = TensorOptions(kInt);
  REQUIRE_OPTIONS(kCPU, -1, kInt, kStrided);

  options =
      TensorOptions(getDeprecatedTypeProperties(Backend::SparseCPU, kFloat));
  REQUIRE_OPTIONS(kCPU, -1, kFloat, kSparse);

  options =
      TensorOptions(getDeprecatedTypeProperties(Backend::SparseCPU, kByte));
  REQUIRE_OPTIONS(kCPU, -1, kByte, kSparse);
}

// 测试用例，验证从 CPU 张量构造张量选项是否正确
TEST(TensorOptionsTest, ConstructsWellFromCPUTensors) {
  auto options = empty(5, kDouble).options();
  REQUIRE_OPTIONS(kCPU, -1, kDouble, kStrided);

  options = empty(5, getDeprecatedTypeProperties(Backend::SparseCPU, kByte))
                .options();
  REQUIRE_OPTIONS(kCPU, -1, kByte, kSparse);
}

// 测试用例，验证从变量构造张量选项是否正确
TEST(TensorOptionsTest, ConstructsWellFromVariables) {
  auto options = torch::empty(5).options();
  REQUIRE_OPTIONS(kCPU, -1, kFloat, kStrided);
  ASSERT_FALSE(options.requires_grad());

  options = torch::empty(5, at::requires_grad()).options();
  REQUIRE_OPTIONS(kCPU, -1, kFloat, kStrided);
  ASSERT_FALSE(options.requires_grad());
}
TEST`
TEST(DeviceTest, ParsesCorrectlyFromString) {
  // 创建一个 Device 对象，初始化为 "cpu:0"
  Device device("cpu:0");
  // 检查 device 是否等于 Device(DeviceType::CPU, 0)
  ASSERT_EQ(device, Device(DeviceType::CPU, 0));

  // 重新初始化 device 为 "cpu"
  device = Device("cpu");
  // 检查 device 是否等于 Device(DeviceType::CPU)
  ASSERT_EQ(device, Device(DeviceType::CPU));

  // 重新初始化 device 为 "cuda:123"
  device = Device("cuda:123");
  // 检查 device 是否等于 Device(DeviceType::CUDA, 123)
  ASSERT_EQ(device, Device(DeviceType::CUDA, 123));

  // 重新初始化 device 为 "cuda"
  device = Device("cuda");
  // 检查 device 是否等于 Device(DeviceType::CUDA)
  ASSERT_EQ(device, Device(DeviceType::CUDA));

  // 重新初始化 device 为 "mkldnn"
  device = Device("mkldnn");
  // 检查 device 是否等于 Device(DeviceType::MKLDNN)
  ASSERT_EQ(device, Device(DeviceType::MKLDNN));

  // 重新初始化 device 为 "opengl"
  device = Device("opengl");
  // 检查 device 是否等于 Device(DeviceType::OPENGL)
  ASSERT_EQ(device, Device(DeviceType::OPENGL));

  // 重新初始化 device 为 "opencl"
  device = Device("opencl");
  // 检查 device 是否等于 Device(DeviceType::OPENCL)
  ASSERT_EQ(device, Device(DeviceType::OPENCL));

  // 重新初始化 device 为 "ideep"
  device = Device("ideep");
  // 检查 device 是否等于 Device(DeviceType::IDEEP)
  ASSERT_EQ(device, Device(DeviceType::IDEEP));

  // 重新初始化 device 为 "hip"
  device = Device("hip");
  // 检查 device 是否等于 Device(DeviceType::HIP)
  ASSERT_EQ(device, Device(DeviceType::HIP));

  // 重新初始化 device 为 "hip:123"
  device = Device("hip:123");
  // 检查 device 是否等于 Device(DeviceType::HIP, 123)
  ASSERT_EQ(device, Device(DeviceType::HIP, 123));

  // 定义一个包含无效设备字符串的字符串向量
  std::vector<std::string> badnesses = {
      "", "cud:1", "cuda:", "cpu::1", ":1", "3", "tpu:4", "??"};
  // 遍历字符串向量，测试每个字符串是否抛出异常
  for (const auto& badness : badnesses) {
    // NOLINTNEXTLINE 注释指示此行代码禁止特定代码风格检查
    ASSERT_ANY_THROW({ Device d(badness); });
  }
}

TEST(DefaultDtypeTest, CanSetAndGetDefaultDtype) {
  // 创建 AutoDefaultDtypeMode 对象，初始化为 kFloat
  AutoDefaultDtypeMode dtype_mode(kFloat);

  // 检查当前默认数据类型是否为 kFloat
  ASSERT_EQ(at::get_default_dtype(), kFloat);
  // 设置默认数据类型为 int
  set_default_dtype(caffe2::TypeMeta::Make<int>());
  // 检查当前默认数据类型是否为 kInt
  ASSERT_EQ(at::get_default_dtype(), kInt);
}

TEST(DefaultDtypeTest, NewTensorOptionsHasCorrectDefault) {
  // 创建 AutoDefaultDtypeMode 对象，初始化为 kFloat
  AutoDefaultDtypeMode dtype_mode(kFloat);

  // 设置默认数据类型为 int
  set_default_dtype(caffe2::TypeMeta::Make<int>());
  // 检查当前默认数据类型是否为 kInt
  ASSERT_EQ(at::get_default_dtype(), kInt);
  // 创建 TensorOptions 对象
  TensorOptions options;
  // 检查 TensorOptions 的数据类型是否为 kInt
  ASSERT_EQ(options.dtype(), kInt);
}

TEST(DefaultDtypeTest, NewTensorsHaveCorrectDefaultDtype) {
  // 创建 AutoDefaultDtypeMode 对象，初始化为 kFloat
  AutoDefaultDtypeMode dtype_mode(kFloat);
  // 设置默认数据类型为 int
  set_default_dtype(caffe2::TypeMeta::Make<int>());
  {
    // 创建一个 tensor，元素全为 1，默认数据类型为 int
    auto tensor = torch::ones(5);
    // 检查 tensor 的数据类型是否为 kInt
    ASSERT_EQ(tensor.dtype(), kInt);
  }
  // 设置默认数据类型为 double
  set_default_dtype(caffe2::TypeMeta::Make<double>());
  {
    // 创建一个 tensor，元素全为 1，默认数据类型为 double
    auto tensor = torch::ones(5);
    // 检查 tensor 的数据类型是否为 kDouble
    ASSERT_EQ(tensor.dtype(), kDouble);
  }
  {
    // 创建一个 tensor，元素全为 1，数据类型为 kFloat
    auto tensor = torch::ones(5, kFloat);
    // 检查 tensor 的数据类型是否为 kFloat
    ASSERT_EQ(tensor.dtype(), kFloat);
  }
}
```