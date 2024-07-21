# `.\pytorch\test\cpp\api\tensor_cuda.cpp`

```py
#include <gtest/gtest.h>

#include <ATen/ATen.h>

#include <cmath>

// 定义一个宏，用于验证张量的设备选项是否符合预期
#define REQUIRE_TENSOR_OPTIONS(device_, index_, type_, layout_)            \
  ASSERT_TRUE(                                                             \
      tensor.device().type() == at::Device((device_), (index_)).type());   \
  ASSERT_TRUE(                                                             \
      tensor.device().index() == at::Device((device_), (index_)).index()); \
  ASSERT_EQ(tensor.dtype(), (type_));                                      \
  ASSERT_TRUE(tensor.layout() == (layout_))

// 测试用例：验证张量在多GPU环境下是否正确分配在指定设备上
TEST(TensorTest, AllocatesTensorOnTheCorrectDevice_MultiCUDA) {
  // 创建一个张量，指定在第1个CUDA设备上
  auto tensor = at::tensor({1, 2, 3}, at::device({at::kCUDA, 1}));
  ASSERT_EQ(tensor.device().type(), at::Device::Type::CUDA);  // 验证设备类型为CUDA
  ASSERT_EQ(tensor.device().index(), 1);                       // 验证设备索引为1
}

// 测试用例：验证张量在不同设备之间的转移操作
TEST(TensorTest, ToDevice_MultiCUDA) {
  // 创建一个空张量
  auto tensor = at::empty({3, 4});
  // 验证张量当前设备为CPU、浮点类型、步进布局
  REQUIRE_TENSOR_OPTIONS(at::kCPU, -1, at::kFloat, at::kStrided);

  // 将张量转移到第1个CUDA设备上
  tensor = tensor.to({at::kCUDA, 1});
  // 验证设备为第1个CUDA设备、浮点类型、步进布局
  REQUIRE_TENSOR_OPTIONS(at::kCUDA, 1, at::kFloat, at::kStrided);

  // 将张量转移到第0个CUDA设备上
  tensor = tensor.to({at::kCUDA, 0});
  // 验证设备为第0个CUDA设备、浮点类型、步进布局
  REQUIRE_TENSOR_OPTIONS(at::kCUDA, 0, at::kFloat, at::kStrided);

  // 再次将张量转移到第1个CUDA设备上
  tensor = tensor.to({at::kCUDA, 1});
  // 验证设备为第1个CUDA设备、浮点类型、步进布局
  REQUIRE_TENSOR_OPTIONS(at::kCUDA, 1, at::kFloat, at::kStrided);

  // 将张量转移到CPU设备上
  tensor = tensor.to(at::Device(at::kCPU));
  // 验证设备为CPU、浮点类型、步进布局
  REQUIRE_TENSOR_OPTIONS(at::kCPU, -1, at::kFloat, at::kStrided);

  // 将张量转移到默认的第0个CUDA设备上
  tensor = tensor.to(at::kCUDA);
  // 验证设备为第0个CUDA设备、浮点类型、步进布局
  REQUIRE_TENSOR_OPTIONS(at::kCUDA, 0, at::kFloat, at::kStrided);

  // 将张量转移到第1个CUDA设备上，使用TensorOptions进行指定
  tensor = tensor.to(at::TensorOptions({at::kCUDA, 1}));
  // 验证设备为第1个CUDA设备、浮点类型、步进布局
  REQUIRE_TENSOR_OPTIONS(at::kCUDA, 1, at::kFloat, at::kStrided);

  // 将张量转移到第0个CUDA设备上，使用TensorOptions进行指定
  tensor = tensor.to(at::TensorOptions({at::kCUDA, 0}));
  // 验证设备为第0个CUDA设备、浮点类型、步进布局
  REQUIRE_TENSOR_OPTIONS(at::kCUDA, 0, at::kFloat, at::kStrided);

  // 将张量转移到双精度浮点类型上
  tensor = tensor.to(at::TensorOptions(at::kDouble));
  // 验证设备为第0个CUDA设备、双精度浮点类型、步进布局
  REQUIRE_TENSOR_OPTIONS(at::kCUDA, 0, at::kDouble, at::kStrided);

  // 将张量转移到第1个CUDA设备上，使用TensorOptions进行指定
  tensor = tensor.to(at::TensorOptions({at::kCUDA, 1}));
  // 验证设备为第1个CUDA设备、双精度浮点类型、步进布局
  REQUIRE_TENSOR_OPTIONS(at::kCUDA, 1, at::kDouble, at::kStrided);

  // 将张量转移到整型类型上
  tensor = tensor.to(at::TensorOptions(at::kInt));
  // 验证设备为第1个CUDA设备、整型类型、步进布局
  REQUIRE_TENSOR_OPTIONS(at::kCUDA, 1, at::kInt, at::kStrided);

  // 将张量转移到CPU设备上
  tensor = tensor.to(at::TensorOptions(at::Device(at::kCPU)));
  // 验证设备为CPU、整型类型、步进布局
  REQUIRE_TENSOR_OPTIONS(at::kCPU, -1, at::kInt, at::kStrided);

  // 将张量转移到默认的第0个CUDA设备上
  tensor = tensor.to(at::TensorOptions(at::kCUDA));
  // 验证设备为第0个CUDA设备、整型类型、步进布局
  REQUIRE_TENSOR_OPTIONS(at::kCUDA, 0, at::kInt, at::kStrided);
}
TEST(TensorTest, ToTensorAndTensorAttributes_MultiCUDA) {
  // 创建一个形状为 {3, 4} 的空张量 tensor
  auto tensor = at::empty({3, 4});
  // 断言张量的选项满足指定要求
  REQUIRE_TENSOR_OPTIONS(at::kCPU, -1, at::kFloat, at::kStrided);

  // 创建另一个形状为 {3, 4}、数据类型为 kFloat 的空张量 other
  auto other = at::empty({3, 4}, at::kFloat);
  // 将 tensor 转换为 other 的类型
  tensor = tensor.to(other);
  // 再次断言张量的选项满足指定要求
  REQUIRE_TENSOR_OPTIONS(at::kCPU, -1, at::kFloat, at::kStrided);

  // 创建一个形状为 {3, 4}、数据类型为 kDouble 的空张量 other
  other = at::empty({3, 4}, at::TensorOptions(at::kCUDA).dtype(at::kDouble));
  // 将 tensor 转换为与 other 的数据类型相同
  tensor = tensor.to(other.dtype());
  // 再次断言张量的选项满足指定要求
  REQUIRE_TENSOR_OPTIONS(at::kCPU, -1, at::kDouble, at::kStrided);
  // 将 tensor 转换到 other 所在的设备
  tensor = tensor.to(other.device());
  // 再次断言张量的选项满足指定要求
  REQUIRE_TENSOR_OPTIONS(at::kCUDA, 0, at::kDouble, at::kStrided);

  // 创建一个形状为 {3, 4}、数据类型为 kLong 的空张量 other
  other = at::empty({3, 4}, at::TensorOptions({at::kCUDA, 1}).dtype(at::kLong));
  // 将 tensor 转换到 other 所在的设备和数据类型
  tensor = tensor.to(other.device(), other.dtype());
  // 再次断言张量的选项满足指定要求
  REQUIRE_TENSOR_OPTIONS(at::kCUDA, 1, at::kLong, at::kStrided);

  // 创建一个形状为 {3, 4}、数据类型为 kFloat 的空张量 other
  other = at::empty({3, 4}, at::kFloat);
  // 将 tensor 转换为与 other 的选项相同
  tensor = tensor.to(other.options());
  // 再次断言张量的选项满足指定要求
  REQUIRE_TENSOR_OPTIONS(at::kCPU, -1, at::kFloat, at::kStrided);
}

TEST(TensorTest, ToDoesNotCopyWhenOptionsAreAllTheSame_CUDA) {
  // 创建一个形状为 {3, 4}、数据类型为 kFloat，设备为 CUDA 的空张量 tensor
  auto tensor = at::empty(
      {3, 4}, at::TensorOptions(at::kFloat).device(at::Device("cuda")));
  // 将 tensor 转换为与自身选项相同的张量，确保不复制数据
  auto hopefully_not_copy = tensor.to(tensor.options());
  // 断言新张量与原张量共享相同的数据指针
  ASSERT_EQ(hopefully_not_copy.data_ptr<float>(), tensor.data_ptr<float>());
  // 将 tensor 转换为数据类型为 kFloat 的张量
  hopefully_not_copy = tensor.to(at::kFloat);
  // 断言新张量与原张量共享相同的数据指针
  ASSERT_EQ(hopefully_not_copy.data_ptr<float>(), tensor.data_ptr<float>());
  // 将 tensor 转换到 CUDA 设备
  hopefully_not_copy = tensor.to("cuda");
  // 断言新张量与原张量共享相同的数据指针
  ASSERT_EQ(hopefully_not_copy.data_ptr<float>(), tensor.data_ptr<float>());
  // 将 tensor 转换到 CUDA 设备的张量选项
  hopefully_not_copy = tensor.to(at::TensorOptions("cuda"));
  // 断言新张量与原张量共享相同的数据指针
  ASSERT_EQ(hopefully_not_copy.data_ptr<float>(), tensor.data_ptr<float>());
  // 将 tensor 转换为数据类型为 kFloat 的张量选项
  hopefully_not_copy = tensor.to(at::TensorOptions(at::kFloat));
  // 断言新张量与原张量共享相同的数据指针
  ASSERT_EQ(hopefully_not_copy.data_ptr<float>(), tensor.data_ptr<float>());
}

TEST(TensorTest, ToDeviceAndDtype_MultiCUDA) {
  // 创建一个形状为 {3, 4} 的空张量 tensor
  auto tensor = at::empty({3, 4});
  // 断言张量的选项满足指定要求
  REQUIRE_TENSOR_OPTIONS(at::kCPU, -1, at::kFloat, at::kStrided);

  // 将 tensor 转换到 CUDA 设备 {at::kCUDA, 1}，数据类型为 kInt
  tensor = tensor.to({at::kCUDA, 1}, at::kInt);
  // 再次断言张量的选项满足指定要求
  REQUIRE_TENSOR_OPTIONS(at::kCUDA, 1, at::kInt, at::kStrided);

  // 将 tensor 转换到 CUDA 设备 {at::kCUDA, 0}，数据类型为 kLong
  tensor = tensor.to(at::TensorOptions({at::kCUDA, 0}).dtype(at::kLong));
  // 再次断言张量的选项满足指定要求
  REQUIRE_TENSOR_OPTIONS(at::kCUDA, 0, at::kLong, at::kStrided);

  // 将 tensor 转换到 CUDA 设备 {at::kCUDA, 1}，数据类型为 kDouble
  tensor = tensor.to(at::TensorOptions({at::kCUDA, 1}).dtype(at::kDouble));
  // 再次断言张量的选项满足指定要求
  REQUIRE_TENSOR_OPTIONS(at::kCUDA, 1, at::kDouble, at::kStrided);

  // 将 tensor 转换到 CPU 设备，数据类型为 kInt
  tensor = tensor.to(at::kCPU, at::kInt);
  // 再次断言张量的选项满足指定要求
  REQUIRE_TENSOR_OPTIONS(at::kCPU, -1, at::kInt, at::kStrided);
}

TEST(TensorTest, MagmaInitializesCorrectly_CUDA) {
  // 创建一个形状为 {4, 4}，数据为 data 的张量，数据类型为 kFloat，存储在 CUDA 设备上
  // 如果 MAGMA 可用，则计算其逆矩阵
  float data[] = {1, 1, 1, 0, 0, 3, 1, 2, 2, 3, 1, 0, 1, 0, 2, 1};
  auto tensor =
      at::from_blob(data, {4, 4}, at::TensorOptions(at::kFloat)).cuda();
  if (at::hasMAGMA()) {
    at::inverse(tensor);
  }
}
```