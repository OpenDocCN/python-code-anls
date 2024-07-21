# `.\pytorch\test\cpp\api\meta_tensor.cpp`

```py
#include <gtest/gtest.h>  // 引入 Google Test 框架的头文件

#include <ATen/MetaFunctions.h>  // 引入 PyTorch 的 ATen 元函数头文件
#include <torch/torch.h>  // 引入 PyTorch 的主头文件

#include <vector>  // 引入 C++ 标准库中的向量容器

TEST(MetaTensorTest, MetaDeviceApi) {
  auto a = at::ones({4}, at::kFloat);  // 创建一个包含全为1的张量 a，数据类型为 Float
  auto b = at::ones({3, 4}, at::kFloat);  // 创建一个包含全为1的二维张量 b，数据类型为 Float
  // 如果输入是元张量，则 at::add() 返回一个元张量。
  auto out_meta = at::add(a.to(c10::kMeta), b.to(c10::kMeta));

  ASSERT_EQ(a.device(), c10::kCPU);  // 断言张量 a 在 CPU 上
  ASSERT_EQ(b.device(), c10::kCPU);  // 断言张量 b 在 CPU 上
  ASSERT_EQ(out_meta.device(), c10::kMeta);  // 断言输出元张量在元设备上
  c10::IntArrayRef sizes_actual = out_meta.sizes();  // 获取输出元张量的尺寸信息
  std::vector<int64_t> sizes_expected = std::vector<int64_t>{3, 4};  // 期待的输出尺寸
  ASSERT_EQ(sizes_actual, sizes_expected);  // 断言实际输出尺寸与期待的输出尺寸相同
}

TEST(MetaTensorTest, MetaNamespaceApi) {
  auto a = at::ones({4}, at::kFloat);  // 创建一个包含全为1的张量 a，数据类型为 Float
  auto b = at::ones({3, 4}, at::kFloat);  // 创建一个包含全为1的二维张量 b，数据类型为 Float
  // at::meta:: 命名空间接受来自任何后端的张量，并返回一个元张量。
  auto out_meta = at::meta::add(a, b);

  ASSERT_EQ(a.device(), c10::kCPU);  // 断言张量 a 在 CPU 上
  ASSERT_EQ(b.device(), c10::kCPU);  // 断言张量 b 在 CPU 上
  ASSERT_EQ(out_meta.device(), c10::kMeta);  // 断言输出元张量在元设备上
  c10::IntArrayRef sizes_actual = out_meta.sizes();  // 获取输出元张量的尺寸信息
  std::vector<int64_t> sizes_expected = std::vector<int64_t>{3, 4};  // 期待的输出尺寸
  ASSERT_EQ(sizes_actual, sizes_expected);  // 断言实际输出尺寸与期待的输出尺寸相同
}
```