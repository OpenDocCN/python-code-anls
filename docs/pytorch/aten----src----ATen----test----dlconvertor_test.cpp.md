# `.\pytorch\aten\src\ATen\test\dlconvertor_test.cpp`

```
// 包含 Google Test 框架的头文件
#include <gtest/gtest.h>

// 包含 PyTorch ATen 库的头文件
#include <ATen/ATen.h>
#include <ATen/DLConvertor.h>

// 包含标准输入输出流的头文件
#include <iostream>

// NOLINTNEXTLINE(modernize-deprecated-headers)
// 包含 C 标准库的头文件，用于字符串操作
#include <string.h>

// 包含字符串流的头文件
#include <sstream>

// 使用 at 命名空间
using namespace at;

// 定义测试用例 TestDlconvertor.TestDlconvertor
TEST(TestDlconvertor, TestDlconvertor) {
  // 设置随机种子为 123
  manual_seed(123);

  // 创建一个大小为 3x4 的随机张量 a
  Tensor a = rand({3, 4});

  // 将张量 a 转换为 DLManagedTensor 结构体指针 dlMTensor
  DLManagedTensor* dlMTensor = toDLPack(a);

  // 从 DLManagedTensor 结构体指针 dlMTensor 转换回张量 b
  Tensor b = fromDLPack(dlMTensor);

  // 断言张量 a 和 b 相等
  ASSERT_TRUE(a.equal(b));
}

// 定义测试用例 TestDlconvertor.TestDlconvertorNoStrides
TEST(TestDlconvertor, TestDlconvertorNoStrides) {
  // 设置随机种子为 123
  manual_seed(123);

  // 创建一个大小为 3x4 的随机张量 a
  Tensor a = rand({3, 4});

  // 将张量 a 转换为 DLManagedTensor 结构体指针 dlMTensor
  DLManagedTensor* dlMTensor = toDLPack(a);

  // 将 DLManagedTensor 结构体指针 dlMTensor 的 strides 指针置为 nullptr
  dlMTensor->dl_tensor.strides = nullptr;

  // 从 DLManagedTensor 结构体指针 dlMTensor 转换回张量 b
  Tensor b = fromDLPack(dlMTensor);

  // 断言张量 a 和 b 相等
  ASSERT_TRUE(a.equal(b));
}
```