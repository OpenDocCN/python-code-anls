# `.\pytorch\test\cpp\api\dispatch.cpp`

```
#include <gtest/gtest.h>  // 引入 Google Test 框架的头文件

#include <ATen/native/Pow.h>  // 引入 PyTorch 的 ATen 模块中的 Pow 头文件
#include <c10/util/irange.h>  // 引入 PyTorch 的 c10 模块中的 irange 头文件
#include <test/cpp/api/support.h>  // 引入 PyTorch 的测试支持头文件
#include <torch/torch.h>  // 引入 PyTorch 的 torch 头文件
#include <torch/types.h>  // 引入 PyTorch 的 tensor 类型头文件
#include <torch/utils.h>  // 引入 PyTorch 的 utils 头文件
#include <cstdlib>  // 引入 C 标准库的 stdlib 头文件，包含环境操作函数
#include <iostream>  // 引入 C++ 标准库的输入输出流头文件
#include <type_traits>  // 引入 C++ 标准库的类型特性头文件
#include <vector>  // 引入 C++ 标准库的向量容器头文件

struct DispatchTest : torch::test::SeedingFixture {};  // 定义一个测试结构体 DispatchTest，继承自 torch 的测试基类 SeedingFixture

TEST_F(DispatchTest, TestAVX2) {  // 定义一个测试用例 TestAVX2，属于 DispatchTest 测试结构体
  const std::vector<int> ints{1, 2, 3, 4};  // 定义一个整数向量 ints
  const std::vector<int> result{1, 4, 27, 256};  // 定义一个整数向量 result，表示预期结果
  const auto vals_tensor = torch::tensor(ints);  // 创建一个 PyTorch 张量 vals_tensor，存储 ints 中的数据
  const auto pows_tensor = torch::tensor(ints);  // 创建一个 PyTorch 张量 pows_tensor，存储 ints 中的数据

#ifdef _WIN32
  _putenv("ATEN_CPU_CAPABILITY=avx2");  // 在 Windows 下设置环境变量 ATEN_CPU_CAPABILITY 为 avx2
#else
  setenv("ATEN_CPU_CAPABILITY", "avx2", 1);  // 在 Linux 下设置环境变量 ATEN_CPU_CAPABILITY 为 avx2
#endif

  const auto actual_pow_avx2 = vals_tensor.pow(pows_tensor);  // 使用 avx2 特性进行张量的幂运算
  for (const auto i : c10::irange(4)) {  // 遍历范围为 0 到 3 的整数
    ASSERT_EQ(result[i], actual_pow_avx2[i].item<int>());  // 断言计算结果与预期结果相等
  }
}

TEST_F(DispatchTest, TestAVX512) {  // 定义一个测试用例 TestAVX512，属于 DispatchTest 测试结构体
  const std::vector<int> ints{1, 2, 3, 4};  // 定义一个整数向量 ints
  const std::vector<int> result{1, 4, 27, 256};  // 定义一个整数向量 result，表示预期结果
  const auto vals_tensor = torch::tensor(ints);  // 创建一个 PyTorch 张量 vals_tensor，存储 ints 中的数据
  const auto pows_tensor = torch::tensor(ints);  // 创建一个 PyTorch 张量 pows_tensor，存储 ints 中的数据

#ifdef _WIN32
  _putenv("ATEN_CPU_CAPABILITY=avx512");  // 在 Windows 下设置环境变量 ATEN_CPU_CAPABILITY 为 avx512
#else
  setenv("ATEN_CPU_CAPABILITY", "avx512", 1);  // 在 Linux 下设置环境变量 ATEN_CPU_CAPABILITY 为 avx512
#endif

  const auto actual_pow_avx512 = vals_tensor.pow(pows_tensor);  // 使用 avx512 特性进行张量的幂运算
  for (const auto i : c10::irange(4)) {  // 遍历范围为 0 到 3 的整数
    ASSERT_EQ(result[i], actual_pow_avx512[i].item<int>());  // 断言计算结果与预期结果相等
  }
}

TEST_F(DispatchTest, TestDefault) {  // 定义一个测试用例 TestDefault，属于 DispatchTest 测试结构体
  const std::vector<int> ints{1, 2, 3, 4};  // 定义一个整数向量 ints
  const std::vector<int> result{1, 4, 27, 256};  // 定义一个整数向量 result，表示预期结果
  const auto vals_tensor = torch::tensor(ints);  // 创建一个 PyTorch 张量 vals_tensor，存储 ints 中的数据
  const auto pows_tensor = torch::tensor(ints);  // 创建一个 PyTorch 张量 pows_tensor，存储 ints 中的数据

#ifdef _WIN32
  _putenv("ATEN_CPU_CAPABILITY=default");  // 在 Windows 下设置环境变量 ATEN_CPU_CAPABILITY 为 default
#else
  setenv("ATEN_CPU_CAPABILITY", "default", 1);  // 在 Linux 下设置环境变量 ATEN_CPU_CAPABILITY 为 default
#endif

  const auto actual_pow_default = vals_tensor.pow(pows_tensor);  // 使用默认特性进行张量的幂运算
  for (const auto i : c10::irange(4)) {  // 遍历范围为 0 到 3 的整数
    ASSERT_EQ(result[i], actual_pow_default[i].item<int>());  // 断言计算结果与预期结果相等
  }
}
```