# `.\pytorch\test\cpp\common\main.cpp`

```
#include <gtest/gtest.h>  // 包含 Google Test 框架的头文件

#include <torch/cuda.h>  // 包含 PyTorch CUDA 相关的头文件

#include <iostream>  // 标准输入输出流
#include <string>    // 标准字符串库

std::string add_negative_flag(const std::string& flag) {
  std::string filter = ::testing::GTEST_FLAG(filter);  // 获取当前 Google Test 的过滤器设置
  if (filter.find('-') == std::string::npos) {  // 如果过滤器中没有 '-' 字符
    filter.push_back('-');  // 在过滤器末尾添加 '-' 字符
  } else {
    filter.push_back(':');  // 否则，在过滤器末尾添加 ':' 字符
  }
  filter += flag;  // 将传入的标志添加到过滤器末尾
  return filter;  // 返回修改后的过滤器字符串
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);  // 初始化 Google Test 框架

  if (!torch::cuda::is_available()) {  // 如果 CUDA 不可用
    std::cout << "CUDA not available. Disabling CUDA and MultiCUDA tests" << std::endl;  // 输出提示信息
    ::testing::GTEST_FLAG(filter) = add_negative_flag("*_CUDA:*_MultiCUDA");  // 设置 Google Test 过滤器，禁用 CUDA 和 MultiCUDA 测试
  } else if (torch::cuda::device_count() < 2) {  // 如果 CUDA 设备数量小于 2
    std::cout << "Only one CUDA device detected. Disabling MultiCUDA tests" << std::endl;  // 输出提示信息
    ::testing::GTEST_FLAG(filter) = add_negative_flag("*_MultiCUDA");  // 设置 Google Test 过滤器，禁用 MultiCUDA 测试
  }

  return RUN_ALL_TESTS();  // 运行所有的 Google Test 测试用例
}
```