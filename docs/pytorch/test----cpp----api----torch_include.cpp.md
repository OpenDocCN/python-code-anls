# `.\pytorch\test\cpp\api\torch_include.cpp`

```py
#include <gtest/gtest.h>  // 包含 Google Test 框架的头文件

#include <torch/torch.h>  // 包含 PyTorch C++ API 的头文件

// NOTE: This test suite exists to make sure that common `torch::` functions
// can be used without additional includes beyond `torch/torch.h`.

TEST(TorchIncludeTest, GetSetNumThreads) {  // 定义名为 TorchIncludeTest 的测试套件，测试线程数量相关函数
  torch::init_num_threads();  // 初始化 PyTorch 线程数
  torch::set_num_threads(2);  // 设置 PyTorch 线程数为 2
  torch::set_num_interop_threads(2);  // 设置 PyTorch 与其它库交互的线程数为 2
  torch::get_num_threads();  // 获取当前 PyTorch 线程数
  torch::get_num_interop_threads();  // 获取当前 PyTorch 与其它库交互的线程数
}
```