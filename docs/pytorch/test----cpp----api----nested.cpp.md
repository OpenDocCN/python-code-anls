# `.\pytorch\test\cpp\api\nested.cpp`

```
// 包含 Google Test 框架的头文件
#include <gtest/gtest.h>

// 包含 Torch C++ API 的相关头文件
#include <torch/nested.h>
#include <torch/torch.h>

// 包含用于支持测试的辅助函数和定义的头文件
#include <test/cpp/api/support.h>

// 对 NestedTest 测试套件进行定义，用于测试嵌套命名空间在 C++ 中的正确注册
TEST(NestedTest, Nested) {
  // 生成一个大小为 (2, 3) 的随机张量 a
  auto a = torch::randn({2, 3});
  // 生成一个大小为 (4, 5) 的随机张量 b
  auto b = torch::randn({4, 5});
  // 调用 torch::nested 命名空间中的 nested_tensor 函数，将张量 a 和 b 组成嵌套张量 nt
  auto nt = torch::nested::nested_tensor({a, b});
  // 调用 torch::nested 命名空间中的 to_padded_tensor 函数，将嵌套张量 nt 转换为填充张量，填充值为 0
  torch::nested::to_padded_tensor(nt, 0);
}
```