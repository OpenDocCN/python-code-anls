# `.\pytorch\test\cpp\api\special.cpp`

```
#include <gtest/gtest.h>  // 包含 Google Test 的头文件

#include <torch/special.h>  // 包含 Torch 的特殊函数头文件
#include <torch/torch.h>    // 包含 Torch 的核心头文件

#include <test/cpp/api/support.h>  // 包含测试支持函数的头文件

// 简单测试，验证特殊函数命名空间在 C++ 中正确注册
TEST(SpecialTest, special) {
  // 生成一个大小为 128x128 的双精度随机张量
  auto t = torch::randn(128, torch::kDouble);
  // 调用 Torch 的特殊函数 gammaln 计算张量 t 的伽马函数的自然对数
  torch::special::gammaln(t);
}
```