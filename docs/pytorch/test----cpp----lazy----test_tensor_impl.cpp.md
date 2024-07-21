# `.\pytorch\test\cpp\lazy\test_tensor_impl.cpp`

```py
#include <gtest/gtest.h>  // 引入 Google Test 框架的头文件

#include <torch/csrc/lazy/core/tensor_impl.h>  // 引入 Lazy Tensor 实现的头文件
#include <torch/torch.h>  // 引入 PyTorch 主头文件

namespace torch {
namespace lazy {

#ifdef FBCODE_CAFFE2
// 如果定义了 FBCODE_CAFFE2 宏，则进行以下测试
TEST(LazyTensorImplTest, BasicThrow) {
  // 断言：期望以下代码块抛出 ::c10::Error 异常
  EXPECT_THROW(
      {
        // 创建一个 lazy device 上的随机张量，尺寸为 {0, 1, 3, 0}，数据类型为 float
        auto input = torch::rand(
            {0, 1, 3, 0}, torch::TensorOptions(torch::kFloat).device("lazy"));
      },
      ::c10::Error);
}
#endif // FBCODE_CAFFE2

} // namespace lazy
} // namespace torch
```