# `.\pytorch\aten\src\ATen\test\reduce_ops_test.cpp`

```py
#include <gtest/gtest.h>

#include <torch/types.h>
#include <torch/utils.h>

using namespace at;

// 定义一个测试用例 ReduceOpsTest，测试最大值和最小值操作
TEST(ReduceOpsTest, MaxValuesAndMinValues) {
  // 设置矩阵的宽度和高度
  const int W = 10;
  const int H = 10;
  
  // 检查是否支持 CUDA
  if (hasCUDA()) {
    // 对于每种数据类型（半精度、单精度、双精度），执行以下操作
    for (const auto dtype : {kHalf, kFloat, kDouble}) {
      // 生成一个在 CUDA 设备上随机初始化的 Tensor 对象 a
      auto a = at::rand({H, W}, TensorOptions(kCUDA).dtype(dtype));
      
      // 断言：按照指定的维度计算最大值，并与整个 Tensor 的最大值进行比较
      ASSERT_FLOAT_EQ(
        a.amax(c10::IntArrayRef{0, 1}).item<double>(),
        a.max().item<double>()
      );
      
      // 断言：按照指定的维度计算最小值，并与整个 Tensor 的最小值进行比较
      ASSERT_FLOAT_EQ(
        a.amin(c10::IntArrayRef{0, 1}).item<double>(),
        a.min().item<double>()
      );
    }
  }
}
```