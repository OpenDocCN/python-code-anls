# `.\pytorch\aten\src\ATen\test\packedtensoraccessor_test.cpp`

```
// 引入 ATen 库的操作符和测试断言
#include <ATen/Operators.h>
// 引入 ATen 库的测试断言
#include <ATen/test/test_assert.h>
// 引入 C10 实用工具中的异常处理
#include <c10/util/Exception.h>
// 引入 C10 实用工具中的范围迭代器
#include <c10/util/irange.h>
// 引入 Google 测试框架的头文件
#include <gtest/gtest.h>

// 引入 ATen 库的核心功能
#include <ATen/ATen.h>
// 引入 ATen 库的张量访问器
#include <ATen/core/TensorAccessor.h>

// 引入断言库
#include <cassert>

// 使用 at 命名空间
using namespace at;

// 定义测试用例 PackedtensoraccessorTest 中的 TransposeTest
TEST(PackedtensoraccessorTest, TransposeTest) {
  // 设置随机种子
  manual_seed(123);
  
  /* test a 3d tensor */
  // 定义 3 维张量的维度和大小
  constexpr int dimension = 3;
  constexpr std::array<int64_t, dimension> sizes{3, 4, 5};
  // 创建随机数填充的 CPU Float 类型的张量
  Tensor t = rand(sizes, CPU(kFloat));
  // 获取原始张量的 packed_accessor64 访问器
  auto original = t.packed_accessor64<float, dimension, DefaultPtrTraits>();
  // 对原始张量进行维度 0 和 2 的转置操作
  auto transposed = original.transpose(0, 2);
  // 断言转置后张量的维度符合预期
  ASSERT_EQ(original.size(0), transposed.size(2));
  ASSERT_EQ(original.size(1), transposed.size(1));
  ASSERT_EQ(original.size(2), transposed.size(0));
  // 遍历并断言原始张量和转置后张量的对应元素相等关系
  for (const auto i : c10::irange(sizes[0])) {
    for (const auto j : c10::irange(sizes[1])) {
      for (const auto k : c10::irange(sizes[2])) {
        ASSERT_EQ(original[i][j][k], transposed[k][j][i]);
      }
    }
  }

  /* test the special case of a 1d tensor */
  // 测试 1 维张量的特殊情况
  int size = 3;
  t = rand({size}, CPU(kFloat));
  // 获取 1 维张量的 packed_accessor64 访问器
  auto original_1d = t.packed_accessor64<float, 1, DefaultPtrTraits>();
  // 对 1 维张量进行维度 0 和 0 的转置操作（无变化）
  auto transposed_1d = original_1d.transpose(0, 0);
  // 断言转置后张量与原始张量元素相等
  for (const auto i : c10::irange(size)) {
    ASSERT_EQ(original_1d[i], transposed_1d[i]);
  }

  /* test the error conditions */
  // 测试错误条件
  // 断言在超出索引范围时会抛出 IndexError 异常
  ASSERT_THROW(original.transpose(2, 5), c10::IndexError);
  ASSERT_THROW(original_1d.transpose(1, 0), c10::IndexError);
}
```