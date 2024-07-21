# `.\pytorch\aten\src\ATen\test\apply_utils_test.cpp`

```
// 包含 Google Test 的头文件
#include <gtest/gtest.h>

// 包含 ATen 的头文件
#include <ATen/ATen.h>
#include <ATen/CPUApplyUtils.h>
#include <ATen/test/test_assert.h>
#include <c10/util/irange.h>

// 包含标准输入输出流库
#include <iostream>

// 使用标准命名空间
using namespace std;
// 使用 ATen 命名空间
using namespace at;

// 填充张量的函数，将给定的标量值乘以索引值写入张量
void fill_tensor(int64_t scalar, Tensor& t_) {
  auto t = t_.view(-1);
  // 使用 C++11 范围迭代器遍历张量中的元素
  for (const auto i : c10::irange(t.numel())) {
    t[i] = (i + 1) * scalar;
  }
}

// 这个函数测试所有的 applyX 函数。给定形状和两个转置维度，
// 创建5个张量（a0, ..., a4）并对每个张量中的维度a和b进行转置。
// 然后对每种浮点类型调用 applyX 函数。其中 a4 只分配为双精度，
// 而 a0, ..., a3 分配为给定的类型。对于每个 applyX 函数，
// 我们既以读取的相同类型写入（使用a0, ..., aX-1），也以双精度写入（使用a4作为目标）。
// 同时也在零维和空张量上进行测试。
void test(DeprecatedTypeProperties& type, IntArrayRef shape, int64_t a = 0, int64_t b = 1) {
  // 创建一个零维张量并对其进行填充和指数运算
  auto zero_dim = at::empty({}, type);
  zero_dim.fill_(2);
  zero_dim.exp_();
  // 使用 AT_DISPATCH_FLOATING_TYPES 宏对浮点类型进行分发并测试
  AT_DISPATCH_FLOATING_TYPES(zero_dim.scalar_type(), "test0", [&] {
    ASSERT(zero_dim.const_data_ptr<scalar_t>()[0] == std::exp(2));
  });

  // 创建一个空张量并对其进行填充和指数运算
  auto empty_t = at::empty({0}, type);
  empty_t.fill_(3);
  empty_t.exp_();

  // 创建五个空张量，a0 到 a4，其中 a4 使用双精度数据类型
  auto a0 = at::empty({0}, type.options());
  auto a1 = at::empty({0}, type.options());
  auto a2 = at::empty({0}, type.options());
  auto a3 = at::empty({0}, type.options());
  auto a4 = at::empty({0}, at::TensorOptions(kCPU).dtype(kDouble));

  // 将五个张量放入向量中
  std::vector<Tensor> tensors({a0, a1, a2, a3, a4});
  // 使用 C++11 范围迭代器遍历张量并设置它们的形状
  for (const auto i : c10::irange(tensors.size())) {
    tensors[i].resize_(shape);
    // 调用 fill_tensor 函数填充张量中的数据
    fill_tensor(i + 1, tensors[i]);
    // 如果指定了转置维度a和b，则对当前张量进行转置
    if (a >= 0 && b >= 0) {
      tensors[i].transpose_(a, b);
    }
  }

  // 使用 AT_DISPATCH_FLOATING_TYPES 宏对浮点类型进行分发并测试
  AT_DISPATCH_FLOATING_TYPES(a0.scalar_type(), "test1", [&] {
    // 调用 CPU_tensor_apply2 函数对 a0 和 a1 进行元素级操作
    CPU_tensor_apply2<scalar_t, scalar_t>(
        a0, a1, [](scalar_t& y, const scalar_t& x) { y = x * x; });
    // 调用 CPU_tensor_apply2 函数对 a4 和 a1 进行元素级操作，将结果转换为双精度
    CPU_tensor_apply2<double, scalar_t>(
        a4, a1, [](double& y, scalar_t x) { y = (double)(x * x); });
    // 使用范围迭代器遍历张量，并断言操作后的结果正确
    for (const auto i : c10::irange(a0.numel())) {
      auto target = a1.const_data_ptr<scalar_t>()[i] * a1.const_data_ptr<scalar_t>()[i];
      ASSERT(a0.const_data_ptr<scalar_t>()[i] == target);
      ASSERT(a4.const_data_ptr<double>()[i] == target);
    }
  });

  // 使用 AT_DISPATCH_FLOATING_TYPES 宏对浮点类型进行分发并测试
  AT_DISPATCH_FLOATING_TYPES(a0.scalar_type(), "test2", [&] {
    // 调用 CPU_tensor_apply3 函数对 a0, a1 和 a2 进行元素级操作
    CPU_tensor_apply3<scalar_t, scalar_t, scalar_t>(
        a0, a1, a2, [](scalar_t& y, const scalar_t& x, const scalar_t& z) {
          y = x * x + z;
        });
    // 调用 CPU_tensor_apply3 函数对 a4, a1 和 a2 进行元素级操作，将结果转换为双精度
    CPU_tensor_apply3<double, scalar_t, scalar_t>(
        a4, a1, a2, [](double& y, const scalar_t& x, const scalar_t& z) {
          y = (double)(x * x + z);
        });
    // 使用范围迭代器遍历 a0 张量中的所有元素
    for (const auto i : c10::irange(a0.numel())) {
      // 计算目标值，即 a1 和 a2 对应位置的数值平方和
      auto target = a1.const_data_ptr<scalar_t>()[i] * a1.const_data_ptr<scalar_t>()[i];
      target = target + a2.const_data_ptr<scalar_t>()[i];
      // 断言检查 a0 中第 i 个位置的值是否等于目标值
      ASSERT(a0.const_data_ptr<scalar_t>()[i] == target);
      // 断言检查 a4 中第 i 个位置的值是否等于目标值（类型转换后的）
      ASSERT(a4.const_data_ptr<double>()[i] == target);
    }
  });

  // 根据 a0 张量的数据类型进行分发，执行 CPU_tensor_apply4 操作
  AT_DISPATCH_FLOATING_TYPES(a0.scalar_type(), "test3", [&] {
    CPU_tensor_apply4<scalar_t, scalar_t, scalar_t, scalar_t>(
        a0,
        a1,
        a2,
        a3,
        // 对 a0, a1, a2, a3 四个张量执行指定操作
        [](scalar_t& y,
           const scalar_t& x,
           const scalar_t& z,
           const scalar_t& a) { y = x * x + z * a; });
    // 对双精度浮点类型执行 CPU_tensor_apply4 操作
    CPU_tensor_apply4<double, scalar_t, scalar_t, scalar_t>(
        a4,
        a1,
        a2,
        a3,
        // 将结果存入 y，计算 x*x + z*a 的值（强制转换为双精度）
        [](double& y, const scalar_t& x, const scalar_t& z, const scalar_t& a) {
          y = (double)(x * x + z * a);
        });
    
    // 使用范围迭代器遍历 a0 张量中的所有元素
    for (const auto i : c10::irange(a0.numel())) {
      // 计算目标值，即 a1 和 (a2 * a3) 对应位置的数值平方和
      auto target = a1.const_data_ptr<scalar_t>()[i] * a1.const_data_ptr<scalar_t>()[i];
      target = target + a2.const_data_ptr<scalar_t>()[i] * a3.const_data_ptr<scalar_t>()[i];
      // 断言检查 a0 中第 i 个位置的值是否等于目标值
      ASSERT(a0.const_data_ptr<scalar_t>()[i] == target);
      // 断言检查 a4 中第 i 个位置的值是否等于目标值
      ASSERT(a4.const_data_ptr<double>()[i] == target);
    }
  });
// 定义一个测试用例 ApplyUtilsTest.Contiguous2D，测试应用工具在 2 维小连续数组上的功能
TEST(ApplyUtilsTest, Contiguous2D) {
  // 设置随机种子为 123
  manual_seed(123);
  // 调用测试函数 test，传入 CPU(kDouble) 和数组维度 {2, 1}，并指定特定参数 -1 和 -1
  test(CPU(kDouble), {2, 1}, -1, -1);
}

// 定义一个测试用例 ApplyUtilsTest.Small2D，测试应用工具在 2 维小数组上的功能
TEST(ApplyUtilsTest, Small2D) {
  // 设置随机种子为 123
  manual_seed(123);
  // 调用测试函数 test，传入 CPU(kDouble) 和数组维度 {2, 1}，未指定特定参数
  test(CPU(kDouble), {2, 1});
}

// 定义一个测试用例 ApplyUtilsTest._2D，测试应用工具在 2 维数组上的功能
TEST(ApplyUtilsTest, _2D) {
  // 设置随机种子为 123
  manual_seed(123);
  // 调用测试函数 test，传入 CPU(kDouble) 和数组维度 {20, 10}
  test(CPU(kDouble), {20, 10});
}

// 定义一个测试用例 ApplyUtilsTest._3D，测试应用工具在 3 维数组上的功能
TEST(ApplyUtilsTest, _3D) {
  // 设置随机种子为 123
  manual_seed(123);
  // 调用测试函数 test，传入 CPU(kDouble) 和数组维度 {3, 4, 2}
  test(CPU(kDouble), {3, 4, 2});
}

// 定义一个测试用例 ApplyUtilsTest.Medium3D，测试应用工具在 3 维中等大小数组上的功能
TEST(ApplyUtilsTest, Medium3D) {
  // 设置随机种子为 123
  manual_seed(123);
  // 调用测试函数 test，传入 CPU(kDouble) 和数组维度 {3, 40, 2}
  test(CPU(kDouble), {3, 40, 2});
}

// 定义一个测试用例 ApplyUtilsTest._10D，测试应用工具在 10 维数组上的功能
TEST(ApplyUtilsTest, _10D) {
  // 设置随机种子为 123
  manual_seed(123);
  // 调用测试函数 test，传入 CPU(kDouble) 和数组维度 {3, 4, 2, 5, 2, 1, 3, 4, 2, 3}
  test(CPU(kDouble), {3, 4, 2, 5, 2, 1, 3, 4, 2, 3});
}
```