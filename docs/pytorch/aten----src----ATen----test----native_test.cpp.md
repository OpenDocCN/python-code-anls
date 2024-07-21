# `.\pytorch\aten\src\ATen\test\native_test.cpp`

```
#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <c10/util/irange.h>

using namespace at;

// 定义宏，用于比较两个张量是否相等
#define ASSERT_EQUAL(t1, t2) ASSERT_TRUE(t1.equal(t2));

// 定义宏，用于比较两个张量的所有元素是否在给定的误差范围内相等
#define ASSERT_ALLCLOSE(t1, t2)     \
  ASSERT_TRUE(t1.is_same_size(t2)); \
  ASSERT_TRUE(t1.allclose(t2));

// 定义宏，用于比较两个张量的所有元素是否在给定的误差范围内相等，带有自定义的绝对误差和相对误差
#define ASSERT_ALLCLOSE_TOLERANCES(t1, t2, atol, rtol) \
  ASSERT_TRUE(t1.is_same_size(t2));                    \
  ASSERT_TRUE(t1.allclose(t2, atol, rtol));

// 函数：比较两个张量列表是否相等
void requireEqualTensorList(TensorList t1, TensorList t2) {
  // 断言两个张量列表的长度相等
  ASSERT_EQ(t1.size(), t2.size());
  // 遍历张量列表中的每个张量，并比较它们是否相等
  for (const auto i : c10::irange(t1.size())) {
    ASSERT_EQUAL(t1[i], t2[i]);
  }
}

// 函数：测试张量的 split 方法和 at::split 函数
void TestSplit(TensorOptions T, Tensor& t) {
  // 使用 split 方法进行分割
  auto splitMethod = t.split(1, 0);
  // 使用 at::split 函数进行分割
  auto splitNs = at::split(t, 1, 0);
  // 要求分割后得到的张量列表相等
  requireEqualTensorList(splitMethod, splitNs);

  // 测试通过 cat 方法重建原始张量
  ASSERT_EQUAL(at::cat(splitMethod, 0), t);
}

// 函数：测试张量的 chunk 方法和 at::chunk 函数
void TestChunk(TensorOptions T, Tensor& t) {
  // 使用 chunk 方法进行分块
  auto chunkMethod = t.chunk(3, 0);
  // 使用 at::chunk 函数进行分块
  auto chunkNs = at::chunk(t, 3, 0);
  // 要求分块后得到的张量列表相等
  requireEqualTensorList(chunkMethod, chunkNs);

  // 测试通过 cat 方法重建原始张量
  ASSERT_EQUAL(at::cat(chunkMethod, 0), t);
}

// 类型别名：指定类型的函数指针，用于测试堆栈操作
typedef Tensor StackFunc (TensorList, int64_t);

// 辅助函数：用于测试堆栈操作
void _test_stack(TensorList inputs, int64_t dim, StackFunc stack_func) {
  auto const &x = inputs[0];

  // 使用指定的堆栈函数进行堆栈操作
  auto res = stack_func(inputs, dim);
  // 使用负向的维度索引进行堆栈操作
  auto res_neg = stack_func(inputs, dim - x.dim() - 1);
  // 预期的张量尺寸
  std::vector<int64_t> expected_size;
  expected_size.insert(
      expected_size.end(), x.sizes().begin(), x.sizes().begin() + dim);
  expected_size.insert(expected_size.end(), inputs.size());
  expected_size.insert(
      expected_size.end(), x.sizes().begin() + dim, x.sizes().end());

  // 断言正向和负向的堆栈操作结果相等
  ASSERT_EQUAL(res, res_neg);
  // 断言堆栈后的张量尺寸符合预期
  ASSERT_TRUE(res.sizes().equals(expected_size));

  // 遍历输入张量列表，分别比较堆栈后的张量是否符合预期
  int d = 0;
  for (auto& t : inputs) {
    ASSERT_EQUAL(res.select(dim, d), t);
    d++;
  }
}

// 函数：测试张量的堆栈操作 at::stack, at::native::_stack, at::native::_stack_cpu
void TestStack(TensorOptions T, Tensor& t) {
  { // 测试 at::stack
    auto x = rand({2, 3, 4});
    auto y = rand({2, 3, 4});
    auto z = rand({2, 3, 4});

    auto inputs = {x, y, z};
    // 遍历不同维度进行堆栈操作测试
    for (const auto dim : c10::irange(4)) {
      _test_stack(inputs, dim, at::stack);
    }
  }

  { // 测试 at::native::_stack
    auto x = rand({2, 3, 4});
    auto y = rand({2, 3, 4});
    auto z = rand({2, 3, 4});

    auto inputs = {x, y, z};
    // 遍历不同维度进行堆栈操作测试
    for (const auto dim : c10::irange(4)) {
      _test_stack(inputs, dim, at::native::_stack);
    }
  }

  { // 测试 at::native::_stack_cpu
    auto x = rand({2, 3, 4});
    auto y = rand({2, 3, 4});
    auto z = rand({2, 3, 4});

    auto inputs = {x, y, z};
    // 遍历不同维度进行堆栈操作测试
    for (const auto dim : c10::irange(4)) {
      _test_stack(inputs, dim, at::native::_stack_cpu);
    }
  }
}

// size / stride
void TestSize(TensorOptions T, Tensor& t) {
  auto scalar = randn({}, T);
  // 抛出异常，如果尝试访问维度0但是张量没有维度
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  ASSERT_ANY_THROW(scalar.size(0));
  // 抛出异常，如果尝试访问维度-1但是张量没有维度
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  ASSERT_ANY_THROW(scalar.size(-1));
  // 抛出异常，如果尝试访问步长0但是张量没有维度
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  ASSERT_ANY_THROW(scalar.stride(0));
  // 抛出异常，如果尝试访问步长-1但是张量没有维度
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  ASSERT_ANY_THROW(scalar.stride(-1));

  auto empty = randn({0}, T);
  // 确保空张量的维度0为0
  ASSERT_EQ(empty.size(0), 0);
  // 确保空张量的维度-1为0
  ASSERT_EQ(empty.size(-1), 0);
  // 确保空张量的步长0为1
  ASSERT_EQ(empty.stride(0), 1);
  // 确保空张量的步长-1为1
  ASSERT_EQ(empty.stride(-1), 1);
}
void TestMatmul(TensorOptions T, Tensor& t, TensorOptions AccT) {
  auto scalar = randn({}, T);  // 生成一个标量张量，数据类型为 T
  auto d1 = randn({3}, T);  // 生成一个形状为 {3} 的随机张量，数据类型为 T
  auto d2 = randn({2, 3}, T);  // 生成一个形状为 {2, 3} 的随机张量，数据类型为 T

  // 0-d
  // 抛出异常，要求 matmul 函数的两个参数至少为 1 维
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  ASSERT_ANY_THROW(scalar.matmul(d2));
  // 抛出异常，要求 matmul 函数的两个参数至少为 1 维
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  ASSERT_ANY_THROW(d2.matmul(scalar));

  // 1-d
  ASSERT_ALLCLOSE(d1.matmul(d1), d1.dot(d1));  // 断言 d1 和 d1 的矩阵乘法结果与点积结果相近
  ASSERT_ALLCLOSE(d2.matmul(d1), d2.mv(d1));  // 断言 d2 和 d1 的矩阵乘法结果与矢量乘法结果相近
  auto d1o = randn({2}, T);  // 生成一个形状为 {2} 的随机张量，数据类型为 T
  ASSERT_ALLCLOSE(d1o.matmul(d2), d1o.unsqueeze(0).mm(d2).squeeze(0));  // 断言 d1o 和 d2 的矩阵乘法结果与扩展和挤压后的矩阵乘法结果相近

  // 2-d
  auto d2o = randn({3, 5}, T);  // 生成一个形状为 {3, 5} 的随机张量，数据类型为 T
  ASSERT_ALLCLOSE(d2.matmul(d2o), d2.mm(d2o));  // 断言 d2 和 d2o 的矩阵乘法结果与矩阵乘法结果相近

  // > 2-d, 1-d
  auto d3 = randn({5, 2, 3}, T);  // 生成一个形状为 {5, 2, 3} 的随机张量，数据类型为 T
  ASSERT_ALLCLOSE(
      d3.matmul(d1), d3.bmm(d1.view({1, 3, 1}).expand({5, 3, 1})).view({5, 2}));
  ASSERT_ALLCLOSE(d1o.matmul(d3), d1o.expand({5, 1, 2}).bmm(d3).view({5, 3}));

  auto d5 = randn({3, 2, 4, 2, 3}, T);  // 生成一个形状为 {3, 2, 4, 2, 3} 的随机张量，数据类型为 T
  ASSERT_ALLCLOSE(
      d5.matmul(d1),
      d5.view({24, 2, 3})
          .bmm(d1.view({1, 3, 1}).expand({24, 3, 1}))
          .view({3, 2, 4, 2}));
  ASSERT_ALLCLOSE(
      d1o.matmul(d5),
      d1o.expand({24, 1, 2}).bmm(d5.view({24, 2, 3})).view({3, 2, 4, 3}));

  // > 2-d, 2-d
  // 在此情况下使用“折叠”算法进行矩阵乘法，因此直接与 bmm 的比较无效；而是与高精度计算结果比较（实际上，应始终如此）。容差是经验选择的。
  double atol = 1e-04;
  double rtol = 1e-06;
  d2 = randn({3, 4}, T);  // 重新生成一个形状为 {3, 4} 的随机张量，数据类型为 T
  d2o = randn({4, 2}, T);  // 重新生成一个形状为 {4, 2} 的随机张量，数据类型为 T
  auto result = d5.matmul(d2).to(AccT);  // 计算 d5 和 d2 的矩阵乘法结果，并转换为 AccT 类型

  auto d5Acc = d5.to(AccT);  // 将 d5 张量转换为 AccT 类型
  auto d2Acc = d2.to(AccT);  // 将 d2 张量转换为 AccT 类型
  auto acc_result = d5Acc.view({24, 2, 3})
                        .bmm(d2Acc.expand({24, 3, 4}))
                        .view({3, 2, 4, 2, 4});
  ASSERT_ALLCLOSE_TOLERANCES(result, acc_result, atol, rtol);  // 断言 result 与 acc_result 在给定容差下相近
  ASSERT_ALLCLOSE(
      d2o.matmul(d5),
      d2o.expand({24, 4, 2}).bmm(d5.view({24, 2, 3})).view({3, 2, 4, 4, 3}));

  // > 2-d, > 2-d
  auto d5o = randn({2, 1, 2, 4, 3, 2}, T);  // 生成一个形状为 {2, 1, 2, 4, 3, 2} 的随机张量，数据类型为 T
  auto d5_bmm_view =
      d5.expand({2, 3, 2, 4, 2, 3}).contiguous().view({48, 2, 3});
  auto d5o_bmm_view =
      d5o.expand({2, 3, 2, 4, 3, 2}).contiguous().view({48, 3, 2});
  ASSERT_ALLCLOSE(
      d5.matmul(d5o), d5_bmm_view.bmm(d5o_bmm_view).view({2, 3, 2, 4, 2, 2}));

  // non-expandable case
  auto d5wrong = randn({2, 4, 2, 4, 3, 2}, T);  // 生成一个形状为 {2, 4, 2, 4, 3, 2} 的随机张量，数据类型为 T
  // 抛出异常，要求 d5 和 d5wrong 张量的形状必须匹配
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  ASSERT_ANY_THROW(d5.matmul(d5wrong));
}
// 定义一个函数 TestStandardGammaGrad，用于测试 _standard_gamma_grad 函数的行为
void TestStandardGammaGrad(TensorOptions T, Tensor& t) {
  // 创建一个空张量 empty，形状为 {0}，类型为 T，并用 ones 初始化
  auto empty = ones({0}, T);
  // 断言 _standard_gamma_grad(empty, empty) 返回的结果等于 empty
  ASSERT_EQUAL(empty, at::_standard_gamma_grad(empty, empty));

  // 创建一个标量张量 one_scalar，形状为空，类型为 T，并乘以 5
  auto one_scalar = ones({}, T).mul(5);
  // 创建一个形状为 {1} 的张量 one_with_dim，类型为 T，并乘以 5
  auto one_with_dim = ones({1}, T).mul(5);
  // 断言 _standard_gamma_grad(one_scalar, one_scalar) 的结果近似等于 _standard_gamma_grad(one_with_dim, one_with_dim) 的和
  ASSERT_ALLCLOSE(
      at::_standard_gamma_grad(one_scalar, one_scalar),
      at::_standard_gamma_grad(one_with_dim, one_with_dim).sum());

  // 创建两个形状为 {3, 4} 的随机张量 t1 和 t2，类型为 T 和 kDouble
  auto t1 = randn({3, 4}, T);
  auto t2 = randn({3, 4}, T).toType(kDouble);
  // 断言调用 _standard_gamma_grad(t1, t2) 会抛出异常，异常信息应以 "expected scalar type" 开头
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  ASSERT_ANY_THROW(at::_standard_gamma_grad(t1, t2));
}

// 定义一个函数 TestWhere，用于测试 at::where 函数的行为
void TestWhere(TensorOptions T, Tensor& t) {
  // 创建一个空张量 empty，形状为 {0}，类型为 T
  auto empty = ones({0}, T);
  // 创建一个类型为 kByte 的张量 bT，将 empty 转换为字节类型
  auto bT = T.dtype(kByte);
  auto empty_byte = ones({0}, bT);
  // 断言 where(empty_byte, empty, empty) 的结果等于 empty
  ASSERT_EQUAL(empty, at::where(empty_byte, empty, empty));

  // 创建三个标量张量 x_scalar、y_scalar 和 cond_scalar，类型为 T 和 bT
  auto x_scalar = ones({}, T).mul(5);
  auto y_scalar = ones({}, T).mul(7);
  auto cond_scalar = zeros({}, bT);
  // 创建 x_scalar 和 y_scalar 的一维张量版本 x_1d 和 y_1d
  auto x_1d = x_scalar.unsqueeze(0);
  auto y_1d = y_scalar.unsqueeze(0);
  auto cond_1d = cond_scalar.unsqueeze(0);
  // 断言 where(cond_scalar, x_scalar, y_scalar) 的结果与 where(cond_1d, x_1d, y_1d) 的结果一致
  ASSERT_ALLCLOSE(
      at::where(cond_scalar, x_scalar, y_scalar).unsqueeze(0),
      at::where(cond_1d, x_1d, y_1d));
}

// 定义一个函数 test，用于串联调用多个测试函数
void test(TensorOptions T, TensorOptions AccT) {
  // 创建一个形状为 {3, 3} 的随机张量 t，类型为 T
  auto t = randn({3, 3}, T);
  // 依次调用各个测试函数
  TestSplit(T, t);
  TestChunk(T, t);
  TestStack(T, t);
  TestSize(T, t);
  TestMatmul(T, t, AccT);
  TestStandardGammaGrad(T, t);
  TestWhere(T, t);
}

// 定义一个测试用例 TestNative，测试 CPU 上的函数行为
TEST(TestNative, NativeTestCPU) {
  // 设置随机种子
  manual_seed(123);

  // 调用 test 函数，使用 CPU 设备和浮点类型进行测试
  test(at::device(kCPU).dtype(kFloat),
       at::device(kCPU).dtype(kDouble));
}

// 定义一个测试用例 TestNative，测试 GPU 上的函数行为（如果 CUDA 可用）
TEST(TestNative, NativeTestGPU) {
  // 设置随机种子
  manual_seed(123);

  // 如果 CUDA 可用，则调用 test 函数，使用 CUDA 设备和浮点类型进行测试
  if (at::hasCUDA()) {
    test(at::device(kCUDA).dtype(kFloat),
         at::device(kCUDA).dtype(kDouble));
  }
}
```