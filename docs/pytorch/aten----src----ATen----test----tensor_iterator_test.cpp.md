# `.\pytorch\aten\src\ATen\test\tensor_iterator_test.cpp`

```py
// 包含 Google Test 框架的头文件
#include <gtest/gtest.h>
// 包含线程相关的头文件
#include <thread>

// 包含 ATen 库的头文件
#include <ATen/ATen.h>
// 包含 ATen 库中与 Tensor 迭代相关的头文件
#include <ATen/native/TensorIterator.h>
// 包含 ATen 库中 CPU 循环相关的头文件
#include <ATen/native/cpu/Loops.h>

// 使用 at 命名空间
using namespace at;

// 测试例程：当 CUDA 张量和 CPU 标量进行操作时，应该保持标量在 CPU 上，并将其提升为参数
TEST(TensorIteratorTest, CPUScalar) {
  // 如果没有 CUDA 支持，则退出测试
  if (!at::hasCUDA()) return;

  // 定义输出张量
  Tensor out;
  // 创建一个在 CUDA 设备上的随机张量
  auto x = at::randn({5, 5}, kCUDA);
  // 创建一个在 CPU 上的标量，并通过 squeeze() 方法去除所有尺寸为 1 的维度
  auto y = at::ones(1, kCPU).squeeze();
  // 创建张量迭代器，对 out、x 和 y 进行二元操作
  auto iter = TensorIterator::binary_op(out, x, y);
  // 断言：迭代器中的第一个设备应为 CUDA
  EXPECT_TRUE(iter.device(0).is_cuda()) << "result should be CUDA";
  // 断言：迭代器中的第二个设备应为 CUDA
  EXPECT_TRUE(iter.device(1).is_cuda()) << "x should be CUDA";
  // 断言：迭代器中的第三个设备应为 CPU
  EXPECT_TRUE(iter.device(2).is_cpu()) << "y should be CPU";
}

// 测试例程：验证多个零维 CPU 输入不会被强制转换为 CUDA
TEST(TensorIteratorTest, CPUScalarInputs) {
  // 如果没有 CUDA 支持，则退出测试
  if (!at::hasCUDA()) return;

  // 定义输出张量为一个空的 CUDA 张量
  Tensor out = at::empty({5, 5}, kCUDA);
  // 创建一个在 CPU 上的标量，并通过 squeeze() 方法去除所有尺寸为 1 的维度
  auto x = at::ones(1, kCPU).squeeze();
  // 创建另一个在 CPU 上的标量，并通过 squeeze() 方法去除所有尺寸为 1 的维度
  auto y = at::ones(1, kCPU).squeeze();
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  // 断言：TensorIterator 不支持对含有多个非零维度的 CPU 张量进行二元操作
  ASSERT_ANY_THROW(TensorIterator::binary_op(out, x, y));
}

// 测试例程：混合使用 CPU 和 CUDA 张量应该引发异常（如果 CPU 张量不是零维的）
TEST(TensorIteratorTest, MixedDevices) {
  // 如果没有 CUDA 支持，则退出测试
  if (!at::hasCUDA()) return;

  // 定义输出张量
  Tensor out;
  // 创建一个在 CUDA 设备上的随机张量
  auto x = at::randn({5, 5}, kCUDA);
  // 创建一个在 CPU 上的张量
  auto y = at::ones({5}, kCPU);
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  // 断言：TensorIterator 不支持混合使用 CPU 和 CUDA 张量进行二元操作
  ASSERT_ANY_THROW(TensorIterator::binary_op(out, x, y));
}

// 根据指定的标量类型生成随机张量
Tensor random_tensor_for_type(at::ScalarType scalar_type) {
  // 如果标量类型是浮点类型
  if (at::isFloatingType(scalar_type)) {
    // 返回一个在指定 CPU 设备上的随机张量
    return at::randn({5, 5}, at::device(kCPU).dtype(scalar_type));
  // 如果标量类型是布尔类型
  } else if (scalar_type == kBool) {
    // 返回一个在指定 CPU 设备上的随机整数张量，范围在 [0, 2] 之间
    return at::randint(0, 2, {5, 5}, at::device(kCPU).dtype(scalar_type));
  // 对于其他标量类型
  } else {
    // 返回一个在指定 CPU 设备上的随机整数张量，范围在 [1, 10] 之间
    return at::randint(1, 10, {5, 5}, at::device(kCPU).dtype(scalar_type));
  }
}

// 用于生成给定标量类型的测试例程的宏定义
#define UNARY_TEST_ITER_FOR_TYPE(ctype,name)                                    \
TEST(TensorIteratorTest, SerialLoopUnary_##name) {                              \
  // 定义输出张量
  Tensor out;                                                                   \
  // 创建一个指定类型的随机输入张量
  auto in = random_tensor_for_type(k##name);                                    \
  // 创建一个期望的输出张量，该张量对输入张量进行加 1 操作
  auto expected = in.add(1);                                                    \
  // 创建张量迭代器，对输出张量和输入张量进行一元操作
  auto iter = TensorIterator::unary_op(out, in);                                \
  // 调用 CPU 上的串行内核函数，该函数将每个元素加 1
  at::native::cpu_serial_kernel(iter, [=](ctype a) -> ctype { return a + 1; }); \
  // 断言：输出张量不等于期望的输出张量，即加法操作未成功
  ASSERT_ANY_THROW(out.equal(expected));                                        \
}
// 定义一个名为 SerialLoopUnaryNoOutput_##name 的测试函数，##name 在宏展开时会替换为具体的类型名称后缀
TEST(TensorIteratorTest, SerialLoopUnaryNoOutput_##name) {                     \
  // 为指定类型创建一个随机张量 in
  auto in = random_tensor_for_type(k##name);                                   \
  // 根据输入张量配置创建一个张量迭代器 iter
  auto iter = at::TensorIteratorConfig()                                       \
      .add_owned_input(in)                                                           \
      .build();                                                                \
  // 初始化一个累加器 acc
  int64_t acc = 0;                                                             \
  // 在 CPU 上执行串行核函数，对每个元素进行操作，将 acc 增加
  at::native::cpu_serial_kernel(iter, [&](ctype a) -> void { acc++; }); \
  // 检查 acc 是否等于输入张量的元素总数
  EXPECT_TRUE(acc == in.numel());                                              \
}

// 定义一个名为 SerialLoopBinary_##name 的测试函数，##name 在宏展开时会替换为具体的类型名称后缀
#define BINARY_TEST_ITER_FOR_TYPE(ctype,name)                                            \
TEST(TensorIteratorTest, SerialLoopBinary_##name) {                                      \
  // 创建一个输出张量 out
  Tensor out;                                                                            \
  // 为指定类型创建两个随机输入张量 in1 和 in2
  auto in1 = random_tensor_for_type(k##name);                                            \
  auto in2 = random_tensor_for_type(k##name);                                            \
  // 计算预期输出 expected，即 in1 和 in2 的逐元素相加结果
  auto expected = in1.add(in2);                                                          \
  // 根据输入张量创建一个二元操作的张量迭代器 iter
  auto iter = TensorIterator::binary_op(out, in1, in2);                                  \
  // 在 CPU 上执行串行核函数，对每对元素进行操作，返回 a + b 的结果
  at::native::cpu_serial_kernel(iter, [=](ctype a, ctype b) -> ctype { return a + b; }); \
  // 断言：检查输出张量 out 是否与预期的 expected 不相等，这里使用 ASSERT_ANY_THROW 表示期望抛出异常
  ASSERT_ANY_THROW(out.equal(expected));                                                 \
}

// 定义一个名为 SerialLoopBinaryNoOutput_##name 的测试函数，##name 在宏展开时会替换为具体的类型名称后缀
#define NO_OUTPUT_BINARY_TEST_ITER_FOR_TYPE(ctype,name)                          \
TEST(TensorIteratorTest, SerialLoopBinaryNoOutput_##name) {                      \
  // 为指定类型创建两个随机输入张量 in1 和 in2
  auto in1 = random_tensor_for_type(k##name);                                    \
  auto in2 = random_tensor_for_type(k##name);                                    \
  // 根据输入张量配置创建一个张量迭代器 iter
  auto iter = at::TensorIteratorConfig()                                         \
      .add_owned_input(in1)                                                            \
      .add_owned_input(in2)                                                            \
      .build();                                                                  \
  // 初始化一个累加器 acc
  int64_t acc = 0;                                                               \
  // 在 CPU 上执行串行核函数，对每对元素进行操作，将 acc 增加
  at::native::cpu_serial_kernel(iter, [&](ctype a, ctype b) -> void { acc++; }); \
  // 检查 acc 是否等于输入张量 in1 的元素总数
  EXPECT_TRUE(acc == in1.numel());                                               \
}

// 定义一个名为 POINTWISE_TEST_ITER_FOR_TYPE 的宏，暂未提供具体实现内容
#define POINTWISE_TEST_ITER_FOR_TYPE(ctype,name)                                                      \
// 定义了一个测试宏，用于测试 TensorIterator 的串行循环点对点操作
TEST(TensorIteratorTest, SerialLoopPointwise_##name) {                                                \
  Tensor out;                                                                                         \  // 定义输出张量对象 `out`
  auto in1 = random_tensor_for_type(k##name);                                                         \  // 生成一个随机的指定类型张量 `in1`
  auto in2 = random_tensor_for_type(k##name);                                                         \  // 生成一个随机的指定类型张量 `in2`
  auto in3 = random_tensor_for_type(k##name);                                                         \  // 生成一个随机的指定类型张量 `in3`
  auto expected = in1.add(in2).add(in3);                                                              \  // 计算预期的张量，等于 `in1 + in2 + in3`
  auto iter = at::TensorIteratorConfig()                                                              \  // 创建张量迭代器配置对象 `iter`
      .add_output(out)                                                                                \  // 添加输出张量 `out` 到迭代器配置中
      .add_owned_input(in1)                                                                                 \  // 添加拥有的输入张量 `in1` 到迭代器配置中
      .add_owned_input(in2)                                                                                 \  // 添加拥有的输入张量 `in2` 到迭代器配置中
      .add_owned_input(in3)                                                                                 \  // 添加拥有的输入张量 `in3` 到迭代器配置中
      .build();                                                                                       \  // 构建张量迭代器
  at::native::cpu_serial_kernel(iter, [=](ctype a, ctype b, ctype c) -> ctype { return a + b + c; }); \  // 在 CPU 上执行串行内核函数，对输入张量执行 a + b + c 操作
  ASSERT_ANY_THROW(out.equal(expected));                                                              \  // 断言：确保输出张量 `out` 与预期张量 `expected` 不相等
}

// 定义了一个测试宏，用于测试 TensorIterator 的串行循环点对点操作，但没有输出
#define NO_OUTPUT_POINTWISE_TEST_ITER_FOR_TYPE(ctype,name)                                \
TEST(TensorIteratorTest, SerialLoopPoinwiseNoOutput_##name) {                             \  // 定义测试函数，没有输出的点对点串行循环测试
  auto in1 = random_tensor_for_type(k##name);                                             \  // 生成一个随机的指定类型张量 `in1`
  auto in2 = random_tensor_for_type(k##name);                                             \  // 生成一个随机的指定类型张量 `in2`
  auto in3 = random_tensor_for_type(k##name);                                             \  // 生成一个随机的指定类型张量 `in3`
  auto iter = at::TensorIteratorConfig()                                                  \  // 创建张量迭代器配置对象 `iter`
      .add_owned_input(in1)                                                                     \  // 添加拥有的输入张量 `in1` 到迭代器配置中
      .add_owned_input(in2)                                                                     \  // 添加拥有的输入张量 `in2` 到迭代器配置中
      .add_owned_input(in3)                                                                     \  // 添加拥有的输入张量 `in3` 到迭代器配置中
      .build();                                                                           \  // 构建张量迭代器
  int64_t acc = 0;                                                                        \  // 初始化计数器 `acc` 为 0
  at::native::cpu_serial_kernel(iter, [&](ctype a, ctype b, ctype c) -> void { acc++; }); \  // 在 CPU 上执行串行内核函数，对输入张量执行递增计数
  EXPECT_TRUE(acc == in1.numel());                                                        \  // 断言：确保计数器 `acc` 等于张量 `in1` 的元素数量
}

// 定义了一个测试宏，用于测试比较操作的张量迭代器
// 为了避免无符号类型 (unit, bool) 在减法操作 (b - a) 中发生溢出，
// 我们会先将其转换为 int 类型，然后进行比较操作
#define COMPARISON_TEST_ITER_FOR_TYPE(ctype,name)                                          \
TEST(TensorIteratorTest, ComparisonLoopBinary_##name) {                                    \
  auto in1 = random_tensor_for_type(k##name);                                              \  // 使用给定的数据类型生成一个随机张量in1
  auto in2 = random_tensor_for_type(k##name);                                              \  // 使用给定的数据类型生成另一个随机张量in2
  Tensor out = at::empty({0}, in1.options().dtype(kBool));                                 \  // 创建一个空张量out，数据类型为布尔型
  Tensor diff;                                                                             \  // 定义张量diff，用于存储两个输入张量之间的差异
  if (k##name == kByte || k##name == kBool) {                                              \  // 如果数据类型是字节型或布尔型
    diff = in2.to(kInt).sub(in1.to(kInt));                                                 \  // 将in1和in2转换为整型后相减，结果存储在diff中
  } else {                                                                                 \  // 否则（数据类型为其他类型）
    diff = in2.sub(in1);                                                                   \  // 直接计算in2减去in1的差异，结果存储在diff中
  }                                                                                        \
  auto expected = diff.clamp_min(0).to(kBool);                                             \  // 将diff中的每个元素都截断至大于等于0，然后转换为布尔型，得到期望的输出
  auto iter = TensorIterator::comparison_op(out, in1, in2);                                \  // 创建一个张量迭代器，用于执行比较操作，将结果存储在out中
  at::native::cpu_serial_kernel(iter, [=](ctype a, ctype b) -> bool { return a < b; });    \  // 在CPU上使用单线程执行比较操作的核心逻辑
  EXPECT_TRUE(out.equal(expected));                                                        \  // 断言out张量的内容与期望的结果expected相等
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables,hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
AT_FORALL_SCALAR_TYPES(UNARY_TEST_ITER_FOR_TYPE)                                           \  // 对所有标量数据类型执行一元操作的测试迭代器
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables,hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
AT_FORALL_SCALAR_TYPES(BINARY_TEST_ITER_FOR_TYPE)                                          \  // 对所有标量数据类型执行二元操作的测试迭代器
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables,hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
AT_FORALL_SCALAR_TYPES(POINTWISE_TEST_ITER_FOR_TYPE)                                       \  // 对所有标量数据类型执行逐点操作的测试迭代器
AT_FORALL_SCALAR_TYPES(NO_OUTPUT_UNARY_TEST_ITER_FOR_TYPE)                                 \  // 对所有标量数据类型执行无输出的一元操作的测试迭代器
AT_FORALL_SCALAR_TYPES(NO_OUTPUT_BINARY_TEST_ITER_FOR_TYPE)                                \  // 对所有标量数据类型执行无输出的二元操作的测试迭代器
AT_FORALL_SCALAR_TYPES(NO_OUTPUT_POINTWISE_TEST_ITER_FOR_TYPE)                             \  // 对所有标量数据类型执行无输出的逐点操作的测试迭代器
AT_FORALL_SCALAR_TYPES_AND(Bool, COMPARISON_TEST_ITER_FOR_TYPE)                             \  // 对所有标量数据类型及布尔型执行比较操作的测试迭代器

TEST(TensorIteratorTest, SerialLoopSingleThread) {                                          \
  std::thread::id thread_id = std::this_thread::get_id();                                   \  // 获取当前线程的ID
  Tensor out;                                                                              \  // 定义一个输出张量out
  auto x = at::zeros({50000}, at::TensorOptions(kCPU).dtype(kInt));                         \  // 创建一个包含50000个元素的零张量x，数据类型为整型，存储在CPU上
  auto iter = TensorIterator::unary_op(out, x);                                             \  // 创建一个张量迭代器，用于执行单元操作，将结果存储在out中
  at::native::cpu_serial_kernel(iter, [=](int a) -> int {                                   \  // 在CPU上使用单线程执行核心逻辑
    std::thread::id lambda_thread_id = std::this_thread::get_id();                          \  // 获取lambda函数执行时的线程ID
    EXPECT_TRUE(lambda_thread_id == thread_id);                                             \  // 断言lambda函数执行时的线程ID与最初获取的线程ID相同
    return a + 1;                                                                           \  // 返回a加1的结果
  });                                                                                       \
}

TEST(TensorIteratorTest, InputDType) {                                                      \
  auto iter = at::TensorIteratorConfig()                                                    \  // 创建一个张量迭代器配置对象
      .check_all_same_dtype(false)                                                          \  // 禁用检查所有输入张量是否具有相同数据类型
      .add_owned_output(at::ones({1, 1}, at::dtype(at::kBool)))                             \  // 添加一个拥有的输出张量，全1，数据类型为布尔型
      .add_owned_input(at::ones({1, 1}, at::dtype(at::kFloat)))                             \  // 添加一个拥有的输入张量，全1，数据类型为单精度浮点型
      .add_owned_input(at::ones({1, 1}, at::dtype(at::kDouble)))                            \  // 添加一个拥有的输入张量，全1，数据类型为双精度浮点型
      .build();                                                                             \  // 构建张量迭代器配置
  EXPECT_TRUE(iter.input_dtype() == at::kFloat);                                            \  // 断言迭代器的输入数据类型为单精度浮点型
  EXPECT_TRUE(iter.input_dtype(0) == at::kFloat);                                           \  // 断言迭代器的第一个输入数据类型为单精度浮点型
  EXPECT_TRUE(iter.input_dtype(1) == at::kDouble);                                          \  // 断言迭代器的第二个输入数据类型为双精度浮点型
}
TEST(TensorIteratorTest, ComputeCommonDTypeInputOnly) {
  // 创建一个张量迭代器配置对象，用于处理张量迭代任务
  auto iter = at::TensorIteratorConfig()
      // 添加一个输出张量（所有元素为1的布尔型张量）
      .add_owned_output(at::ones({1, 1}, at::dtype(at::kBool)))
      // 添加一个输入张量（所有元素为1的单精度浮点型张量）
      .add_owned_input(at::ones({1, 1}, at::dtype(at::kFloat)))
      // 添加一个输入张量（所有元素为1的双精度浮点型张量）
      .add_owned_input(at::ones({1, 1}, at::dtype(at::kDouble)))
      // 将所有输入张量提升到共同的数据类型
      .promote_inputs_to_common_dtype(true)
      // 构建迭代器配置
      .build();
  // 断言第一个张量的数据类型为布尔型
  EXPECT_TRUE(iter.dtype(0) == at::kBool);
  // 断言第二个张量的数据类型为双精度浮点型
  EXPECT_TRUE(iter.dtype(1) == at::kDouble);
  // 断言第三个张量的数据类型为双精度浮点型
  EXPECT_TRUE(iter.dtype(2) == at::kDouble);
  // 断言迭代器的共同数据类型为双精度浮点型
  EXPECT_TRUE(iter.common_dtype() == at::kDouble);
}

TEST(TensorIteratorTest, DoNotComputeCommonDTypeInputOnly) {
  // 创建一个张量迭代器配置对象，用于处理张量迭代任务
  auto iter = at::TensorIteratorConfig()
      // 不检查所有张量是否具有相同的数据类型
      .check_all_same_dtype(false)
      // 添加一个输出张量（所有元素为1的长整型张量）
      .add_owned_output(at::ones({1, 1}, at::dtype(at::kLong)))
      // 添加一个输入张量（所有元素为1的单精度浮点型张量）
      .add_owned_input(at::ones({1, 1}, at::dtype(at::kFloat)))
      // 添加一个输入张量（所有元素为1的双精度浮点型张量）
      .add_owned_input(at::ones({1, 1}, at::dtype(at::kDouble)))
      // 构建迭代器配置
      .build();
  // 断言第一个张量的数据类型为长整型
  EXPECT_TRUE(iter.dtype(0) == at::kLong);
  // 断言第二个张量的数据类型为单精度浮点型
  EXPECT_TRUE(iter.dtype(1) == at::kFloat);
  // 断言第三个张量的数据类型为双精度浮点型
  EXPECT_TRUE(iter.dtype(2) == at::kDouble);
}

TEST(TensorIteratorTest, FailNonPromotingBinaryOp) {
  // 定义一个张量对象
  Tensor out;
  // 创建一个张量迭代器配置对象
  at::TensorIteratorConfig config;
  // 添加输出张量
  config.add_output(out);
  // 添加一个输入张量（所有元素为1的双精度浮点型张量）
  config.add_owned_input(at::ones({1,1}, at::dtype(at::kDouble)));
  // 添加一个输入张量（所有元素为1的整型张量）
  config.add_owned_input(at::ones({1,1}, at::dtype(at::kInt)));
  // 抛出异常如果不能建立迭代器配置（非提升二元运算）
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  ASSERT_ANY_THROW(config.build());
}

TEST(TensorIteratorTest, ForEachConstInput) {
  // 创建一个全零张量（长度为10）
  at::Tensor out = at::zeros({10});
  // 创建一个浮点型张量，内容为0到9的序列
  at::Tensor a = at::_lazy_clone(at::arange({10}).to(at::kFloat));
  // 断言张量是否共享数据指针
  EXPECT_TRUE(c10::impl::cow::is_cow_data_ptr(a.storage().data_ptr()));

  // 创建张量迭代器配置对象
  at::TensorIteratorConfig iter_config;
  // 添加输出张量
  iter_config
    .add_output(out)
    // 添加一个常量输入张量
    .add_const_input(a);
  // 构建迭代器
  auto iter = iter_config.build();

  // 自定义循环函数
  auto my_loop = [](char** data, const int64_t* strides, int64_t n) {
    auto* out_data = data[0];
    auto* in_data = data[1];
    for (int64_t i = 0; i < n; i++) {
      // 将输出数据视为浮点数，累加输入数据的浮点数值
      *reinterpret_cast<float*>(out_data) += *reinterpret_cast<float*>(in_data);
      out_data += strides[0];
      in_data += strides[1];
    }
  };

  // 应用自定义循环函数到迭代器
  iter.for_each(my_loop);
  // 断言张量是否共享数据指针
  EXPECT_TRUE(c10::impl::cow::is_cow_data_ptr(a.storage().data_ptr()));
  // 断言输出张量是否与输入张量相等
  EXPECT_TRUE(out.eq(a).all().item<bool>());
}

#define MULTIPLE_OUTPUTS_TEST_ITER_FOR_TYPE(ctype,name)                                             \
// 定义一个单元测试函数，测试 TensorIterator 的多输出情况，模板参数为 name
TEST(TensorIteratorTest, CpuKernelMultipleOutputs_##name) {
  // 创建随机张量 in1 和 in2，数据类型为 k##name
  auto in1 = random_tensor_for_type(k##name);
  auto in2 = random_tensor_for_type(k##name);
  
  // 创建空张量 out1 和 out2，使用 in1 的选项
  Tensor out1 = at::empty({0}, in1.options());
  Tensor out2 = at::empty({0}, in1.options());
  
  // 计算预期输出 expected1 和 expected2
  auto expected1 = in1.add(in2);
  auto expected2 = in1.mul(in2);
  
  // 配置 TensorIteratorConfig 对象 iter，设置输出和输入张量
  auto iter = at::TensorIteratorConfig()
    .add_output(out1)
    .add_output(out2)
    .add_owned_input(in1)
    .add_owned_input(in2)
    .build();
  
  // 调用 native 的 CPU 多输出内核函数，使用 lambda 表达式计算每对输入的加法和乘法结果
  at::native::cpu_kernel_multiple_outputs(iter, [=](ctype a, ctype b) -> std::tuple<ctype, ctype> {
    ctype add = a + b;
    ctype mul = a * b;
    return std::tuple<ctype, ctype>(add, mul);
  });
  
  // 断言 out1 和 out2 与预期结果 expected1 和 expected2 相等
  EXPECT_TRUE(out1.equal(expected1));
  EXPECT_TRUE(out2.equal(expected2));
}
// 针对所有标量类型，生成测试用例 MULTIPLE_OUTPUTS_TEST_ITER_FOR_TYPE
AT_FORALL_SCALAR_TYPES(MULTIPLE_OUTPUTS_TEST_ITER_FOR_TYPE)
```