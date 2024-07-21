# `.\pytorch\test\mobile\nnc\test_context.cpp`

```py
#include <gtest/gtest.h>
#include <torch/csrc/jit/mobile/nnc/context.h>
#include <torch/csrc/jit/mobile/nnc/registry.h>
#include <ATen/Functions.h>

// 定义命名空间，包含移动端神经网络计算相关的功能
namespace torch {
namespace jit {
namespace mobile {
namespace nnc {

// 声明 C 风格的外部函数
extern "C" {

// 慢速乘法内核函数，计算 a 与 n 的乘积，结果存入 out 中
int slow_mul_kernel(void** args) {
  const int size = 128;
  // 从给定的内存块 args[0] 创建大小为 {size} 的浮点数张量 a
  at::Tensor a = at::from_blob(args[0], {size}, at::kFloat);
  // 从给定的内存块 args[1] 创建大小为 {size} 的浮点数张量 out
  at::Tensor out = at::from_blob(args[1], {size}, at::kFloat);
  // 从给定的内存块 args[2] 创建大小为 {1} 的整数张量 n
  at::Tensor n = at::from_blob(args[2], {1}, at::kInt);
  // 从给定的内存块 args[3] 创建大小为 {size} 的浮点数张量 tmp
  at::Tensor tmp = at::from_blob(args[3], {size}, at::kFloat);

  // 将 tmp 张量清零
  tmp.zero_();
  // 执行 a 与 n 之间的乘法运算，将结果累加到 tmp 中
  for (int i = n.item().toInt(); i > 0; i--) {
    tmp.add_(a);
  }
  // 将 tmp 中的数据复制到 out 中
  out.copy_(tmp);
  return 0;
}

// 空白的虚拟内核函数，不执行任何操作
int dummy_kernel(void** /* args */) {
  return 0;
}

} // extern "C"

// 注册慢速乘法内核函数到移动端神经网络计算的内核注册表中
REGISTER_NNC_KERNEL("slow_mul", slow_mul_kernel)
// 注册虚拟内核函数到移动端神经网络计算的内核注册表中
REGISTER_NNC_KERNEL("dummy", dummy_kernel)

// 创建测试输入规格，根据给定的尺寸创建浮点数类型的输入规格
InputSpec create_test_input_spec(const std::vector<int64_t>& sizes) {
  InputSpec input_spec;
  input_spec.sizes_ = sizes;
  input_spec.dtype_ = at::kFloat;
  return input_spec;
}

// 创建测试输出规格，根据给定的尺寸创建浮点数类型的输出规格
OutputSpec create_test_output_spec(const std::vector<int64_t>& sizes) {
  OutputSpec output_spec;
  output_spec.sizes_ = sizes;
  output_spec.dtype_ = at::kFloat;
  return output_spec;
}

// 创建测试内存计划，根据给定的缓冲区尺寸创建内存计划
MemoryPlan create_test_memory_plan(const std::vector<int64_t>& buffer_sizes) {
  MemoryPlan memory_plan;
  memory_plan.buffer_sizes_ = buffer_sizes;
  return memory_plan;
}

// 测试用例，验证慢速乘法函数的执行结果是否符合预期
TEST(Function, ExecuteSlowMul) {
  const int a = 999;
  const int n = 100;
  const int size = 128;
  Function f;

  // 设置要使用的内核函数为 "slow_mul"
  f.set_nnc_kernel_id("slow_mul");
  // 设置输入规格为包含 {size} 尺寸的输入
  f.set_input_specs({create_test_input_spec({size})});
  // 设置输出规格为包含 {size} 尺寸的输出
  f.set_output_specs({create_test_output_spec({size})});
  // 设置参数为一个浮点数类型的列表，包含乘法因子 n
  f.set_parameters(c10::impl::toList(c10::List<at::Tensor>({
      at::ones({1}, at::kInt).mul(n)
  })));
  // 设置内存计划，分配足够存储 {size} 个浮点数的内存
  f.set_memory_plan(create_test_memory_plan({sizeof(float) * size}));

  // 创建输入张量列表，包含一个大小为 {size} 的张量，每个元素都是浮点数 a
  c10::List<at::Tensor> input({
      at::ones({size}, at::kFloat).mul(a)
  });
  // 运行函数 f，并获取输出
  auto outputs = f.run(c10::impl::toList(input));
  // 将输出转换为张量对象
  auto output = ((const c10::IValue&) outputs[0]).toTensor();
  // 创建预期的输出张量，每个元素都是浮点数 a * n
  auto expected_output = at::ones({size}, at::kFloat).mul(a * n);
  // 断言实际输出与预期输出是否相等
  EXPECT_TRUE(output.equal(expected_output));
}

} // namespace nnc
} // namespace mobile
} // namespace jit
} // namespace torch
TEST(Function, Serialization) {
  // 创建一个 Function 对象实例
  Function f;
  // 设置函数名称为 "test_function"
  f.set_name("test_function");
  // 设置 NNC 内核 ID 为 "test_kernel"
  f.set_nnc_kernel_id("test_kernel");
  // 设置输入规格为一个测试输入规格的列表
  f.set_input_specs({create_test_input_spec({1, 3, 224, 224})});
  // 设置输出规格为一个测试输出规格的列表
  f.set_output_specs({create_test_output_spec({1000})});

  // 设置函数的参数为包含三个张量的列表
  f.set_parameters(c10::impl::toList(c10::List<at::Tensor>({
      at::ones({1, 16, 3, 3}, at::kFloat),
      at::ones({16, 32, 1, 1}, at::kFloat),
      at::ones({32, 1, 3, 3}, at::kFloat)
  })));
  // 设置内存计划为一个测试内存计划
  f.set_memory_plan(create_test_memory_plan({
      sizeof(float) * 1024,
      sizeof(float) * 2048,
  }));

  // 序列化 Function 对象
  auto serialized = f.serialize();
  // 从序列化数据中创建新的 Function 对象实例 f2
  Function f2(serialized);
  
  // 断言 f2 的名称为 "test_function"
  EXPECT_EQ(f2.name(), "test_function");
  // 断言 f2 的 NNC 内核 ID 为 "test_kernel"
  EXPECT_EQ(f2.nnc_kernel_id(), "test_kernel");
  // 断言 f2 的输入规格数量为 1
  EXPECT_EQ(f2.input_specs().size(), 1);
  // 断言 f2 的第一个输入规格的大小为 {1, 3, 224, 224}，数据类型为 float
  EXPECT_EQ(f2.input_specs()[0].sizes_, std::vector<int64_t>({1, 3, 224, 224}));
  EXPECT_EQ(f2.input_specs()[0].dtype_, at::kFloat);

  // 断言 f2 的输出规格数量为 1
  EXPECT_EQ(f2.output_specs().size(), 1);
  // 断言 f2 的第一个输出规格的大小为 {1000}，数据类型为 float
  EXPECT_EQ(f2.output_specs()[0].sizes_, std::vector<int64_t>({1000}));
  EXPECT_EQ(f2.output_specs()[0].dtype_, at::kFloat);

  // 断言 f2 的参数数量为 3
  EXPECT_EQ(f2.parameters().size(), 3);
  // 断言 f2 的第一个参数张量的大小为 {1, 16, 3, 3}
  EXPECT_EQ(f2.parameters()[0].toTensor().sizes(), at::IntArrayRef({1, 16, 3, 3}));
  // 断言 f2 的第二个参数张量的大小为 {16, 32, 1, 1}
  EXPECT_EQ(f2.parameters()[1].toTensor().sizes(), at::IntArrayRef({16, 32, 1, 1}));
  // 断言 f2 的第三个参数张量的大小为 {32, 1, 3, 3}
  EXPECT_EQ(f2.parameters()[2].toTensor().sizes(), at::IntArrayRef({32, 1, 3, 3}));

  // 断言 f2 的内存计划的缓冲区大小为 2
  EXPECT_EQ(f2.memory_plan().buffer_sizes_.size(), 2);
  // 断言 f2 的第一个缓冲区大小为 sizeof(float) * 1024
  EXPECT_EQ(f2.memory_plan().buffer_sizes_[0], sizeof(float) * 1024);
  // 断言 f2 的第二个缓冲区大小为 sizeof(float) * 2048
  EXPECT_EQ(f2.memory_plan().buffer_sizes_[1], sizeof(float) * 2048);
}

TEST(Function, ValidInput) {
  // 定义输入大小为 128
  const int size = 128;
  // 创建一个 Function 对象实例
  Function f;
  // 设置 NNC 内核 ID 为 "dummy"
  f.set_nnc_kernel_id("dummy");
  // 设置输入规格为一个测试输入规格的列表
  f.set_input_specs({create_test_input_spec({size})});

  // 创建一个包含一个大小为 128 的 float 类型张量的列表 input
  c10::List<at::Tensor> input({
      at::ones({size}, at::kFloat)
  });
  // 使用 EXPECT_NO_THROW 确保运行 f.run(input) 没有抛出异常
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  EXPECT_NO_THROW(
      f.run(c10::impl::toList(input)));
}

TEST(Function, InvalidInput) {
  // 定义输入大小为 128
  const int size = 128;
  // 创建一个 Function 对象实例
  Function f;
  // 设置 NNC 内核 ID 为 "dummy"
  f.set_nnc_kernel_id("dummy");
  // 设置输入规格为一个测试输入规格的列表
  f.set_input_specs({create_test_input_spec({size})});

  // 创建一个包含一个大小为 256 的 float 类型张量的列表 input
  c10::List<at::Tensor> input({
      at::ones({size * 2}, at::kFloat)
  });
  // 使用 EXPECT_THROW 确保运行 f.run(input) 抛出 c10::Error 异常
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  EXPECT_THROW(
      f.run(c10::impl::toList(input)),
      c10::Error);
}

} // namespace nnc
} // namespace mobile
} // namespace jit
} // namespace torch
```