# `.\pytorch\aten\src\ATen\core\boxing\impl\kernel_stackbased_test.cpp`

```py
// 包含 Google Test 框架的头文件
#include <gtest/gtest.h>
// 包含 ATen 核心的测试辅助函数头文件
#include <ATen/core/boxing/impl/test_helpers.h>

// 包含 ATen 核心的操作注册头文件
#include <ATen/core/op_registration/op_registration.h>
// 包含 ATen 核心的张量类头文件
#include <ATen/core/Tensor.h>
// 包含 Torch 的函数模式解析器头文件
#include <torch/csrc/jit/frontend/function_schema_parser.h>
// 包含 Torch 库头文件
#include <torch/library.h>

// 包含 ATen 核心的旧类型分发头文件
#include <ATen/core/LegacyTypeDispatch.h>

// 使用 c10 命名空间中的 RegisterOperators 类
using c10::RegisterOperators;
// 使用 c10 命名空间中的 DispatchKey 类
using c10::DispatchKey;
// 使用 c10 命名空间中的 Stack 类
using c10::Stack;
// 使用 std 命名空间中的 make_unique 函数
using std::make_unique;
// 使用 c10 命名空间中的 OperatorHandle 类
using c10::OperatorHandle;
// 使用 std 命名空间中的 unique_ptr 类
using std::unique_ptr;

// 匿名命名空间，用于定义内部函数和变量
namespace {

// 错误内核函数，不应被调用
void errorKernel(const OperatorHandle&, Stack* stack) {
  EXPECT_TRUE(false); // this kernel should never be called
}

// 自增内核函数，从栈中取出整数，加一后再推入栈中
void incrementKernel(const OperatorHandle&, Stack* stack) {
  int input = torch::jit::pop(*stack).toInt();
  torch::jit::pop(*stack); // pop the dummy tensor
  torch::jit::push(*stack, input + 1);
}

// 自减内核函数，从栈中取出整数，减一后再推入栈中
void decrementKernel(const OperatorHandle&, Stack* stack) {
  int input = torch::jit::pop(*stack).toInt();
  torch::jit::pop(*stack); // pop the dummy tensor
  torch::jit::push(*stack, input - 1);
}

// 带有 DispatchKeySet 的重新分派内核函数
void redispatchingKernel_with_DispatchKeySet(const OperatorHandle& op, c10::DispatchKeySet ks, Stack* stack) {
  // 这个内核函数是空操作，只是重新分派到优先级较低的内核
  called_redispatching_kernel = true;
  auto updated_ks = ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::TESTING_ONLY_GenericWrapper);
  op.redispatchBoxed(updated_ks, stack);
}

// 预期调用自增操作的函数，使用给定的 DispatchKeySet
void expectCallsIncrement(c10::DispatchKeySet ks) {
  at::AutoDispatchBelowAutograd mode;

  // 断言模式和 CPU 内核是否存在
  auto op = c10::Dispatcher::singleton().findSchema({"_test::my_op", ""});
  ASSERT_TRUE(op.has_value());
  auto result = callOp(*op, dummyTensor(ks), 5);
  EXPECT_EQ(1, result.size());
  EXPECT_EQ(6, result[0].toInt());
}

// 预期调用自增操作的函数，使用给定的 DispatchKey
void expectCallsIncrement(DispatchKey dispatch_key) {
  expectCallsIncrement(c10::DispatchKeySet(dispatch_key));
}

// 预期调用自增操作的非装箱函数，使用给定的 DispatchKey
void expectCallsIncrementUnboxed(DispatchKey dispatch_key) {
  at::AutoDispatchBelowAutograd mode;

  // 断言模式和 CPU 内核是否存在
  auto op = c10::Dispatcher::singleton().findSchema({"_test::my_op", ""});
  ASSERT_TRUE(op.has_value());
  int64_t result = callOpUnboxed<int64_t, at::Tensor, int64_t>(*op, dummyTensor(dispatch_key), 5);
  EXPECT_EQ(6, result);
}

// 预期调用自减操作的函数，使用给定的 DispatchKey
void expectCallsDecrement(DispatchKey dispatch_key) {
  at::AutoDispatchBelowAutograd mode;

  // 断言模式和 CPU 内核是否存在
  auto op = c10::Dispatcher::singleton().findSchema({"_test::my_op", ""});
  ASSERT_TRUE(op.has_value());
  auto result = callOp(*op, dummyTensor(dispatch_key), 5);
  EXPECT_EQ(1, result.size());
  EXPECT_EQ(4, result[0].toInt());
}

// 测试注册并调用基于栈的内核函数
TEST(OperatorRegistrationTestStackBasedKernel, givenKernel_whenRegistered_thenCanBeCalled) {
  // 使用 RegisterOperators 注册一个操作，指定内核为 incrementKernel，使用 CPU DispatchKey
  auto registrar = RegisterOperators().op("_test::my_op(Tensor dummy, int input) -> int", RegisterOperators::options().kernel<&incrementKernel>(DispatchKey::CPU));
  expectCallsIncrement(DispatchKey::CPU);
}

} // end namespace
TEST(OperatorRegistrationTestStackBasedKernel, givenMultipleOperatorsAndKernels_whenRegisteredInOneRegistrar_thenCallsRightKernel) {
  // 创建一个操作符注册器对象 registrar，用于注册多个操作符和对应的内核函数
  auto registrar = RegisterOperators()
      // 注册名为 "_test::my_op" 的操作符，指定 CPU 和 CUDA 的内核函数
      .op("_test::my_op(Tensor dummy, int input) -> int", RegisterOperators::options().kernel<&incrementKernel>(DispatchKey::CPU)
                                                                                      .kernel<&errorKernel>(DispatchKey::CUDA))
      // 注册名为 "_test::error" 的操作符，指定 CPU 和 CUDA 的相同的内核函数 errorKernel
      .op("_test::error(Tensor dummy, int input) -> int", RegisterOperators::options().kernel<&errorKernel>(DispatchKey::CPU)
                                                                                      .kernel<&errorKernel>(DispatchKey::CUDA));
  // 验证调用 DispatchKey::CPU 时是否正确调用了 incrementKernel
  expectCallsIncrement(DispatchKey::CPU);
}

TEST(OperatorRegistrationTestStackBasedKernel, givenMultipleOperatorsAndKernels_whenRegisteredInMultipleRegistrars_thenCallsRightKernel) {
  // 创建第一个操作符注册器对象 registrar1，注册名为 "_test::my_op" 的操作符，指定不同的 CPU 和 CUDA 的内核函数
  auto registrar1 = RegisterOperators().op("_test::my_op(Tensor dummy, int input) -> int", RegisterOperators::options().kernel<&incrementKernel>(DispatchKey::CPU)
                                                                                                                       .kernel<&errorKernel>(DispatchKey::CUDA));
  // 创建第二个操作符注册器对象 registrar2，注册名为 "_test::error" 的操作符，指定相同的 CPU 和 CUDA 的内核函数 errorKernel
  auto registrar2 = RegisterOperators().op("_test::error(Tensor dummy, int input) -> int", RegisterOperators::options().kernel<&errorKernel>(DispatchKey::CPU)
                                                                                                                       .kernel<&errorKernel>(DispatchKey::CUDA));
  // 验证调用 DispatchKey::CPU 时是否正确调用了 incrementKernel
  expectCallsIncrement(DispatchKey::CPU);
}

TEST(OperatorRegistrationTestStackBasedKernel, givenKernel_whenRegistrationRunsOutOfScope_thenCannotBeCalledAnymore) {
  {
    // 创建 Torch 库 _test
    auto m = MAKE_TORCH_LIBRARY(_test);
    // 在 _test 库中定义名为 "_test::my_op" 的操作符
    m.def("_test::my_op(Tensor dummy, int input) -> int");
    // 在 _test::my_op 中的 CPU 实现注册增量内核函数 incrementKernel
    auto m_cpu = MAKE_TORCH_LIBRARY_IMPL(_test, CPU);
    m_cpu.impl("my_op", DispatchKey::CPU, torch::CppFunction::makeFromBoxedFunction<incrementKernel>());
    {
      // 在 _test::my_op 中的 CUDA 实现注册减量内核函数 decrementKernel
      auto m_cuda = MAKE_TORCH_LIBRARY_IMPL(_test, CUDA);
      m_cuda.impl("my_op", DispatchKey::CUDA, torch::CppFunction::makeFromBoxedFunction<decrementKernel>());

      // 断言检查 schema 和 CPU 内核是否存在
      expectCallsIncrement(DispatchKey::CPU);
      expectCallsDecrement(DispatchKey::CUDA);
    }

    // 现在 m_cuda 已经销毁。断言检查 schema 仍然存在，但 CPU 内核不存在
    expectCallsIncrement(DispatchKey::CPU);
    expectDoesntFindKernel("_test::my_op", DispatchKey::CUDA);
  }

  // 现在两个注册器都已销毁。断言检查整个 schema 已经不存在
  expectDoesntFindOperator("_test::my_op");
}

bool called = false;

void kernelWithoutInputs(const OperatorHandle&, Stack*) {
  called = true;
}
TEST(OperatorRegistrationTestStackBasedKernel, givenFallbackKernelWithoutAnyArguments_whenRegistered_thenCanBeCalled) {
  // 注册一个没有张量参数的回退内核。由于没有办法获取分发键，非回退内核无法正常工作。
  // 对于只有回退内核的运算符，这必须为向后兼容性工作。
  auto registrar = RegisterOperators()
      .op("_test::no_tensor_args() -> ()", RegisterOperators::options().catchAllKernel<&kernelWithoutInputs>());

  // 查找操作符架构，确认操作符已注册
  auto op = c10::Dispatcher::singleton().findSchema({"_test::no_tensor_args", ""});
  ASSERT_TRUE(op.has_value());

  // 调用操作符并验证是否被调用
  called = false;
  auto outputs = callOp(*op);
  EXPECT_TRUE(called);
}

void kernelWithoutTensorInputs(const OperatorHandle&, Stack* stack) {
  // 将栈顶元素转换为整数并加一
  stack->back() = stack->back().toInt() + 1;
}

TEST(OperatorRegistrationTestStackBasedKernel, givenFallbackKernelWithoutTensorArguments_whenRegistered_thenCanBeCalled) {
  // 注册一个没有张量参数的回退内核。由于没有办法获取分发键，非回退内核无法正常工作。
  // 对于只有回退内核的运算符，这必须为向后兼容性工作。
  auto registrar = RegisterOperators()
      .op("_test::no_tensor_args(int arg) -> int", RegisterOperators::options().catchAllKernel<&kernelWithoutTensorInputs>());

  // 查找操作符架构，确认操作符已注册
  auto op = c10::Dispatcher::singleton().findSchema({"_test::no_tensor_args", ""});
  ASSERT_TRUE(op.has_value());

  // 调用操作符并验证返回结果
  auto outputs = callOp(*op, 3);
  EXPECT_EQ(1, outputs.size());
  EXPECT_EQ(4, outputs[0].toInt());
}

void kernelForSchemaInference(const OperatorHandle&, Stack* stack) {
  // 空的操作内核函数，用于模式推断
}

TEST(OperatorRegistrationTestStackBasedKernel, givenKernel_whenRegisteredWithoutSpecifyingSchema_thenFailsBecauseItCannotInferFromStackBasedKernel) {
  // 预期抛出异常，因为无法从基于堆栈的内核推断操作符架构
  expectThrows<c10::Error>([] {
      RegisterOperators().op("_test::no_schema_specified", RegisterOperators::options().catchAllKernel<&kernelForSchemaInference>());
  }, "Cannot infer operator schema for this kind of kernel in registration of operator _test::no_schema_specified");
}

TEST(OperatorRegistrationTestStackBasedKernel, givenKernel_whenRegistered_thenCanAlsoBeCalledUnboxed) {
  // 注册一个具有指定分发键的内核
  auto registrar = RegisterOperators().op("_test::my_op(Tensor dummy, int input) -> int", RegisterOperators::options().kernel<&incrementKernel>(DispatchKey::CPU));
  // 期望使用未装箱调用增量内核
  expectCallsIncrementUnboxed(DispatchKey::CPU);
}
TEST(OperatorRegistrationTestStackBasedKernel, callKernelsWithDispatchKeySetConvention_redispatchesToLowerPriorityKernels) {
  // 创建名为 `_test` 的 Torch 库
  auto m = MAKE_TORCH_LIBRARY(_test);
  // 定义名为 `my_op` 的操作，接受 `Tensor dummy` 和 `int input`，返回 `int`
  m.def("my_op(Tensor dummy, int input) -> int");
  // 创建名为 `_` 和 `CPU` 的 Torch 库实现
  auto m_cpu = MAKE_TORCH_LIBRARY_IMPL(_, CPU);
  // 将 `incrementKernel` 函数注册为 CPU 上的回退函数
  m_cpu.fallback(torch::CppFunction::makeFromBoxedFunction<&incrementKernel>());
  // 创建名为 `_` 和 `TESTING_ONLY_GenericWrapper` 的 Torch 库实现
  auto m_testing = MAKE_TORCH_LIBRARY_IMPL(_, TESTING_ONLY_GenericWrapper);
  // 将 `redispatchingKernel_with_DispatchKeySet` 函数注册为 TESTING_ONLY_GenericWrapper 上的回退函数
  m_testing.fallback(torch::CppFunction::makeFromBoxedFunction<&redispatchingKernel_with_DispatchKeySet>());

  // 查找名为 `_test::my_op` 的操作模式
  auto op = c10::Dispatcher::singleton().findSchema({"_test::my_op", ""});
  // 断言找到了相应的操作模式
  ASSERT_TRUE(op.has_value());

  // 创建一个包含 TESTING_ONLY_GenericWrapper 和 CPU 的 DispatchKeySet
  auto testing_cpu_set = c10::DispatchKeySet()
                                    .add(c10::DispatchKey::TESTING_ONLY_GenericWrapper)
                                    .add(c10::DispatchKey::CPU);
  // 初始化一个变量 `called_redispatching_kernel` 并设置为 false
  called_redispatching_kernel = false;

  // 调用 CPU (而不是 TESTING_ONLY_GenericWrapper)
  expectCallsIncrement(DispatchKey::CPU);
  // 断言 `called_redispatching_kernel` 仍然为 false
  ASSERT_FALSE(called_redispatching_kernel);

  // 调用 TESTING_ONLY_GenericWrapper -> 调用 CPU
  expectCallsIncrement(testing_cpu_set);
  // 断言 `called_redispatching_kernel` 现在为 true
  ASSERT_TRUE(called_redispatching_kernel);
}

// 结束测试用例定义
}
```