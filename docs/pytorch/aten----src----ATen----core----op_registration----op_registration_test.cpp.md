# `.\pytorch\aten\src\ATen\core\op_registration\op_registration_test.cpp`

```
/**
 * This file contains some general registration test cases.
 * More detailed test cases containing different APIs for registering kernels
 * are found in other files in this directory.
 */

#include <gtest/gtest.h>

// This file intentionally tests some deprecated APIs
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

#include <ATen/core/boxing/impl/test_helpers.h>
#include <ATen/core/op_registration/op_registration.h>
#include <torch/library.h>
#include <ATen/core/Tensor.h>
#include <functional>

#include <ATen/core/LegacyTypeDispatch.h>

#include <algorithm>

using c10::RegisterOperators;
using c10::OperatorKernel;
using c10::OperatorHandle;
using c10::Dispatcher;
using c10::IValue;
using c10::DispatchKey;

using torch::Library;
using torch::CppFunction;

using at::Tensor;

namespace {

struct DummyKernel final : OperatorKernel {
  void operator()(Tensor) {}
};

struct MockKernel final : OperatorKernel {
  MockKernel(bool* called): called_(called) {}

  void operator()(Tensor) {
    *called_ = true;
  }
private:
  bool* called_;
};

/**
 * @brief Test case to verify registering an operator with schema before defining the kernel in options.
 */
TEST(OperatorRegistrationTest, whenRegisteringWithSchemaBeforeKernelInOptionsObject_thenCanBeCalled) {
  bool called = false;
  auto registrar = c10::RegisterOperators().op(c10::RegisterOperators::options().schema("_test::dummy(Tensor dummy) -> ()").catchAllKernel<MockKernel>(&called));

  auto op = Dispatcher::singleton().findSchema({"_test::dummy", ""});
  ASSERT_TRUE(op.has_value());
  EXPECT_FALSE(called);
  callOp(*op, dummyTensor(c10::DispatchKey::CUDA));
  EXPECT_TRUE(called);
}

/**
 * @brief Test case to verify registering an operator with schema after defining the kernel in options.
 */
TEST(OperatorRegistrationTest, whenRegisteringWithSchemaAfterKernelInOptionsObject_thenCanBeCalled) {
  bool called = false;
  auto registrar = c10::RegisterOperators().op(c10::RegisterOperators::options().catchAllKernel<MockKernel>(&called).schema("_test::dummy(Tensor dummy) -> ()"));

  auto op = Dispatcher::singleton().findSchema({"_test::dummy", ""});
  ASSERT_TRUE(op.has_value());
  EXPECT_FALSE(called);
  callOp(*op, dummyTensor(c10::DispatchKey::CUDA));
  EXPECT_TRUE(called);
}

/**
 * @brief Test case to verify registering an operator with name only before defining the kernel in options.
 */
TEST(OperatorRegistrationTest, whenRegisteringWithNameBeforeKernelInOptionsObject_thenCanBeCalled) {
  bool called = false;
  auto registrar = c10::RegisterOperators().op(c10::RegisterOperators::options().schema("_test::dummy").catchAllKernel<MockKernel>(&called));

  auto op = Dispatcher::singleton().findSchema({"_test::dummy", ""});
  ASSERT_TRUE(op.has_value());
  EXPECT_FALSE(called);
  callOp(*op, dummyTensor(c10::DispatchKey::CUDA));
  EXPECT_TRUE(called);
}

} // namespace
TEST(OperatorRegistrationTest, whenRegisteringWithNameAfterKernelInOptionsObject_thenCanBeCalled) {
  // 标记是否已调用的标志位，初始为 false
  bool called = false;
  // 注册操作符并设置捕获所有内核，使用 MockKernel，并指定调用后将 called 设为 true，同时指定 schema 为 "_test::dummy"
  auto registrar = c10::RegisterOperators().op(c10::RegisterOperators::options().catchAllKernel<MockKernel>(&called).schema("_test::dummy"));

  // 查找并获取 "_test::dummy" 的操作模式
  auto op = Dispatcher::singleton().findSchema({"_test::dummy", ""});
  // 断言操作模式已找到
  ASSERT_TRUE(op.has_value());
  // 验证在调用前 called 为 false
  EXPECT_FALSE(called);
  // 调用操作模式 op，传入一个 CUDA DispatchKey 的虚拟张量
  callOp(*op, dummyTensor(c10::DispatchKey::CUDA));
  // 验证调用后 called 变为 true
  EXPECT_TRUE(called);
}

TEST(OperatorRegistrationTest, whenRegisteringWithoutSchema_thenFails) {
  // 期望抛出 c10::Error 异常，因为注册操作符时没有指定 schema 或操作符名称
  expectThrows<c10::Error>([] {
    c10::RegisterOperators().op(c10::RegisterOperators::options().catchAllKernel<DummyKernel>());
  }, "In operator registration: Tried to register an operator without specifying a schema or operator name.");
}

TEST(OperatorRegistrationTest, whenCallingOpWithWrongDispatchKey_thenFails) {
  // 注册操作符，并指定 kernel 类型为 DummyKernel，但使用了错误的 DispatchKey (CPU)，应该失败
  auto registrar = c10::RegisterOperators().op("_test::dummy(Tensor dummy) -> ()", c10::RegisterOperators::options().kernel<DummyKernel>(c10::DispatchKey::CPU));

  // 查找并获取 "_test::dummy" 的操作模式
  auto op = Dispatcher::singleton().findSchema({"_test::dummy", ""});
  // 断言操作模式已找到
  ASSERT_TRUE(op.has_value());
  // 期望抛出 c10::Error 异常，因为尝试在 CUDA 后端运行 "_test::dummy" 操作
  expectThrows<c10::Error>([&] {
    callOp(*op, dummyTensor(c10::DispatchKey::CUDA));
  }, "Could not run '_test::dummy' with arguments from the 'CUDA'"
  " backend.");
}

TEST(OperatorRegistrationTest, givenOpWithCatchallKernel_whenCallingOp_thenCallsCatchallKernel) {
  // 标记是否已调用的标志位，初始为 false
  bool called = false;
  // 注册操作符，并指定捕获所有内核为 MockKernel，同时设定在调用后将 called 设为 true
  auto registrar = c10::RegisterOperators().op("_test::dummy(Tensor dummy) -> ()", c10::RegisterOperators::options().catchAllKernel<MockKernel>(&called));

  // 查找并获取 "_test::dummy" 的操作模式
  auto op = Dispatcher::singleton().findSchema({"_test::dummy", ""});
  // 断言操作模式已找到
  ASSERT_TRUE(op.has_value());
  // 验证在调用前 called 为 false
  EXPECT_FALSE(called);
  // 调用操作模式 op，传入一个 CUDA DispatchKey 的虚拟张量
  callOp(*op, dummyTensor(c10::DispatchKey::CUDA));
  // 验证调用后 called 变为 true
  EXPECT_TRUE(called);
}

// TODO Rewrite (since this is now allowed) and reenable
// TEST(OperatorRegistrationTest, givenOpWithCatchallKernel_whenRegisteringDispatchedKernel_thenFails) {
//   bool called = false;
//   auto registrar = c10::RegisterOperators().op("_test::dummy(Tensor dummy) -> ()", c10::RegisterOperators::options().catchAllKernel<MockKernel>(&called));
//   expectThrows<c10::Error>([&] {
//     c10::RegisterOperators().op("_test::dummy(Tensor dummy) -> ()", c10::RegisterOperators::options().kernel<MockKernel>(c10::DispatchKey::CPU, &called));
//   }, "for an operator which already has a catch-all kernel registered");
// }

// TEST(OperatorRegistrationTest, givenOpWithCatchallKernel_whenRegisteringDispatchedKernelInSameOpCall_thenFails) {
//   bool called = false;
//   expectThrows<c10::Error>([&] {
//     auto registrar = c10::RegisterOperators().op("_test::dummy(Tensor dummy) -> ()", c10::RegisterOperators::options()
//       .catchAllKernel<MockKernel>(&called)
//       .kernel<MockKernel>(c10::DispatchKey::CPU, &called));
//   }, "for an operator which already has a catch-all kernel registered");
// }
TEST(OperatorRegistrationTest, givenOpWithCatchallKernelOutOfScope_whenRegisteringDispatchedKernelAndCallingOp_thenCallsCatchallKernel) {
  bool called = false;
  {
    // 在作用域内注册操作符 "_test::dummy(Tensor dummy) -> ()"，使用 MockKernel 作为 catch-all kernel，并将 called 作为标记传递
    auto registrar = c10::RegisterOperators().op("_test::dummy(Tensor dummy) -> ()", c10::RegisterOperators::options().catchAllKernel<MockKernel>(&called));
  }

  // 在作用域外注册操作符 "_test::dummy(Tensor dummy) -> ()"，使用 MockKernel 作为 CPU 分发键的 kernel，并将 called 作为标记传递
  auto registrar = c10::RegisterOperators().op("_test::dummy(Tensor dummy) -> ()", c10::RegisterOperators::options().kernel<MockKernel>(c10::DispatchKey::CPU, &called));

  // 查找操作符 "_test::dummy" 的注册信息
  auto op = Dispatcher::singleton().findSchema({"_test::dummy", ""});
  // 确保成功找到操作符的注册信息
  ASSERT_TRUE(op.has_value());
  // 确保在调用操作前，called 标记为 false
  EXPECT_FALSE(called);
  // 调用操作符，并期望 catch-all kernel 被调用
  callOp(*op, dummyTensor(c10::DispatchKey::CPU));
  // 确保调用后，called 标记为 true
  EXPECT_TRUE(called);
}
TEST(OperatorRegistrationTest, givenOpWithoutKernels_whenRegisteringWithSchema_thenOnlyRegistersSchema) {
  // 创建操作注册器，注册不带内核的操作，并返回注册器对象
  auto registrar = c10::RegisterOperators().op("_test::dummy(Tensor dummy) -> ()");

  // 使用调度器查找并获取给定名称和空字符串命名空间的操作模式
  auto op = Dispatcher::singleton().findSchema({"_test::dummy", ""});
  // 断言操作模式已注册
  ASSERT_TRUE(op.has_value()); // assert schema is registered
  // 期望抛出异常，调用具有特定参数的操作，以验证注册是否成功
  expectThrows<c10::Error>([&] {
    callOp(*op, dummyTensor(c10::DispatchKey::CPU));
  }, "Could not run '_test::dummy' with arguments from the 'CPU'"
  " backend.");
}

TEST(OperatorRegistrationTest, givenOpWithoutKernels_whenRegisteringWithoutSchema_thenFails) {
  // 期望抛出异常，因为在操作注册中没有指定内核，无法推断操作模式
  expectThrows<c10::Error>([&] {
    c10::RegisterOperators().op("_test::dummy");
  }, "Cannot infer operator schema in registration of operator _test::dummy because there is no kernel specified.");
}

TEST(OperatorRegistrationTest, givenOpWithoutKernels_whenRunningOutOfScope_thenSchemaIsGone) {
  {
    // 在作用域内创建操作注册器，注册不带内核的操作，并返回注册器对象
    auto registrar = c10::RegisterOperators().op("_test::dummy(Tensor dummy) -> ()");
  }

  // 使用调度器查找并获取给定名称和空字符串命名空间的操作模式
  auto op = Dispatcher::singleton().findSchema({"_test::dummy", ""});
  // 断言操作模式未注册
  EXPECT_FALSE(op.has_value());
}

TEST(OperatorRegistrationTest, givenOpWithoutKernelsWithoutTensorInputs_whenRegistering_thenRegisters) {
  // 只要不注册非全捕获内核，没有张量参数的操作是可以的
  auto registrar = c10::RegisterOperators().op("_test::dummy() -> ()");

  // 使用调度器查找并获取给定名称和空字符串命名空间的操作模式
  auto op = Dispatcher::singleton().findSchema({"_test::dummy", ""});
  // 断言操作模式已注册
  ASSERT_TRUE(op.has_value()); // assert schema is registered
}

TEST(OperatorRegistrationTest, givenMultipleKernelsWithSameDispatchKey_whenRegisteringInSameOpCall_thenFails) {
  // 期望抛出异常，因为在同一操作注册中尝试注册相同调度键的多个内核
  expectThrows<c10::Error>([&] {
    auto registrar = c10::RegisterOperators()
        .op("_test::dummy(Tensor dummy) -> ()", c10::RegisterOperators::options()
            .kernel<DummyKernel>(c10::DispatchKey::CPU)
            .kernel<DummyKernel>(c10::DispatchKey::CPU));
  }, "In operator registration: Tried to register multiple kernels with same dispatch key CPU for operator schema _test::dummy");
}

TEST(OperatorRegistrationTest, givenMultipleCatchallKernels_whenRegisteringInSameOpCall_thenFails) {
  // 期望抛出异常，因为在同一操作注册中尝试注册多个全捕获内核
  expectThrows<c10::Error>([&] {
    auto registrar = c10::RegisterOperators()
        .op("_test::dummy(Tensor dummy) -> ()", c10::RegisterOperators::options()
            .catchAllKernel<DummyKernel>()
            .catchAllKernel<DummyKernel>());
  }, "Tried to register multiple catch-all kernels for operator schema _test::dummy");
}

TEST(OperatorRegistrationTest, whenRegisteringCPUTensorType_thenCanOnlyCallUnboxedWithCPUDispatchKey) {
  // 初始化布尔变量，用于标记是否调用了 CPU 内核
  bool called_kernel_cpu = false;
  // 创建操作注册器，注册带有指定张量参数和 CPU 调度键的操作，并返回注册器对象
  auto registrar= c10::RegisterOperators().op("_test::dummy(Tensor dummy) -> ()", c10::RegisterOperators::options()
  // 使用 MockKernel 类型的 kernel 对象进行注册，指定 CPU 的 dispatch key，并设置回调函数为 called_kernel_cpu
  .kernel<MockKernel>(c10::DispatchKey::CPU, &called_kernel_cpu));

  // 查找名为 "_test::dummy" 的操作的 schema，并断言其已注册
  auto op = Dispatcher::singleton().findSchema({"_test::dummy", ""});
  ASSERT_TRUE(op.has_value()); // 断言 schema 已注册

  // 确保调度器不是从张量中获取 dispatch key，而是直接从直接参数中获取
  called_kernel_cpu = false;
  // 使用预先计算的 dispatch key 集合调用未封装的操作，期望 CPU 的 dispatch key，但提供的是 CUDA 的 dummyTensor
  callOpUnboxedWithPrecomputedDispatchKeySet<void, Tensor>(*op, c10::DispatchKeySet(c10::DispatchKey::CPU), dummyTensor(c10::DispatchKey::CUDA));
  EXPECT_TRUE(called_kernel_cpu);

  // 确保这里不使用张量的 dispatch key
  called_kernel_cpu = false;
  // 期望抛出 c10::Error，调用未封装的操作时使用了 CUDA 的 dispatch key，但提供的是 CPU 的 dummyTensor
  expectThrows<c10::Error>([&] {
    callOpUnboxedWithPrecomputedDispatchKeySet<void, Tensor>(*op, c10::DispatchKeySet(c10::DispatchKey::CUDA), dummyTensor(c10::DispatchKey::CPU));
  }, "Could not run '_test::dummy' with arguments from the 'CUDA'"
     " backend.");
}

// 根据给定的调度键生成相应的后端预期消息字符串
std::string expectedMessageForBackend(DispatchKey key) {
  // 将调度键转换为字符串形式
  std::string key_str(c10::toString(key));
  // 构建并返回特定后端的错误消息字符串
  return "Could not run '_test::dummy' with arguments from the '" + key_str + "' backend";
}

// 测试用例：当在同一操作调用中注册多个内核并调用时，确保调用正确的内核
TEST(OperatorRegistrationTest, whenRegisteringMultipleKernelsInSameOpCallAndCalling_thenCallsCorrectKernel) {
  bool called_kernel1 = false;
  bool called_kernel2 = false;
  // 在注册操作符时，同时为CPU和CUDA调度键注册不同的内核函数
  auto registrar0 = c10::RegisterOperators().op("_test::dummy(Tensor dummy) -> ()", c10::RegisterOperators::options()
    .kernel<MockKernel>(c10::DispatchKey::CPU, &called_kernel1)
    .kernel<MockKernel>(c10::DispatchKey::CUDA, &called_kernel2));

  // 获取操作符模式
  auto op = Dispatcher::singleton().findSchema({"_test::dummy", ""});
  // 断言操作符模式已注册
  ASSERT_TRUE(op.has_value()); // assert schema is registered

  // 重置标志位
  called_kernel1 = called_kernel2 = false;
  // 调用CPU调度键的操作函数，并检查是否调用了正确的内核
  callOp(*op, dummyTensor(c10::DispatchKey::CPU));
  EXPECT_TRUE(called_kernel1);
  EXPECT_FALSE(called_kernel2);

  // 重置标志位
  called_kernel1 = called_kernel2 = false;
  // 调用CUDA调度键的操作函数，并检查是否调用了正确的内核
  callOp(*op, dummyTensor(c10::DispatchKey::CUDA));
  EXPECT_FALSE(called_kernel1);
  EXPECT_TRUE(called_kernel2);

  // 测试对于树外延迟后端的情况，Lazy调度键现在在树中注册到TS后端
  for (c10::DispatchKey key : {c10::DispatchKey::XLA}) {
    // 生成预期的后端消息字符串
    std::string expectMessage = expectedMessageForBackend(key);
    // 断言调用操作函数时抛出特定错误消息
    expectThrows<c10::Error>([&] {
      callOp(*op, dummyTensor(key));
    }, expectMessage.c_str());

    // 断言错误消息中包含可用的张量类型ID，但不检查其顺序
    expectThrows<c10::Error>([&] {
      callOp(*op, dummyTensor(key));
    }, "CPU");
    expectThrows<c10::Error>([&] {
      callOp(*op, dummyTensor(key));
    }, "CUDA");
  }
}

// 全局变量：标识基于堆栈的内核是否已调用
bool called_stackbased_kernel = false;

// 基于堆栈的内核函数
void stackBasedKernel(const OperatorHandle&, c10::Stack* stack) {
  // 标记基于堆栈的内核已被调用
  called_stackbased_kernel = true;
}

// 测试用例：当按名称注册多个内核且无法推断模式时，则失败
TEST(OperatorRegistrationTest, whenRegisteringMultipleKernelsByNameAndNoneCanInferSchema_thenFails) {
  // 断言在注册操作符时，对于无法推断模式的内核，抛出特定错误消息
  expectThrows<c10::Error>([&] {
    auto registrar1 = c10::RegisterOperators().op("_test::dummy", c10::RegisterOperators::options()
      .kernel<&stackBasedKernel>(c10::DispatchKey::CPU)
      .kernel<&stackBasedKernel>(c10::DispatchKey::CUDA)
      .kernel<&stackBasedKernel>(c10::DispatchKey::XLA)
      .kernel<&stackBasedKernel>(c10::DispatchKey::Lazy));
  }, "Cannot infer operator schema for this kind of kernel in registration of operator _test::dummy");
}

// 测试用例：当按模式注册多个内核且无法推断模式时，则成功
TEST(OperatorRegistrationTest, whenRegisteringMultipleKernelsBySchemaAndNoneCanInferSchema_thenSucceeds) {
  bool called_kernel = false;
  // 在注册操作符时，为CPU、CUDA和XLA调度键注册相同的基于堆栈的内核函数
  auto registrar1 = c10::RegisterOperators().op("_test::dummy(Tensor dummy) -> ()", c10::RegisterOperators::options()
    .kernel<&stackBasedKernel>(c10::DispatchKey::CPU)
    .kernel<&stackBasedKernel>(c10::DispatchKey::CUDA)
    .kernel<&stackBasedKernel>(c10::DispatchKey::XLA)
  // 使用给定的堆栈基础内核(kernel<&stackBasedKernel>)注册一个分发调度键为Lazy的操作
  auto op = Dispatcher::singleton().findSchema({"_test::dummy", ""});
  // 断言操作模式已经注册
  ASSERT_TRUE(op.has_value()); // assert schema is registered

  // 重置标志位，用于测试函数调用时是否被正确设置
  called_kernel = called_stackbased_kernel = false;
  // 调用指定操作op，并传入一个CPU调度键的虚拟张量
  callOp(*op, dummyTensor(c10::DispatchKey::CPU));
  // 验证堆栈基础内核被调用
  EXPECT_TRUE(called_stackbased_kernel);
  // 验证普通内核未被调用
  EXPECT_FALSE(called_kernel);

  // 重置标志位
  called_kernel = called_stackbased_kernel = false;
  // 再次调用指定操作op，并传入一个CUDA调度键的虚拟张量
  callOp(*op, dummyTensor(c10::DispatchKey::CUDA));
  // 验证堆栈基础内核被调用
  EXPECT_TRUE(called_stackbased_kernel);
  // 验证普通内核未被调用
  EXPECT_FALSE(called_kernel);

  // 遍历XLA和Lazy两种调度键
  for (c10::DispatchKey key : {c10::DispatchKey::XLA, c10::DispatchKey::Lazy}) {
    // 重置标志位
    called_kernel = called_stackbased_kernel = false;
    // 对每个调度键调用指定操作op，并传入对应的虚拟张量
    callOp(*op, dummyTensor(key));
    // 验证堆栈基础内核被调用
    EXPECT_TRUE(called_stackbased_kernel);
    // 验证普通内核未被调用
    EXPECT_FALSE(called_kernel);
  }
TEST(OperatorRegistrationTest, whenRegisteringMultipleKernelsByNameAndOnlyOneCanInferSchema_thenSucceeds) {
  // 初始化一个标志位，用于检测是否调用了指定的内核
  bool called_kernel = false;
  // 注册运算符，指定运算符名称为 "_test::dummy"
  auto registrar1 = c10::RegisterOperators().op("_test::dummy", c10::RegisterOperators::options()
    // 指定 CPU 上的内核为 stackBasedKernel
    .kernel<&stackBasedKernel>(c10::DispatchKey::CPU)
    // 指定 CUDA 上的内核为 MockKernel，并传入调用标志位
    .kernel<MockKernel>(c10::DispatchKey::CUDA, &called_kernel)
    // 指定 XLA 上的内核为 stackBasedKernel
    .kernel<&stackBasedKernel>(c10::DispatchKey::XLA)
    // 指定 Lazy 上的内核为 stackBasedKernel
    .kernel<&stackBasedKernel>(c10::DispatchKey::Lazy));

  // 查找注册的运算符模式
  auto op = Dispatcher::singleton().findSchema({"_test::dummy", ""});
  // 断言确认模式已注册
  ASSERT_TRUE(op.has_value()); // assert schema is registered

  // 重置内核调用的标志位
  called_kernel = called_stackbased_kernel = false;
  // 调用 CPU 上的 dummyTensor
  callOp(*op, dummyTensor(c10::DispatchKey::CPU));
  // 期望 stackBasedKernel 被调用，MockKernel 未被调用
  EXPECT_TRUE(called_stackbased_kernel);
  EXPECT_FALSE(called_kernel);

  // 重置内核调用的标志位
  called_kernel = called_stackbased_kernel = false;
  // 调用 CUDA 上的 dummyTensor
  callOp(*op, dummyTensor(c10::DispatchKey::CUDA));
  // 期望 MockKernel 被调用，stackBasedKernel 未被调用
  EXPECT_FALSE(called_stackbased_kernel);
  EXPECT_TRUE(called_kernel);

  // 遍历 XLA 和 Lazy 的调度键
  for (c10::DispatchKey key : {c10::DispatchKey::XLA, c10::DispatchKey::Lazy}) {
    // 重置内核调用的标志位
    called_kernel = called_stackbased_kernel = false;
    // 分别调用 XLA 和 Lazy 上的 dummyTensor
    callOp(*op, dummyTensor(key));
    // 期望 stackBasedKernel 被调用，MockKernel 未被调用
    EXPECT_TRUE(called_stackbased_kernel);
    EXPECT_FALSE(called_kernel);
  }
}

TEST(OperatorRegistrationTest, whenRegisteringMultipleKernelsBySchemaAndOnlyOneCanInferSchema_thenSucceeds) {
  // 初始化一个标志位，用于检测是否调用了指定的内核
  bool called_kernel = false;
  // 注册运算符，指定运算符模式为 "_test::dummy(Tensor dummy) -> ()"
  auto registrar1 = c10::RegisterOperators().op("_test::dummy(Tensor dummy) -> ()", c10::RegisterOperators::options()
    // 指定 CPU 上的内核为 stackBasedKernel
    .kernel<&stackBasedKernel>(c10::DispatchKey::CPU)
    // 指定 CUDA 上的内核为 MockKernel，并传入调用标志位
    .kernel<MockKernel>(c10::DispatchKey::CUDA, &called_kernel)
    // 指定 XLA 上的内核为 stackBasedKernel
    .kernel<&stackBasedKernel>(c10::DispatchKey::XLA)
    // 指定 Lazy 上的内核为 stackBasedKernel
    .kernel<&stackBasedKernel>(c10::DispatchKey::Lazy));

  // 查找注册的运算符模式
  auto op = Dispatcher::singleton().findSchema({"_test::dummy", ""});
  // 断言确认模式已注册
  ASSERT_TRUE(op.has_value()); // assert schema is registered

  // 重置内核调用的标志位
  called_kernel = called_stackbased_kernel = false;
  // 调用 CPU 上的 dummyTensor
  callOp(*op, dummyTensor(c10::DispatchKey::CPU));
  // 期望 stackBasedKernel 被调用，MockKernel 未被调用
  EXPECT_TRUE(called_stackbased_kernel);
  EXPECT_FALSE(called_kernel);

  // 重置内核调用的标志位
  called_kernel = called_stackbased_kernel = false;
  // 调用 CUDA 上的 dummyTensor
  callOp(*op, dummyTensor(c10::DispatchKey::CUDA));
  // 期望 MockKernel 被调用，stackBasedKernel 未被调用
  EXPECT_FALSE(called_stackbased_kernel);
  EXPECT_TRUE(called_kernel);

  // 遍历 XLA 和 Lazy 的调度键
  for (c10::DispatchKey key : {c10::DispatchKey::XLA, c10::DispatchKey::Lazy}) {
    // 重置内核调用的标志位
    called_kernel = called_stackbased_kernel = false;
    // 分别调用 XLA 和 Lazy 上的 dummyTensor
    callOp(*op, dummyTensor(key));
    // 期望 stackBasedKernel 被调用，MockKernel 未被调用
    EXPECT_TRUE(called_stackbased_kernel);
    EXPECT_FALSE(called_kernel);
  }
}

struct DummyKernelWithIntParam final : OperatorKernel {
  // 定义一个运算符内核，接受一个 Tensor 和一个 int64_t 参数
  void operator()(Tensor, int64_t) {}
};

TEST(OperatorRegistrationTest, whenRegisteringMismatchingKernelsInSameOpCall_thenFails) {
  // 初始化一个标志位，用于检测是否调用了指定的内核
  bool called_kernel = false;
  // 期望抛出 c10::Error 异常，因为尝试注册不匹配的内核签名
  expectThrows<c10::Error>([&] {
    // 注册运算符 "_test::dummy"，尝试注册 CPU 上的 DummyKernelWithIntParam 和 CUDA 上的 MockKernel
    auto registrar1 = c10::RegisterOperators().op("_test::dummy", c10::RegisterOperators::options()
      .kernel<DummyKernelWithIntParam>(c10::DispatchKey::CPU)
      .kernel<MockKernel>(c10::DispatchKey::CUDA, &called_kernel));
  }, "Mismatch in kernel C++ signatures");
}
void backend_fallback_kernel(const c10::OperatorHandle& op, c10::Stack* stack) {
  // 将栈中索引为1的元素转换为字符串引用，并附加操作符的名称
  (*stack)[1] = (*stack)[1].toStringRef() + op.schema().name();
}

TEST(OperatorRegistrationTest, whenRegisteringBackendFallbackKernel_thenCanBeCalled) {
  // 注册 CPU 分发键的后备内核函数
  auto registrar = c10::Dispatcher::singleton().registerFallback(c10::DispatchKey::CPU, c10::KernelFunction::makeFromBoxedFunction<&backend_fallback_kernel>(), "");

  // 注册一个名为 "_test::dummy" 的操作符
  auto registrar1 = c10::RegisterOperators().op("_test::dummy(Tensor dummy, str input) -> ()");
  // 查找名为 "_test::dummy" 的操作符模式
  auto op = Dispatcher::singleton().findSchema({"_test::dummy", ""});
  ASSERT_TRUE(op.has_value());
  // 调用操作符，并期望返回栈中索引为1的字符串为 "hello _test::dummy"
  auto stack = callOp(*op, dummyTensor(c10::DispatchKey::CPU), "hello ");
  EXPECT_EQ("hello _test::dummy", stack[1].toStringRef());
}

TEST(OperatorRegistrationTest, whenRegisteringBackendFallbackKernelForWrongBackend_thenCannotBeCalled) {
  // 注册 CUDA 分发键的后备内核函数，这里期望会失败
  auto registrar = c10::Dispatcher::singleton().registerFallback(c10::DispatchKey::CUDA, c10::KernelFunction::makeFromBoxedFunction<&backend_fallback_kernel>(), "");

  // 注册一个名为 "_test::dummy" 的操作符
  auto registrar1 = c10::RegisterOperators().op("_test::dummy(Tensor dummy, str input) -> ()");
  // 查找名为 "_test::dummy" 的操作符模式
  auto op = Dispatcher::singleton().findSchema({"_test::dummy", ""});
  ASSERT_TRUE(op.has_value());
  // 期望抛出 c10::Error，指明无法在 'CPU' 后端运行 '_test::dummy'
  expectThrows<c10::Error>([&] {
    auto stack = callOp(*op, dummyTensor(c10::DispatchKey::CPU), "hello ");
  }, "Could not run '_test::dummy' with arguments from the 'CPU' backend.");
}

bool called = false;

TEST(OperatorRegistrationTest, whenRegisteringBackendFallbackKernelAndRegularKernelForDifferentBackend_thenRegularKernelCanBeCalled) {
  // 注册 CPU 分发键的后备内核函数
  auto registrar = c10::Dispatcher::singleton().registerFallback(c10::DispatchKey::CPU, c10::KernelFunction::makeFromBoxedFunction<&backend_fallback_kernel>(), "");

  // 注册一个名为 "_test::dummy" 的操作符，并指定 CUDA 后端的内核函数
  auto registrar1 = c10::RegisterOperators().op("_test::dummy(Tensor dummy, str input) -> ()", c10::RegisterOperators::options()
      .kernel(c10::DispatchKey::CUDA, [] (Tensor, std::string) {
        called = true;
      }));
  // 查找名为 "_test::dummy" 的操作符模式
  auto op = Dispatcher::singleton().findSchema({"_test::dummy", ""});
  ASSERT_TRUE(op.has_value());

  called = false;
  // 调用操作符，并期望正常调用 CUDA 后端的内核函数
  auto stack = callOp(*op, dummyTensor(c10::DispatchKey::CUDA), "hello ");
  EXPECT_TRUE(called);
}

TEST(OperatorRegistrationTest, whenRegisteringBackendFallbackKernelAndRegularKernelForDifferentBackend_thenFallbackKernelCanBeCalled) {
  // 注册 CPU 分发键的后备内核函数
  auto registrar = c10::Dispatcher::singleton().registerFallback(c10::DispatchKey::CPU, c10::KernelFunction::makeFromBoxedFunction<&backend_fallback_kernel>(), "");

  // 注册一个名为 "_test::dummy" 的操作符，并指定 CUDA 后端的内核函数
  auto registrar1 = c10::RegisterOperators().op("_test::dummy(Tensor dummy, str input) -> ()", c10::RegisterOperators::options()
      .kernel(c10::DispatchKey::CUDA, [] (Tensor, std::string) {
        called = true;
      }));
  // 查找名为 "_test::dummy" 的操作符模式
  auto op = Dispatcher::singleton().findSchema({"_test::dummy", ""});
  ASSERT_TRUE(op.has_value());

  called = false;
  // 调用操作符，并期望调用后备内核函数而非 CUDA 后端的内核函数
  auto stack = callOp(*op, dummyTensor(c10::DispatchKey::CPU), "hello ");
  EXPECT_FALSE(called);
  // 期望返回栈中索引为1的字符串为 "hello _test::dummy"
  EXPECT_EQ("hello _test::dummy", stack[1].toStringRef());
}
TEST(OperatorRegistrationTest, whenRegisteringBackendFallbackKernelAndRegularKernelForSameBackend_thenCallsRegularKernel) {
  // 注册一个后备内核和常规内核，当它们针对相同后端注册时，调用常规内核
  auto registrar = c10::Dispatcher::singleton().registerFallback(c10::DispatchKey::CPU, c10::KernelFunction::makeFromBoxedFunction<&backend_fallback_kernel>(), "");

  // 注册一个操作符 "_test::dummy"，指定 CPU 分发键，使用 lambda 函数作为内核实现
  auto registrar1 = c10::RegisterOperators().op("_test::dummy(Tensor dummy, str input) -> ()", c10::RegisterOperators::options()
      .kernel(c10::DispatchKey::CPU, [] (Tensor, std::string) {
        called = true;
      }));

  // 查找并断言操作符 "_test::dummy" 的架构存在
  auto op = Dispatcher::singleton().findSchema({"_test::dummy", ""});
  ASSERT_TRUE(op.has_value());

  // 初始化调用状态标志
  called = false;
  // 调用操作符，传入指定的张量和字符串参数
  auto stack = callOp(*op, dummyTensor(c10::DispatchKey::CPU), "hello ");
  // 断言调用状态为真
  EXPECT_TRUE(called);
}

bool called_autograd = false;
bool called_nonautograd = false;

void nonautograd_kernel(Tensor a) {
  called_nonautograd = true;
}

void autograd_kernel(Tensor a) {
  called_autograd = true;
}

TEST(OperatorRegistrationTest, whenRegisteringAutogradKernel_thenCanCallAutogradKernel) {
  // 注册一个自动求导内核，指定 DispatchKey::Autograd
  auto registrar = c10::RegisterOperators().op("_test::dummy(Tensor dummy) -> ()", c10::RegisterOperators::options()
    .kernel<decltype(autograd_kernel), &autograd_kernel>(DispatchKey::Autograd));

  // 查找并断言操作符 "_test::dummy" 的架构存在
  auto op = Dispatcher::singleton().findSchema({"_test::dummy", ""});
  ASSERT_TRUE(op.has_value());

  // 重置调用状态标志
  called_autograd = false;
  // 预期抛出 c10::Error 异常，因为 CPU 后端无法运行带自动求导参数的操作符
  expectThrows<c10::Error>([&] {
    callOp(*op, dummyTensor(c10::DispatchKey::CPU));
  }, "Could not run '_test::dummy' with arguments from the 'CPU'"
  " backend.");

  // 调用操作符，传入指定的张量参数，带有 requires_grad 标志为 true
  op->typed<void(Tensor)>().call(dummyTensor(DispatchKey::CPU, /*requires_grad=*/true));
  // 断言调用状态为真
  EXPECT_TRUE(called_autograd);
}

TEST(OperatorRegistrationTest, whenRegisteringAutogradKernelWithRegularKernel_thenCanCallAutogradKernel) {
  // 注册一个非自动求导内核和自动求导内核，分别指定 DispatchKey::CPU 和 DispatchKey::Autograd
  auto registrar = c10::RegisterOperators().op("_test::dummy(Tensor dummy) -> ()", c10::RegisterOperators::options()
    .kernel<decltype(nonautograd_kernel), nonautograd_kernel>(DispatchKey::CPU)
    .kernel<decltype(autograd_kernel), &autograd_kernel>(DispatchKey::Autograd));

  // 查找并断言操作符 "_test::dummy" 的架构存在
  auto op = Dispatcher::singleton().findSchema({"_test::dummy", ""});
  ASSERT_TRUE(op.has_value());

  // 重置调用状态标志
  called_nonautograd = called_autograd = false;
  // 调用操作符，传入指定的张量参数，带有 requires_grad 标志为 true
  op->typed<void (Tensor)>().call(dummyTensor(DispatchKey::CPU, /*requires_grad=*/true));
  // 断言非自动求导内核未被调用
  EXPECT_FALSE(called_nonautograd);
  // 断言自动求导内核已被调用
  EXPECT_TRUE(called_autograd);
}

TEST(
    OperatorRegistrationTest,
    `
    // 定义一个测试用例，验证在注册自动微分内核时，使用通用内核可以调用通用内核
    whenRegisteringAutogradKernelWithCatchAllKernel_thenCanCallCatchallKernel) {
      // 创建操作符注册器
      auto registrar = c10::RegisterOperators().op(
          "_test::dummy(Tensor dummy) -> ()",  // 注册一个名为 _test::dummy 的操作符，接受一个 Tensor 类型参数，并返回空
          c10::RegisterOperators::options()
              .catchAllKernel<decltype(nonautograd_kernel), nonautograd_kernel>()  // 使用通用内核注册非自动微分内核
              .kernel<decltype(autograd_kernel), &autograd_kernel>(DispatchKey::Autograd));  // 使用自动微分内核注册自动微分内核
    
      // 查找已注册的操作符模式
      auto op = Dispatcher::singleton().findSchema({"_test::dummy", ""});
      ASSERT_TRUE(op.has_value());  // 断言找到了对应的操作符模式
    
      // 现在 catchAll 映射到 CompositeImplicitAutograd，它比 Autograd 有更高的优先级
      called_nonautograd = called_autograd = false;  // 重置标志位
      // 调用操作符的方法，传入带有 requires_grad=true 的虚拟 Tensor
      op->typed<void(Tensor)>().call(
          dummyTensor(DispatchKey::CPU, /*requires_grad=*/true));
      EXPECT_TRUE(called_nonautograd);  // 断言调用了非自动微分内核
      EXPECT_FALSE(called_autograd);  // 断言未调用自动微分内核
    
      called_nonautograd = called_autograd = false;  // 重置标志位
      // 再次调用操作符的方法，传入不or(DispatchKey::CPU));
      EXPECT_TRUE(called_nonautograd);
      EXPECT_FALSE(called_autograd);
}

// 定义测试函数 OperatorRegistrationTest.AutogradBackendOverridesAutogradKernel
TEST(OperatorRegistrationTest, AutogradBackendOverridesAutogradKernel) {
  // 注册运算符 "_test::dummy(Tensor dummy) -> ()"
  auto registrar = c10::RegisterOperators().op("_test::dummy(Tensor dummy) -> ()", c10::RegisterOperators::options()
    // 使用非自动微分内核注册到 AutogradCPU 调度键
    .kernel<decltype(nonautograd_kernel), &nonautograd_kernel>(DispatchKey::AutogradCPU)
    // 使用自动微分内核注册到 Autograd 调度键
    .kernel<decltype(autograd_kernel), &autograd_kernel>(DispatchKey::Autograd));

  // 查找注册的操作模式
  auto op = Dispatcher::singleton().findSchema({"_test::dummy", ""});
  ASSERT_TRUE(op.has_value());

  // 预期抛出异常：在 CPU 后端无法运行 '_test::dummy' 操作的消息
  expectThrows<c10::Error>([&] {
    callOp(*op, dummyTensor(c10::DispatchKey::CPU));
  }, "Could not run '_test::dummy' with arguments from the 'CPU'"
  " backend.");

  // 预期抛出异常：在 CUDA 后端无法运行 '_test::dummy' 操作的消息
  expectThrows<c10::Error>([&] {
    callOp(*op, dummyTensor(c10::DispatchKey::CUDA));
  }, "Could not run '_test::dummy' with arguments from the 'CUDA'"
  " backend.");

  // 重置标志位为假
  called_nonautograd = called_autograd = false;

  // 调用具有 requires_grad=true 的 CPU 张量版本的操作，并期望调用了非自动微分内核
  op->typed<void (Tensor)>().call(dummyTensor(DispatchKey::CPU, /*requires_grad=*/true));
  EXPECT_TRUE(called_nonautograd);
  EXPECT_FALSE(called_autograd);

  // 重置标志位为假
  called_nonautograd = called_autograd = false;

  // 调用具有 requires_grad=true 的 CUDA 张量版本的操作，并期望调用了自动微分内核
  op->typed<void (Tensor)>().call(dummyTensor(DispatchKey::CUDA, /*requires_grad=*/true));
  EXPECT_TRUE(called_autograd);
  EXPECT_FALSE(called_nonautograd);
}

// 定义函数 LazyBackendsAutogradOverridesAutogradKernel
void LazyBackendsAutogradOverridesAutogradKernel(DispatchKey key) {
  // 注册运算符 "_test::dummy(Tensor dummy) -> ()"
  auto registrar = c10::RegisterOperators().op("_test::dummy(Tensor dummy) -> ()", c10::RegisterOperators::options()
    // 使用非自动微分内核注册到对应自动微分后端的调度键
    .kernel<decltype(nonautograd_kernel), &nonautograd_kernel>(c10::getAutogradKeyFromBackend(toBackendComponent(key)))
    // 使用自动微分内核注册到 Autograd 调度键
    .kernel<decltype(autograd_kernel), &autograd_kernel>(DispatchKey::Autograd));

  // 查找注册的操作模式
  auto op = Dispatcher::singleton().findSchema({"_test::dummy", ""});
  ASSERT_TRUE(op.has_value());

  // 获取预期的错误消息
  std::string expectedMessage = expectedMessageForBackend(key);

  // 预期抛出异常：使用指定后端无法运行 '_test::dummy' 操作的消息
  expectThrows<c10::Error>([&] {
    callOp(*op, dummyTensor(key));
  }, expectedMessage.c_str());

  // 重置标志位为假
  called_nonautograd = called_autograd = false;

  // 调用具有 requires_grad=true 的指定后端张量版本的操作，并期望调用了非自动微分内核
  op->typed<void (Tensor)>().call(dummyTensor(key, /*requires_grad=*/true));
  EXPECT_TRUE(called_nonautograd);
  EXPECT_FALSE(called_autograd);

  // 重置标志位为假
  called_nonautograd = called_autograd = false;

  // 调用具有 requires_grad=true 的 CPU 张量版本的操作，并期望调用了自动微分内核
  op->typed<void (Tensor)>().call(dummyTensor(DispatchKey::CPU, /*requires_grad=*/true));
  EXPECT_TRUE(called_autograd);
  EXPECT_FALSE(called_nonautograd);
}

// 不再测试 ::Lazy 键，因为它现在在树内注册到 TS 后端，并且行为不同，不会抛出预期的 'could not run..' 消息
// 定义测试函数 OperatorRegistrationTest.AutogradXLAOverridesAutogradKernel
TEST(OperatorRegistrationTest, AutogradXLAOverridesAutogradKernel) {
  // 调用 LazyBackendsAutogradOverridesAutogradKernel 函数来测试 AutogradXLAOverridesAutogradKernel
  LazyBackendsAutogradOverridesAutogradKernel(DispatchKey::XLA);
}

// 定义函数 whenRegisterWithLazyBackendsAndCatchAll_AutogradLazyBackendsIsNotFilled
void whenRegisterWithLazyBackendsAndCatchAll_AutogradLazyBackendsIsNotFilled(DispatchKey key) {
  {
    // 注册运算符 "_test::dummy(Tensor dummy) -> ()"，使用非自动微分内核
    auto registrar = c10::RegisterOperators().op("_test::dummy(Tensor dummy) -> ()", c10::RegisterOperators::options()
      .catchAllKernel<decltype(nonautograd_kernel), nonautograd_kernel>());

    // 查找注册的操作模式
    auto op = Dispatcher::singleton().findSchema({"_test::dummy", ""});
    ASSERT_TRUE(op.has_value());
    // 初始化两个布尔变量，用于跟踪是否调用了对应的函数
    called_nonautograd = called_autograd = false;
    // 调用 op 对象的 typed<void (Tensor)>() 方法，传入一个带有 requires_grad=true 的 dummyTensor
    op->typed<void (Tensor)>().call(dummyTensor(key, /*requires_grad=*/true));
    // 断言 called_nonautograd 为真
    EXPECT_TRUE(called_nonautograd);
    // 断言 called_autograd 为假
    EXPECT_FALSE(called_autograd);

    called_nonautograd = called_autograd = false;
    // 再次调用 op 对象的 typed<void (Tensor)>() 方法，传入一个默认参数的 dummyTensor
    op->typed<void (Tensor)>().call(dummyTensor(key));
    // 断言 called_autograd 为假
    EXPECT_FALSE(called_autograd);
    // 断言 called_nonautograd 为真
    EXPECT_TRUE(called_nonautograd);
  }
  {
    // 创建一个操作符注册对象 registrar，注册了 _test::dummy 的两个操作核心
    auto registrar = c10::RegisterOperators().op("_test::dummy(Tensor dummy) -> ()", c10::RegisterOperators::options()
      .kernel<decltype(autograd_kernel), &autograd_kernel>(key)
      .catchAllKernel<decltype(nonautograd_kernel), nonautograd_kernel>());

    // 查找 Dispatcher 中是否存在 _test::dummy 的模式
    auto op = Dispatcher::singleton().findSchema({"_test::dummy", ""});
    // 断言 op 存在值
    ASSERT_TRUE(op.has_value());

    // 当直接注册到 XLA / Lazy 后端时，Autograd{XLA, Lazy} 在预计算中不会捕捉到 catchAll kernel
    // 而是使用后端的回退内核。因此会进入 Autograd{XLA, Lazy}，并调用 XLA / Lazy key 的内核。
    called_nonautograd = called_autograd = false;
    // 调用 op 对象的 typed<void (Tensor)>() 方法，传入一个带有 requires_grad=true 的 dummyTensor
    op->typed<void (Tensor)>().call(dummyTensor(key, /*requires_grad=*/true));
    // 断言 called_nonautograd 为假
    EXPECT_FALSE(called_nonautograd);
    // 断言 called_autograd 为真
    EXPECT_TRUE(called_autograd);

    called_nonautograd = called_autograd = false;
    // 再次调用 op 对象的 typed<void (Tensor)>() 方法，传入一个默认参数的 dummyTensor
    op->typed<void (Tensor)>().call(dummyTensor(key));
    // 断言 called_autograd 为真
    EXPECT_TRUE(called_autograd);
    // 断言 called_nonautograd 为假
    EXPECT_FALSE(called_nonautograd);
  }
TEST(OperatorRegistrationTest, whenRegisterWithXLAKernelAndCatchAll_AutogradXLAIsNotFilled) {
  // 调用 whenRegisterWithLazyBackendsAndCatchAll_AutogradLazyBackendsIsNotFilled 函数，传入 DispatchKey::XLA 参数
  whenRegisterWithLazyBackendsAndCatchAll_AutogradLazyBackendsIsNotFilled(DispatchKey::XLA);
}

TEST(OperatorRegistrationTest, whenRegisterWithLazyKernelAndCatchAll_AutogradLazyIsNotFilled) {
  // 调用 whenRegisterWithLazyBackendsAndCatchAll_AutogradLazyBackendsIsNotFilled 函数，传入 DispatchKey::Lazy 参数
  whenRegisterWithLazyBackendsAndCatchAll_AutogradLazyBackendsIsNotFilled(DispatchKey::Lazy);
}

TEST(OperatorRegistrationTest, whenregisteringwithinvalidoverloadname) {
  // 使用 expectThrows 函数捕获 c10::Error 异常，确保注册操作符时使用了非法的重载名称
  expectThrows<c10::Error>([] {
    auto registrar = c10::RegisterOperators().op("_test::dummy.default", c10::RegisterOperators::options()
      .kernel(DispatchKey::CPU, [] (const int64_t&) {}));
  }, "default is not a legal overload name for aten operators");

  // 使用 expectThrows 函数捕获 c10::Error 异常，确保注册操作符时使用了非法的重载名称
  expectThrows<c10::Error>([] {
    auto registrar = c10::RegisterOperators().op("_test::dummy.__name__", c10::RegisterOperators::options()
      .kernel(DispatchKey::CPU, [] (const int64_t&) {}));
  }, "__name__ is not a legal overload name for aten operators");
}

TEST(OperatorRegistrationTest, givenLambdaKernel_whenRegisteringWithMismatchingCppSignatures_thenFails) {
  // 使用 expectThrows 函数捕获 c10::Error 异常，确保注册操作符时在不同的后端（CPU 和 CUDA）使用了不匹配的 C++ 签名
  expectThrows<c10::Error>([] {
    auto registrar = c10::RegisterOperators().op("_test::dummy", c10::RegisterOperators::options()
      .kernel(DispatchKey::CPU, [] (const int64_t&) {})
      .kernel(DispatchKey::CUDA, [] (int64_t) {}));
  }, "Mismatch in kernel C++ signatures");
}

TEST(OperatorRegistrationTest, givenLambdaKernel_whenRegisteringCatchAllAndBackendWithMismatchingCppSignatures_thenFails) {
  // 使用 expectThrows 函数捕获 c10::Error 异常，确保注册操作符时在捕获所有情况和后端之间使用了不匹配的 C++ 签名
  expectThrows<c10::Error>([] {
    auto registrar = c10::RegisterOperators().op("_test::dummy", c10::RegisterOperators::options()
      .kernel(DispatchKey::CPU, [] (const int64_t&) {})
      .catchAllKernel([] (int64_t) {}));
  }, "Mismatch in kernel C++ signatures");
}

TEST(OperatorRegistrationTest, givenLambdaKernel_whenRegisteringBackendAndCatchAllWithMismatchingCppSignatures_thenFails) {
  // 使用 expectThrows 函数捕获 c10::Error 异常，确保注册操作符时在后端和捕获所有情况之间使用了不匹配的 C++ 签名
  expectThrows<c10::Error>([] {
    auto registrar = c10::RegisterOperators().op("_test::dummy", c10::RegisterOperators::options()
      .catchAllKernel([] (const int64_t&) {})
      .kernel(DispatchKey::CPU, [] (int64_t) {}));
  }, "Mismatch in kernel C++ signatures");
}

TEST(OperatorRegistrationTest, givenLambdaKernel_whenAccessingWithMismatchingCppSignatures_thenFails) {
  // 注册一个带有不匹配 C++ 签名的 lambda 内核操作符
  auto registrar = c10::RegisterOperators().op("_test::dummy", c10::RegisterOperators::options()
    .kernel(DispatchKey::CPU, [] (int64_t) {}));
  // 使用 expectThrows 函数捕获 c10::Error 异常，确保访问或调用操作符时提供了错误的签名
  expectThrows<c10::Error>([] {
    c10::Dispatcher::singleton().findSchemaOrThrow("_test::dummy", "")
      .typed<void(const int64_t&)>();
  }, "Tried to access or call an operator with a wrong signature.\n  operator: _test::dummy(int _0) -> ()");
}

TEST(OperatorRegistrationTest, givenLambdaKernel_whenAccessingCatchAllWithMismatchingCppSignatures_thenFails) {
  // 注册一个带有不匹配 C++ 签名的 lambda 捕获所有情况的内核操作符
  auto registrar = c10::RegisterOperators().op("_test::dummy", c10::RegisterOperators::options()
    .catchAllKernel([] (int64_t) {}));
  // 使用 expectThrows 函数捕获 c10::Error 异常，确保访问或调用捕获所有情况操作符时提供了错误的签名
  expectThrows<c10::Error>([] {
    c10::Dispatcher::singleton().findSchemaOrThrow("_test::dummy", "")
      .typed<void(const int64_t&)>();



// 获取名为 "_test::dummy" 的操作符的调度器单例，并查找其对应的模式或抛出异常
c10::Dispatcher::singleton().findSchemaOrThrow("_test::dummy", "")
      .typed<void(const int64_t&)>();



  }, "Tried to access or call an operator with a wrong signature.\n  operator: _test::dummy(int _0) -> ()");



// 当访问或调用具有错误签名的操作符时抛出异常，并提供错误消息
}, "Tried to access or call an operator with a wrong signature.\n  operator: _test::dummy(int _0) -> ()");
}

TEST(OperatorRegistrationTest, givenTorchLibrary_whenRegisteringWithMismatchingCppSignatures_thenFails) {
  // 创建名为 m 的 Torch 库对象
  auto m = MAKE_TORCH_LIBRARY(_test);
  // 定义名为 "dummy(int a) -> ()" 的函数原型
  m.def("dummy(int a) -> ()");
  // 在 DispatchKey 为 CPU 的情况下注册一个函数实现，该函数不接受参数
  m.impl("dummy", DispatchKey::CPU, [] (int64_t) {});
  // 预期在 CUDA DispatchKey 下注册一个不符合 C++ 签名的函数实现会抛出异常
  expectThrows<c10::Error>([&] {
    m.impl("dummy", DispatchKey::CUDA, [] (const int64_t&) {});
  }, "Mismatch in kernel C++ signatures");
}

TEST(OperatorRegistrationTest, givenTorchLibrary_whenAccessingWithMismatchingCppSignatures_thenFails) {
  // 创建名为 m 的 Torch 库对象
  auto m = MAKE_TORCH_LIBRARY(_test);
  // 定义名为 "dummy(int a) -> ()" 的函数原型
  m.def("dummy(int a) -> ()");
  // 在 DispatchKey 为 CPU 的情况下注册一个函数实现，该函数不接受参数
  m.impl("dummy", DispatchKey::CPU, [] (int64_t) {});
  // 预期尝试访问或调用带有错误签名的运算符时会抛出异常
  expectThrows<c10::Error>([] {
    c10::Dispatcher::singleton().findSchemaOrThrow("_test::dummy", "")
      .typed<void(const int64_t&)>();
  }, "Tried to access or call an operator with a wrong signature.\n  operator: _test::dummy(int a) -> ()");
}

TEST(OperatorRegistrationTest, givenTorchLibrary_whenAccessingCatchAllWithMismatchingCppSignatures_thenFails) {
  // 创建名为 m 的 Torch 库对象
  auto m = MAKE_TORCH_LIBRARY(_test);
  // 定义名为 "dummy(int a) -> ()" 的函数原型，并提供一个接受 int64_t 参数的 Lambda 函数作为实现
  m.def("dummy(int a) -> ()", [] (int64_t) {});
  // 预期尝试访问或调用带有错误签名的 catch-all 运算符时会抛出异常
  expectThrows<c10::Error>([] {
    c10::Dispatcher::singleton().findSchemaOrThrow("_test::dummy", "")
      .typed<void(const int64_t&)>();
  }, "Tried to access or call an operator with a wrong signature.\n  operator: _test::dummy(int a) -> ()");
}

/**
 * This is used to check that a given type works correctly when passed as input
 * to or as output from a kernel.
 *
 * Call ArgTypeTestKernel<Input, Output>::test(input, inputExpectation, output, outputExpectation, schema)
 * to test that a kernel with `Input` as input type and `Output` as output types,
 * when called with `input` fulfills `inputExpectation` inside the kernel, then
 * returns `output` and the returned value fulfills `outputExpectation`.
 *
 * `inputExpectation` and `outputExpectation` should be lambdas that run
 * googletest expect macros (or use other ways to assert the expectation is met).
 *
 * Optionally, you can specify the argument list part of a function schema
 * (e.g. "(Tensor a) -> Tensor") as an additional argument to use when
 * registering the kernel. In this case, the operator registration logic will
 * check that the kernel function signature matches the one you specified.
 */
struct TestModernAPI final {};
struct TestLegacyAPI final {};
struct TestModernAndLegacyAPI final {};

template<class InputType, class OutputType = InputType>
struct ArgTypeTestKernel final : OperatorKernel {
  // 构造函数，接受输入类型 InputType，期望的输入验证函数 inputExpectation，以及输出类型 OutputType
  explicit ArgTypeTestKernel(InputType input, std::function<void(const InputType&)> inputExpectation, OutputType output)
  : input_(std::move(input)), inputExpectation_(std::move(inputExpectation)), output_(std::move(output)) {}

  // 操作符重载，用于执行核函数，接受输入参数 input，返回输出类型 OutputType
  OutputType operator()(InputType input) const {
    // 在核函数内部执行输入验证函数 inputExpectation
    inputExpectation_(std::move(input));
    // 返回 output_ 变量的值
      return output_;
    }
    
    // 测试函数，同时使用现代和旧版 API 进行测试
    static void test(TestModernAndLegacyAPI, InputType input, std::function<void(const InputType&)> inputExpectation, OutputType output, std::function<void(const c10::Stack&)> outputExpectation, const std::string& schema) {
      // 使用现代 API 进行测试
      test(TestModernAPI(), input, inputExpectation, output, outputExpectation, schema);
      // 使用旧版 API 进行测试
      test(TestLegacyAPI(), input, inputExpectation, output, outputExpectation, schema);
    }
    
    // 测试函数，使用现代 API 进行测试
    static void test(TestModernAPI, InputType input, std::function<void(const InputType&)> inputExpectation, OutputType output, std::function<void(const c10::Stack&)> outputExpectation, const std::string& schema) {
      // 调用 test_ 函数注册运算符，使用现代 API
      return test_([&] {
        return c10::RegisterOperators().op("_test::my_op" + schema, c10::RegisterOperators::options().catchAllKernel<ArgTypeTestKernel>(input, inputExpectation, output));
      }, input, inputExpectation, output, outputExpectation, schema);
    }
    
    // 测试函数，使用旧版 API 进行测试
    static void test(TestLegacyAPI, InputType input, std::function<void(const InputType&)> inputExpectation, OutputType output, std::function<void(const c10::Stack&)> outputExpectation, const std::string& schema) {
      // 调用 test_ 函数注册运算符，使用旧版 API
      return test_([&] {
        // 定义匿名函数作为运算符的实现，使用旧版 API
        return c10::RegisterOperators().op("_test::my_op" + schema, [=] (InputType input) -> OutputType {
          inputExpectation(std::move(input));
          return output;
        });
      }, input, inputExpectation, output, outputExpectation, schema);
    }
// 定义一个私有静态方法 test_，用于测试注册操作符的行为
private:
  static void test_(std::function<c10::RegisterOperators()> registration, // 接受一个注册操作符的函数
                    InputType input, // 输入参数类型
                    std::function<void(const InputType&)> inputExpectation, // 输入参数的期望行为函数
                    OutputType output, // 输出参数类型
                    std::function<void(const c10::Stack&)> outputExpectation, // 输出参数的期望行为函数
                    const std::string& schema) // 操作符的描述字符串
  {
    auto registry = registration(); // 调用注册函数获取注册的操作符
    auto op = Dispatcher::singleton().findSchema({"_test::my_op", ""}); // 查找特定名称的操作符模式
    ASSERT_TRUE(op.has_value()); // 断言确保找到了指定的操作符模式

    auto actualOutput = callOp(*op, input); // 调用操作符执行函数，并获得实际输出
    outputExpectation(actualOutput); // 对实际输出进行期望行为的验证
  }

  InputType input_; // 输入参数
  std::function<void(const InputType&)> inputExpectation_; // 输入参数期望行为函数
  OutputType output_; // 输出参数
  std::string schema_; // 操作符的描述字符串
};

template<class InputType, class OutputType = InputType>
struct testArgTypes final {
  template<class APIType = TestModernAndLegacyAPI>
  static void test(InputType input, // 输入参数类型
                   std::function<void(const InputType&)> inputExpectation, // 输入参数的期望行为函数
                   OutputType output, // 输出参数类型
                   std::function<void(const IValue&)> outputExpectation, // 输出参数的期望行为函数
                   const std::string& schema) // 操作符的描述字符串
  {
    // 使用指定的 schema 进行测试
    ArgTypeTestKernel<InputType, OutputType>::test(
      APIType(), input, inputExpectation, output,
      [&] (const c10::Stack& output) { // Lambda 表达式用于验证输出
        EXPECT_EQ(1, output.size()); // 断言确保输出的栈大小为 1
        outputExpectation(output[0]); // 对输出的第一个元素进行期望行为验证
      }, schema
    );

    // 使用推断的 schema 进行测试
    ArgTypeTestKernel<InputType, OutputType>::test(
      APIType(), input, inputExpectation, output,
      [&] (const c10::Stack& output) { // Lambda 表达式用于验证输出
        EXPECT_EQ(1, output.size()); // 断言确保输出的栈大小为 1
        outputExpectation(output[0]); // 对输出的第一个元素进行期望行为验证
      }, ""
    );

    // 测试接收参数并且不返回任何内容的情况
    ArgTypeTestKernel<InputType, std::tuple<>>::test(
      APIType(), input, inputExpectation, {}, 
      [] (const c10::Stack&) {}, "" // 空的期望行为 Lambda 函数和空的 schema
    );

    // 测试接收参数并且返回多个输出的情况
    ArgTypeTestKernel<InputType, std::tuple<int64_t, OutputType>>::test(
      APIType(), input, inputExpectation, std::tuple<int64_t, OutputType>{3, output},
      [&] (const c10::Stack& output) { // Lambda 表达式用于验证输出
        EXPECT_EQ(2, output.size()); // 断言确保输出的栈大小为 2
        EXPECT_EQ(3, output[0].toInt()); // 断言确保第一个输出的整数值为 3
        outputExpectation(output[1]); // 对第二个输出进行期望行为验证
      }, ""
    );
  }
};

// OperatorRegistrationTest 测试用例，验证各种可用的参数类型
TEST(OperatorRegistrationTest, testAvailableArgTypes) {
  // TODO Test Scalar

  // 测试原始类型 double
  testArgTypes<double>::test(
    1.5, [] (const double& v) {EXPECT_EQ(1.5, v);}, // 输入期望和输出期望的 Lambda 函数
    2.5, [] (const IValue& v) {EXPECT_EQ(2.5, v.toDouble());}, // 输入期望和输出期望的 Lambda 函数
    "(float a) -> float"); // 操作符的描述字符串

  // 测试原始类型 int64_t
  testArgTypes<int64_t>::test(
    1, [] (const int64_t& v) {EXPECT_EQ(1, v);}, // 输入期望和输出期望的 Lambda 函数
    2, [] (const IValue& v) {EXPECT_EQ(2, v.toInt());}, // 输入期望和输出期望的 Lambda 函数
    "(int a) -> int"); // 操作符的描述字符串

  // 测试原始类型 bool
  testArgTypes<bool>::test(
    true, [] (const bool& v) {EXPECT_EQ(true, v);}, // 输入期望和输出期望的 Lambda 函数
    false, [] (const IValue& v) {EXPECT_EQ(false, v.toBool());}, // 输入期望和输出期望的 Lambda 函数
    "(bool a) -> bool"); // 操作符的描述字符串

  // 测试原始类型 bool 的另一种情况
  testArgTypes<bool>::test(
    false, [] (const bool& v) {EXPECT_EQ(false, v);}, // 输入期望和输出期望的 Lambda 函数
    true, [] (const IValue& v) {EXPECT_EQ(true, v.toBool());}, // 输入期望和输出期望的 Lambda 函数
    "(bool a) -> bool"); // 操作符的描述字符串

  // 测试原始类型 std::string
  testArgTypes<std::string>::test(
    "string1", [] (const std::string& v) {EXPECT_EQ("string1", v);}, // 输入期望和输出期望的 Lambda 函数
    // 后续代码缺失，截止到此处为止
  // 测试 std::optional<double> 类型的参数
  testArgTypes<std::optional<double>>::test(
    // 使用值为 1.5 的 std::optional<double> 进行测试，验证是否等于 1.5
    std::optional<double>(1.5), [] (const std::optional<double>& v) {EXPECT_EQ(1.5, v.value());},
    // 使用 IValue 类型的值为 2.5 的 std::optional<double> 进行测试，验证是否转换为 double 并等于 2.5
    std::optional<double>(2.5), [] (const IValue& v) {EXPECT_EQ(2.5, v.toDouble());},
    // 函数签名描述为 "(float? a) -> float?"
    "(float? a) -> float?");

  // 测试 std::optional<int64_t> 类型的参数
  testArgTypes<std::optional<int64_t>>::test(
    // 使用值为 1 的 std::optional<int64_t> 进行测试，验证是否等于 1
    std::optional<int64_t>(1), [] (const std::optional<int64_t>& v) {EXPECT_EQ(1, v.value());},
    // 使用 IValue 类型的值为 2 的 std::optional<int64_t> 进行测试，验证是否转换为 int 并等于 2
    std::optional<int64_t>(2), [] (const IValue& v) {EXPECT_EQ(2, v.toInt());},
    // 函数签名描述为 "(int? a) -> int?"
    "(int? a) -> int?");

  // 测试 std::optional<bool> 类型的参数
  testArgTypes<std::optional<bool>>::test(
    // 使用值为 true 的 std::optional<bool> 进行测试，验证是否等于 true
    std::optional<bool>(true), [] (const std::optional<bool>& v) {EXPECT_EQ(true, v.value());},
    // 使用 IValue 类型的值为 false 的 std::optional<bool> 进行测试，验证是否转换为 bool 并等于 false
    std::optional<bool>(false), [] (const IValue& v) {EXPECT_EQ(false, v.toBool());},
    // 函数签名描述为 "(bool? a) -> bool?"
    "(bool? a) -> bool?");

  // 再次测试 std::optional<bool> 类型的参数，这次反向测试
  testArgTypes<std::optional<bool>>::test(
    // 使用值为 false 的 std::optional<bool> 进行测试，验证是否等于 false
    std::optional<bool>(false), [] (const std::optional<bool>& v) {EXPECT_EQ(false, v.value());},
    // 使用 IValue 类型的值为 true 的 std::optional<bool> 进行测试，验证是否转换为 bool 并等于 true
    std::optional<bool>(true), [] (const IValue& v) {EXPECT_EQ(true, v.toBool());},
    // 函数签名描述为 "(bool? a) -> bool?"
    "(bool? a) -> bool?");

  // 测试 std::optional<std::string> 类型的参数
  testArgTypes<std::optional<std::string>>::test(
    // 使用值为 "string1" 的 std::optional<std::string> 进行测试，验证是否等于 "string1"
    std::optional<std::string>("string1"), [] (const std::optional<std::string>& v) {EXPECT_EQ("string1", v.value());},
    // 使用 IValue 类型的值为 "string2" 的 std::optional<std::string> 进行测试，验证是否转换为字符串并等于 "string2"
    std::optional<std::string>("string2"), [] (const IValue& v) {EXPECT_EQ("string2", v.toStringRef());},
    // 函数签名描述为 "(str? a) -> str?"
    "(str? a) -> str?");

  // 测试 std::optional<Tensor> 类型的参数
  testArgTypes<std::optional<Tensor>>::test(
    // 使用 dummyTensor(c10::DispatchKey::CPU) 创建的 std::optional<Tensor> 进行测试，验证 dispatch key 是否为 CPU
    std::optional<Tensor>(dummyTensor(c10::DispatchKey::CPU)), [] (const std::optional<Tensor>& v) {EXPECT_EQ(c10::DispatchKey::CPU, extractDispatchKey(v.value()));},
    // 使用 dummyTensor(c10::DispatchKey::CUDA) 创建的 IValue 类型的 std::optional<Tensor> 进行测试，验证 dispatch key 是否为 CUDA
    std::optional<Tensor>(dummyTensor(c10::DispatchKey::CUDA)), [] (const IValue& v) {EXPECT_EQ(c10::DispatchKey::CUDA, extractDispatchKey(v.toTensor()));},
    // 函数签名描述为 "(Tensor? a) -> Tensor?"
    "(Tensor? a) -> Tensor?");


  // 测试 std::optional<double>, std::optional<int64_t>, std::optional<bool> 类型的参数，值为 c10::nullopt 表示空值的情况
  testArgTypes<std::optional<double>>::test(
    // 使用 c10::nullopt 表示空值的 std::optional<double> 进行测试，验证其没有值
    std::optional<double>(c10::nullopt), [] (const std::optional<double>& v) {EXPECT_FALSE(v.has_value());},
    // 使用 c10::nullopt 表示空值的 IValue 类型的 std::optional<double> 进行测试，验证其为 None
    std::optional<double>(c10::nullopt), [] (const IValue& v) {EXPECT_TRUE(v.isNone());},
    // 函数签名描述为 "(float? a) -> float?"
    "(float? a) -> float?");

  // 测试 std::optional<int64_t> 类型的参数，值为 c10::nullopt 表示空值的情况
  testArgTypes<std::optional<int64_t>>::test(
    // 使用 c10::nullopt 表示空值的 std::optional<int64_t> 进行测试，验证其没有值
    std::optional<int64_t>(c10::nullopt), [] (const std::optional<int64_t>& v) {EXPECT_FALSE(v.has_value());},
    // 使用 c10::nullopt 表示空值的 IValue 类型的 std::optional<int64_t> 进行测试，验证其为 None
    std::optional<int64_t>(c10::nullopt), [] (const IValue& v) {EXPECT_TRUE(v.isNone());},
    // 函数签名描述为 "(int? a) -> int?"
    "(int? a) -> int?");

  // 测试 std::optional<bool> 类型的参数，值为 c10::nullopt 表示空值的情况
  testArgTypes<std::optional<bool>>::test(
    // 使用 c10::nullopt 表示空值的 std::optional<bool> 进行测试，验证其没有值
    std::optional<bool>(c10::nullopt), [] (const std::optional<bool>& v) {EXPECT_FALSE(v.has_value());},
    // 函数签名描述为 "(bool? a) -> bool?"
    "(bool? a) -> bool?");
  // 使用 std::optional<bool> 进行类型测试，首先测试空值的情况
  testArgTypes<std::optional<bool>>::test(
    // 使用 c10::nullopt 创建一个空的 std::optional<bool>，期望断言为 false
    std::optional<bool>(c10::nullopt), [] (const std::optional<bool>& v) {EXPECT_FALSE(v.has_value());},
    // 对应的 IValue 也应为空值
    std::optional<bool>(c10::nullopt), [] (const IValue& v) {EXPECT_TRUE(v.isNone());},
    // 函数签名说明为 "(bool? a) -> bool?"
    "(bool? a) -> bool?");

  // 使用 std::optional<std::string> 进行类型测试，测试空值的情况
  testArgTypes<std::optional<std::string>>::test(
    // 使用 c10::nullopt 创建一个空的 std::optional<std::string>，期望断言为 false
    std::optional<std::string>(c10::nullopt), [] (const std::optional<std::string>& v) {EXPECT_FALSE(v.has_value());},
    // 对应的 IValue 也应为空值
    std::optional<std::string>(c10::nullopt), [] (const IValue& v) {EXPECT_TRUE(v.isNone());},
    // 函数签名说明为 "(str? a) -> str?"
    "(str? a) -> str?");

  // 使用 std::optional<Tensor> 进行类型测试，测试空值的情况
  testArgTypes<std::optional<Tensor>>::test(
    // 使用 c10::nullopt 创建一个空的 std::optional<Tensor>，期望断言为 false
    std::optional<Tensor>(c10::nullopt), [] (const std::optional<Tensor>& v) {EXPECT_FALSE(v.has_value());},
    // 对应的 IValue 也应为空值
    std::optional<Tensor>(c10::nullopt), [] (const IValue& v) {EXPECT_TRUE(v.isNone());},
    // 函数签名说明为 "(Tensor? a) -> Tensor?"
    "(Tensor? a) -> Tensor?");


  // 列表类型的测试（空列表）
  testArgTypes<c10::List<double>>::test(
    // 创建一个空的 c10::List<double>，并期望其大小为 0
    c10::List<double>(), [] (const c10::List<double>& v) {EXPECT_EQ(0, v.size());},
    // 对应的 IValue 应该转换为一个空的 c10::List<double>，并期望其大小为 0
    c10::List<double>(), [] (const IValue& v) {EXPECT_EQ(0, v.to<c10::List<double>>().size());},
    // 函数签名说明为 "(float[] a) -> float[]"
    "(float[] a) -> float[]");

  // 类型为 c10::List<int64_t> 的测试（空列表）
  testArgTypes<c10::List<int64_t>>::test(
    // 创建一个空的 c10::List<int64_t>，并期望其大小为 0
    c10::List<int64_t>(), [] (const c10::List<int64_t>& v) {EXPECT_EQ(0, v.size());},
    // 对应的 IValue 应该转换为一个空的 c10::List<int64_t>，并期望其大小为 0
    c10::List<int64_t>(), [] (const IValue& v) {EXPECT_EQ(0, v.to<c10::List<int64_t>>().size());},
    // 函数签名说明为 "(int[] a) -> int[]"
    "(int[] a) -> int[]");

  // 类型为 c10::List<bool> 的测试（空列表）
  testArgTypes<c10::List<bool>>::test(
    // 创建一个空的 c10::List<bool>，并期望其大小为 0
    c10::List<bool>(), [] (const c10::List<bool>& v) {EXPECT_EQ(0, v.size());},
    // 对应的 IValue 应该转换为一个空的 c10::List<bool>，并期望其大小为 0
    c10::List<bool>(), [] (const IValue& v) {EXPECT_EQ(0, v.to<c10::List<bool>>().size());},
    // 函数签名说明为 "(bool[] a) -> bool[]"
    "(bool[] a) -> bool[]");

  // 类型为 c10::List<std::string> 的测试（空列表）
  testArgTypes<c10::List<std::string>>::test(
    // 创建一个空的 c10::List<std::string>，并期望其大小为 0
    c10::List<std::string>(), [] (const c10::List<std::string>& v) {EXPECT_EQ(0, v.size());},
    // 对应的 IValue 应该转换为一个空的 std::vector<std::string>，并期望其大小为 0
    c10::List<std::string>(), [] (const IValue& v) {EXPECT_EQ(0, v.toListRef().size());},
    // 函数签名说明为 "(str[] a) -> str[]"
    "(str[] a) -> str[]");


  // 列表类型的测试（非空列表）
  testArgTypes<c10::List<double>>::test(
    // 创建一个包含 {1.5, 2.5} 的 c10::List<double>，并检查其是否相等
    c10::List<double>({1.5, 2.5}), [] (const c10::List<double>& v) {expectListEquals({1.5, 2.5}, v);},
    // 对应的 IValue 应该转换为包含 {3.5, 4.5} 的 c10::List<double>，并检查其是否相等
    c10::List<double>({3.5, 4.5}), [] (const IValue& v) {expectListEquals({3.5, 4.5}, v.to<c10::List<double>>());},
    // 函数签名说明为 "(float[] a) -> float[]"
    "(float[] a) -> float[]");

  // 类型为 c10::List<int64_t> 的测试（非空列表）
  testArgTypes<c10::List<int64_t>>::test(
    // 创建一个包含 {1, 2} 的 c10::List<int64_t>，并检查其是否相等
    c10::List<int64_t>({1, 2}), [] (const c10::List<int64_t>& v) {expectListEquals({1, 2}, v);},
    // 对应的 IValue 应该转换为包含 {3, 4} 的 c10::List<int64_t>，并检查其是否相等
    c10::List<int64_t>({3, 4}), [] (const IValue& v) {expectListEquals({3, 4}, v.to<c10::List<int64_t>>());},
    // 函数签名说明为 "(int[] a) -> int[]"
    "(int[] a) -> int[]");

  // 类型为 c10::List<bool> 的测试（非空列表）
  testArgTypes<c10::List<bool>>::test(
    // 创建一个包含 {true, false} 的 c10::List<bool>，并检查其是否相等
    c10::List<bool>({true, false}), [] (const c10::List<bool>& v) {expectListEquals({true, false}, v);},
    // 对应的 IValue 应该转换为包含 {true, false} 的 c10::List<bool>，并检查其是否相等
    c10::List<bool>({true, false}), [] (const IValue& v) {expectListEquals({true, false}, v.to<c10::List<bool>>());},
    // 函数签名说明为 "(bool[] a) -> bool[]"
    "(bool[] a) -> bool[]");

  // 类型为 c10::List<std::string> 的测试（非空列表）的开头部分
  testArgTypes<c10::List<std::string>>::test(
    // 创建一个包含空列表的 c10::List<std::string> 并期望其大小为 0
    c10::List<std::string>(), [] (const c10::List<std::string>& v) {EXPECT_EQ(0, v.size());},
    // IValue 对象转换为 c10::List<std::string> 并且也期望其大小为 0
    c10::List<std::string>(), [] (const IValue& v) {EXPECT_EQ(0, v.toListRef().size());},
    // 函数签名说明为 "(str[] a) -> str[]"
    "(str[] a) -> str[]");
    c10::List<std::string>({"first", "second"}), [] (const c10::List<std::string>& v) {expectListEquals({"first", "second"}, v);},
    // 使用 c10::List<std::string> 初始化一个包含 {"first", "second"} 的列表，并定义 lambda 函数对列表进行断言比较
    c10::List<std::string>({"first", "second"}), [] (const IValue& v) {
      // 对 IValue 类型的对象进行断言比较，确保其包含两个元素，分别为 "first" 和 "second"
      EXPECT_EQ(2, v.toListRef().size());
      EXPECT_EQ("first", v.toListRef()[0].toStringRef());
      EXPECT_EQ("second", v.toListRef()[1].toStringRef());
    },
    "(str[] a) -> str[]");
  // 使用字符串 "(str[] a) -> str[]" 进行测试参数类型

  testArgTypes<c10::List<Tensor>>::test(
    // 使用 c10::List<Tensor> 初始化一个包含两个 Tensor 的列表，并定义 lambda 函数对列表进行断言比较
    c10::List<Tensor>({dummyTensor(c10::DispatchKey::CPU), dummyTensor(c10::DispatchKey::CUDA)}), [] (const c10::List<Tensor>& v) {
      EXPECT_EQ(2, v.size());
      EXPECT_EQ(c10::DispatchKey::CPU, extractDispatchKey(v.get(0)));
      EXPECT_EQ(c10::DispatchKey::CUDA, extractDispatchKey(v.get(1)));
    },
    // 使用 c10::List<Tensor> 初始化另一个列表，并定义 lambda 函数对 IValue 类型对象进行断言比较
    c10::List<Tensor>({dummyTensor(c10::DispatchKey::CUDA), dummyTensor(c10::DispatchKey::CPU)}), [] (const IValue& v) {
      EXPECT_EQ(2, v.to<c10::List<at::Tensor>>().size());
      EXPECT_EQ(c10::DispatchKey::CUDA, extractDispatchKey(v.to<c10::List<at::Tensor>>().get(0)));
      EXPECT_EQ(c10::DispatchKey::CPU, extractDispatchKey(v.to<c10::List<at::Tensor>>().get(1)));
    },
    "(Tensor[] a) -> Tensor[]");

  // ArrayRef list types (with empty list)
  testArgTypes<c10::ArrayRef<double>, c10::List<double>>::test(
    // 使用空的 c10::ArrayRef<double> 和 lambda 函数对其进行断言比较
    c10::ArrayRef<double>(), [] (c10::ArrayRef<double> v) {EXPECT_EQ(0, v.size());},
    // 使用空的 c10::List<double> 和 lambda 函数对 IValue 类型对象进行断言比较
    c10::List<double>(), [] (const IValue& v) {EXPECT_EQ(0, v.to<c10::List<double>>().size());},
    "(float[] a) -> float[]");

  testArgTypes<c10::ArrayRef<int64_t>, c10::List<int64_t>>::test(
    // 使用空的 c10::ArrayRef<int64_t> 和 lambda 函数对其进行断言比较
    c10::ArrayRef<int64_t>(), [] (c10::ArrayRef<int64_t> v) {EXPECT_EQ(0, v.size());},
    // 使用空的 c10::List<int64_t> 和 lambda 函数对 IValue 类型对象进行断言比较
    c10::List<int64_t>(), [] (const IValue& v) {EXPECT_EQ(0, v.to<c10::List<int64_t>>().size());},
    "(int[] a) -> int[]");

  testArgTypes<c10::ArrayRef<std::string>, c10::List<std::string>>::test(
    // 使用空的 c10::ArrayRef<std::string> 和 lambda 函数对其进行断言比较
    c10::ArrayRef<std::string>(), [] (c10::ArrayRef<std::string> v) {EXPECT_EQ(0, v.size());},
    // 使用空的 c10::List<std::string> 和 lambda 函数对 IValue 类型对象进行断言比较
    c10::List<std::string>(), [] (const IValue& v) {EXPECT_EQ(0, v.toListRef().size());},
    "(str[] a) -> str[]");


  // list types (with non-empty list)
  testArgTypes<c10::ArrayRef<double>, c10::List<double>>::test(
    // 使用包含 {1.5, 2.5} 的 c10::ArrayRef<double> 和 lambda 函数对其进行断言比较
    c10::ArrayRef<double>({1.5, 2.5}), [] (c10::ArrayRef<double> v) {expectListEquals({1.5, 2.5}, v);},
    // 使用包含 {3.5, 4.5} 的 c10::List<double> 和 lambda 函数对 IValue 类型对象进行断言比较
    c10::List<double>({3.5, 4.5}), [] (const IValue& v) {expectListEquals({3.5, 4.5}, v.to<c10::List<double>>());},
    "(float[] a) -> float[]");

  testArgTypes<c10::ArrayRef<int64_t>, c10::List<int64_t>>::test(
    // 使用包含 {1, 2} 的 c10::ArrayRef<int64_t> 和 lambda 函数对其进行断言比较
    c10::ArrayRef<int64_t>({1, 2}), [] (c10::ArrayRef<int64_t> v) {expectListEquals({1, 2}, v);},
    // 使用包含 {3, 4} 的 c10::List<int64_t> 和 lambda 函数对 IValue 类型对象进行断言比较
    c10::List<int64_t>({3, 4}), [] (const IValue& v) {expectListEquals({3, 4}, v.to<c10::List<int64_t>>());},
    "(int[] a) -> int[]");

  testArgTypes<c10::ArrayRef<std::string>, c10::List<std::string>>::test(
    // 使用包含 {"first", "second"} 的 c10::ArrayRef<std::string> 和 lambda 函数对其进行断言比较
    c10::ArrayRef<std::string>({"first", "second"}), [] (c10::ArrayRef<std::string> v) {expectListEquals({"first", "second"}, v);},
    c10::List<std::string>({"first", "second"}), [] (const IValue& v) {
      // 检查列表长度是否为2
      EXPECT_EQ(2, v.toListRef().size());
      // 检查第一个元素是否为字符串 "first"
      EXPECT_EQ("first", v.toListRef()[0].toStringRef());
      // 检查第二个元素是否为字符串 "second"
      EXPECT_EQ("second", v.toListRef()[1].toStringRef());
    },
    "(str[] a) -> str[]");
  testArgTypes<c10::ArrayRef<Tensor>, c10::List<Tensor>>::test(
    c10::ArrayRef<Tensor>({dummyTensor(c10::DispatchKey::CPUTensorId), dummyTensor(c10::DispatchKey::CUDATensorId)}), [] (c10::ArrayRef<Tensor> v) {
      // 检查张量数组长度是否为2
      EXPECT_EQ(2, v.size());
      // 检查第一个张量的调度键是否为 CPU
      EXPECT_EQ(c10::DispatchKey::CPUTensorId, extractDispatchKey(v[0]));
      // 检查第二个张量的调度键是否为 CUDA
      EXPECT_EQ(c10::DispatchKey::CUDATensorId, extractDispatchKey(v[1]));
    },
    // 创建包含两个张量的张量列表
    c10::List<Tensor>({dummyTensor(c10::DispatchKey::CUDATensorId), dummyTensor(c10::DispatchKey::CPUTensorId)}), [] (const IValue& v) {
      // 检查张量列表长度是否为2
      EXPECT_EQ(2, v.to<c10::List<at::Tensor>>().size());
      // 检查第一个张量的调度键是否为 CUDA
      EXPECT_EQ(c10::DispatchKey::CUDATensorId, extractDispatchKey(v.to<c10::List<at::Tensor>>().get(0)));
      // 检查第二个张量的调度键是否为 CPU
      EXPECT_EQ(c10::DispatchKey::CPUTensorId, extractDispatchKey(v.to<c10::List<at::Tensor>>().get(1)));
    },
    "(Tensor[] a) -> Tensor[]");


  // std::array list types (with empty list)
  testArgTypes<std::array<double, 0>>::test(
    // 创建一个空的 double 类型的 std::array
    std::array<double, 0>(), [] (std::array<double, 0> v) {},
    // 创建一个空的 double 类型的 std::array
    std::array<double, 0>(), [] (const IValue& v) {EXPECT_EQ(0, (v.to<c10::List<double>>().size()));},
    // 定义一个接受空 double 数组并返回空 double 数组的函数签名
    "(float[0] a) -> float[0]");
  testArgTypes<std::array<int64_t, 0>>::test(
    // 创建一个空的 int64_t 类型的 std::array
    std::array<int64_t, 0>(), [] (std::array<int64_t, 0> v) {},
    // 创建一个空的 int64_t 类型的 std::array
    std::array<int64_t, 0>(), [] (const IValue& v) {EXPECT_EQ(0, (v.to<c10::List<int64_t>>().size()));},
    // 定义一个接受空 int64 数组并返回空 int64 数组的函数签名
    "(int[0] a) -> int[0]");
  testArgTypes<std::array<bool, 0>>::test(
    // 创建一个空的 bool 类型的 std::array
    std::array<bool, 0>(), [] (std::array<bool, 0> v) {},
    // 创建一个空的 bool 类型的 std::array
    std::array<bool, 0>(), [] (const IValue& v) {EXPECT_EQ(0, (v.to<std::array<bool, 0>>().size()));},
    // 定义一个接受空 bool 数组并返回空 bool 数组的函数签名
    "(bool[0] a) -> bool[0]");
  testArgTypes<std::array<std::string, 0>>::test(
    // 创建一个空的 std::string 类型的 std::array
    std::array<std::string, 0>(), [] (std::array<std::string, 0> v) {EXPECT_EQ(0, v.size());},
    // 创建一个空的 std::string 类型的 std::array
    std::array<std::string, 0>(), [] (const IValue& v) {EXPECT_EQ(0, v.toListRef().size());},
    // 定义一个接受空字符串数组并返回空字符串数组的函数签名
    "(str[0] a) -> str[0]");


  // std::array list types (with non-empty list)
  testArgTypes<std::array<double, 2>>::test(
    // 创建一个包含两个 double 元素的 std::array
    std::array<double, 2>({1.5, 2.5}), [] (std::array<double, 2> v) {expectListEquals({1.5, 2.5}, v);},
    // 创建一个包含两个 double 元素的 std::array
    std::array<double, 2>({3.5, 4.5}), [] (const IValue& v) {expectListEquals({3.5, 4.5}, v.to<std::array<double, 2>>());},
    // 定义一个接受包含两个 double 元素的数组并返回包含两个 double 元素的数组的函数签名
    "(float[2] a) -> float[2]");
  testArgTypes<std::array<int64_t, 2>>::test(
    // 创建一个包含两个 int64_t 元素的 std::array
    std::array<int64_t, 2>({1, 2}), [] (std::array<int64_t, 2> v) {expectListEquals({1, 2}, v);},
    // 创建一个包含两个 int64_t 元素的 std::array
    std::array<int64_t, 2>({3, 4}), [] (const IValue& v) {expectListEquals({3, 4}, v.to<std::array<int64_t, 2>>());},
    // 定义一个接受包含两个 int64_t 元素的数组并返回包含两个 int64_t 元素的数组的函数签名
    "(int[2] a) -> int[2]");
  testArgTypes<std::array<bool, 2>>::test(
    // 创建一个包含两个 bool 元素的 std::array
    std::array<bool, 2>({true, false}), [] (std::array<bool, 2> v) {expectListEquals({true, false}, v);},
  // 使用测试函数 testArgTypes 测试 std::array<bool, 2> 类型的参数
  testArgTypes<std::array<bool, 2>>::test(
    // 提供 std::array<bool, 2> 类型的参数值 {true, false}
    std::array<bool, 2>({true, false}), 
    // 定义匿名函数，检查传入参数是否与期望值 {true, false} 相同
    [] (const IValue& v) {expectListEquals({true, false}, v.to<std::array<bool, 2>>());},
    // 提供期望的返回类型签名字符串 "(bool[2] a) -> bool[2]"
    "(bool[2] a) -> bool[2]");

  // 使用测试函数 testArgTypes 测试 std::array<std::string, 2> 类型的参数
  testArgTypes<std::array<std::string, 2>>::test(
    // 提供 std::array<std::string, 2> 类型的参数值 {"first", "second"}
    std::array<std::string, 2>({"first", "second"}), 
    // 定义匿名函数，检查传入参数是否与期望值 {"first", "second"} 相同
    [] (std::array<std::string, 2> v) {expectListEquals({"first", "second"}, v);},
    // 定义匿名函数，检查 IValue 类型参数是否符合预期的内容
    std::array<std::string, 2>({"first", "second"}), [] (const IValue& v) {
      // 检查列表长度是否为 2
      EXPECT_EQ(2, v.toListRef().size());
      // 检查第一个元素是否为 "first"
      EXPECT_EQ("first", v.toListRef()[0].toStringRef());
      // 检查第二个元素是否为 "second"
      EXPECT_EQ("second", v.toListRef()[1].toStringRef());
    },
    // 提供期望的返回类型签名字符串 "(str[2] a) -> str[2]"
    "(str[2] a) -> str[2]");

  // 使用测试函数 testArgTypes 测试 std::array<Tensor, 2> 类型的参数
  testArgTypes<std::array<Tensor, 2>>::test(
    // 提供 std::array<Tensor, 2> 类型的参数值，包含两个 dummyTensor 对象
    std::array<Tensor, 2>({dummyTensor(c10::DispatchKey::CPUTensorId), dummyTensor(c10::DispatchKey::CUDATensorId)}), 
    // 定义匿名函数，检查传入参数是否符合预期
    [] (std::array<Tensor, 2> v) {
      // 检查数组长度是否为 2
      EXPECT_EQ(2, v.size());
      // 检查第一个元素的 DispatchKey 是否为 CPUTensorId
      EXPECT_EQ(c10::DispatchKey::CPUTensorId, extractDispatchKey(v[0]));
      // 检查第二个元素的 DispatchKey 是否为 CUDATensorId
      EXPECT_EQ(c10::DispatchKey::CUDATensorId, extractDispatchKey(v[1]));
    },
    // 提供期望的返回类型签名字符串 "(Tensor[2] a) -> Tensor[2]"
    "(Tensor[2] a) -> Tensor[2]");

  // deprecated list types (with empty list)
  // 使用测试函数 testArgTypes 测试 std::vector<double> 类型的参数，使用 TestLegacyAPI
  testArgTypes<std::vector<double>>::test<TestLegacyAPI>(
    // 提供空的 std::vector<double> 类型参数
    std::vector<double>(), 
    // 定义匿名函数，检查传入参数是否为空列表
    [] (const std::vector<double>& v) {EXPECT_EQ(0, v.size());},
    // 提供空的 std::vector<double> 类型参数作为期望值
    std::vector<double>(), 
    // 定义匿名函数，检查 IValue 类型参数是否为空列表
    [] (const IValue& v) {EXPECT_EQ(0, v.to<c10::List<double>>().size());},
    // 提供期望的返回类型签名字符串 "(float[] a) -> float[]"
    "(float[] a) -> float[]");

  // 同上，测试 std::vector<int64_t> 类型的参数
  testArgTypes<std::vector<int64_t>>::test<TestLegacyAPI>(
    std::vector<int64_t>(), [] (const std::vector<int64_t>& v) {EXPECT_EQ(0, v.size());},
    std::vector<int64_t>(), [] (const IValue& v) {EXPECT_EQ(0, v.to<c10::List<int64_t>>().size());},
    "(int[] a) -> int[]");

  // 注意：vector<bool> 不支持，应使用 List<bool>
  // 使用测试函数 testArgTypes 测试 std::vector<std::string> 类型的参数，使用 TestLegacyAPI
  testArgTypes<std::vector<std::string>>::test<TestLegacyAPI>(
    // 提供空的 std::vector<std::string> 类型参数
    std::vector<std::string>(), 
    // 定义匿名函数，检查传入参数是否为空列表
    [] (const std::vector<std::string>& v) {EXPECT_EQ(0, v.size());},
    // 提供空的 std::vector<std::string> 类型参数作为期望值
    std::vector<std::string>(), 
    // 定义匿名函数，检查 IValue 类型参数是否为空列表
    [] (const IValue& v) {EXPECT_EQ(0, v.toListRef().size());},
    // 提供期望的返回类型签名字符串 "(str[] a) -> str[]"
    "(str[] a) -> str[]");

  // deprecated list types (with non-empty list)
  // 使用测试函数 testArgTypes 测试 std::vector<double> 类型的参数，使用 TestLegacyAPI
  testArgTypes<std::vector<double>>::test<TestLegacyAPI>(
    // 提供包含值 {1.5, 2.5} 的 std::vector<double> 类型参数
    std::vector<double>({1.5, 2.5}), 
    // 定义匿名函数，检查传入参数是否与期望值 {1.5, 2.5} 相同
    [] (const std::vector<double>& v) {expectListEquals({1.5, 2.5}, v);},
    // 提供包含值 {3.5, 4.5} 的 std::vector<double> 类型参数作为期望值
    std::vector<double>({3.5, 4.5}), 
    // 定义匿名函数，检查 IValue 类型参数是否与期望值 {3.5, 4.5} 相同
    [] (const IValue& v) {expectListEquals({3.5, 4.5}, v.to<c10::List<double>>());},
    // 提供期望的返回类型签名字符串 "(float[] a) -> float[]"
    "(float[] a) -> float[]");

  // 同上，测试 std::vector<int64_t> 类型的参数
  testArgTypes<std::vector<int64_t>>::test<TestLegacyAPI>(
    std::vector<int64_t>({1, 2}), [] (const std::vector<int64_t>& v) {expectListEquals({1, 2}, v);},
    std::vector<int64_t>({3, 4}), [] (const IValue& v) {expectListEquals({3, 4}, v.to<c10::List<int64_t>>());},
    "(int[] a) -> int[]");
    // 测试参数类型为 std::vector<int64_t>
    testArgTypes<std::vector<int64_t>>::test<TestLegacyAPI>(
      // 传入值为包含 {3, 4} 的 std::vector<int64_t>
      std::vector<int64_t>({3, 4}), 
      // 匿名函数，验证传入值是否等于 {3, 4}
      [] (const IValue& v) {expectListEquals({3, 4}, v.to<c10::List<int64_t>>());},
      // 期望的返回类型声明为 "(int[] a) -> int[]"
      "(int[] a) -> int[]");
    
    // 注意：vector<bool> 不支持，应使用 List<bool> 替代。
    // 测试参数类型为 std::vector<std::string>
    testArgTypes<std::vector<std::string>>::test<TestLegacyAPI>(
      // 传入值为包含 {"first", "second"} 的 std::vector<std::string>
      std::vector<std::string>({"first", "second"}), 
      // 匿名函数，验证传入值是否等于 {"first", "second"}
      [] (const std::vector<std::string>& v) {expectListEquals({"first", "second"}, v);},
      // 传入值为包含 {"first", "second"} 的 std::vector<std::string>
      std::vector<std::string>({"first", "second"}), 
      // 匿名函数，验证传入值的 IValue 表示是否符合预期
      [] (const IValue& v) {
        EXPECT_EQ(2, v.toListRef().size());
        EXPECT_EQ("first", v.toListRef()[0].toStringRef());
        EXPECT_EQ("second", v.toListRef()[1].toStringRef());
      },
      // 期望的返回类型声明为 "(str[] a) -> str[]"
      "(str[] a) -> str[]");
    
    // 测试参数类型为 std::vector<Tensor>
    testArgTypes<std::vector<Tensor>>::test<TestLegacyAPI>(
      // 传入值为两个 dummyTensor，分别指定 CPU 和 CUDA 的 DispatchKey
      std::vector<Tensor>({dummyTensor(c10::DispatchKey::CPU), dummyTensor(c10::DispatchKey::CUDA)}), 
      // 匿名函数，验证传入值是否包含两个 Tensor
      [] (const std::vector<Tensor>& v) {
        EXPECT_EQ(2, v.size());
        EXPECT_EQ(c10::DispatchKey::CPU, extractDispatchKey(v.at(0)));
        EXPECT_EQ(c10::DispatchKey::CUDA, extractDispatchKey(v.at(1)));
      },
      // 传入值为两个 dummyTensor，顺序相反，分别指定 CUDA 和 CPU 的 DispatchKey
      std::vector<Tensor>({dummyTensor(c10::DispatchKey::CUDA), dummyTensor(c10::DispatchKey::CPU)}), 
      // 匿名函数，验证传入值的 IValue 表示是否符合预期
      [] (const IValue& v) {
        EXPECT_EQ(2, v.to<c10::List<at::Tensor>>().size());
        EXPECT_EQ(c10::DispatchKey::CUDA, extractDispatchKey(v.to<c10::List<at::Tensor>>().get(0)));
        EXPECT_EQ(c10::DispatchKey::CPU, extractDispatchKey(v.to<c10::List<at::Tensor>>().get(1)));
      },
      // 期望的返回类型声明为 "(Tensor[] a) -> Tensor[]"
      "(Tensor[] a) -> Tensor[]");
    
    // 测试可选的列表（为空的情况）
    testArgTypes<std::optional<c10::List<int64_t>>>::test(
      // 传入值为 std::optional<c10::List<int64_t>>，值为空
      std::optional<c10::List<int64_t>>(c10::nullopt), 
      // 匿名函数，验证传入值是否为空
      [] (const std::optional<c10::List<int64_t>>& v) {EXPECT_FALSE(v.has_value());},
      // 传入值为 std::optional<c10::List<int64_t>>，值为空
      std::optional<c10::List<int64_t>>(c10::nullopt), 
      // 匿名函数，验证传入值的 IValue 表示是否为 None
      [] (const IValue& v) {EXPECT_TRUE(v.isNone());},
      // 期望的返回类型声明为 "(int[]? a) -> int[]?"
      "(int[]? a) -> int[]?");
    
    // 测试可选的列表（包含空列表）
    testArgTypes<std::optional<c10::List<int64_t>>>::test(
      // 传入值为 std::optional<c10::List<int64_t>>，包含空的 c10::List<int64_t>
      std::optional<c10::List<int64_t>>(c10::List<int64_t>({})), 
      // 匿名函数，验证传入值的 c10::List<int64_t> 是否为空
      [] (const std::optional<c10::List<int64_t>>& v) {EXPECT_EQ(0, v.value().size());},
      // 传入值为 std::optional<c10::List<int64_t>>，包含空的 c10::List<int64_t>
      std::optional<c10::List<int64_t>>(c10::List<int64_t>({})), 
      // 匿名函数，验证传入值的 IValue 表示是否为空的 c10::List<int64_t>
      [] (const IValue& v) {EXPECT_EQ(0, v.to<c10::List<int64_t>>().size());},
      // 期望的返回类型声明为 "(int[]? a) -> int[]?"
      "(int[]? a) -> int[]?");
    
    // 测试可选的列表（包含值）
    testArgTypes<std::optional<c10::List<int64_t>>>::test(
      // 传入值为 std::optional<c10::List<int64_t>>，包含 {1, 2} 的 c10::List<int64_t>
      std::optional<c10::List<int64_t>>(c10::List<int64_t>({1, 2})), 
      // 匿名函数，验证传入值是否等于 {1, 2}
      [] (const std::optional<c10::List<int64_t>>& v) {expectListEquals({1, 2}, v.value());},
      // 传入值为 std::optional<c10::List<int64_t>>，包含 {3, 4} 的 c10::List<int64_t>
      std::optional<c10::List<int64_t>>(c10::List<int64_t>({3, 4})), 
      // 匿名函数，验证传入值的 IValue 表示是否等于 {3, 4}
      [] (const IValue& v) {expectListEquals({3, 4}, v.to<c10::List<int64_t>>());},
      // 期望的返回类型声明为 "(int[]? a) -> int[]?"
      "(int[]? a) -> int[]?");
    
    // 测试列表的可选项（包含空列表）
    testArgTypes<c10::List<::std::optional<int64_t>>>::test(
      // 传入值为包含空 c10::List<::std::optional<int64_t>> 的 c10::List<::std::optional<int64_t>>
      c10::List<::std::optional<int64_t>>(c10::List<::std::optional<int64_t>>({})), 
      // 匿名函数，验证传入值是否为空
      [] (const c10::List<::std::optional<int64_t>>& v) {EXPECT_EQ(0, v.size());},
    // 测试针对空列表的可选值的列表类型
    testArgTypes<c10::List<::std::optional<int64_t>>>::test(
        // 创建一个空的 c10::List<std::optional<int64_t>>
        c10::List<::std::optional<int64_t>>(c10::List<::std::optional<int64_t>>({})),
        // 匿名函数，验证传入值的大小是否为零
        [] (const IValue& v) {EXPECT_EQ(0, v.to<c10::List<::std::optional<int64_t>>>().size());},
        "(int?[] a) -> int?[]");

    // 测试带值的可选值列表类型
    testArgTypes<c10::List<::std::optional<int64_t>>>::test(
        // 创建一个带有值的 c10::List<std::optional<int64_t>>
        c10::List<::std::optional<int64_t>>(c10::List<::std::optional<int64_t>>({3, c10::nullopt, 2})),
        // 匿名函数，验证传入值是否与期望列表相同
        [] (const c10::List<::std::optional<int64_t>>& v) {expectListEquals<std::optional<int64_t>>({3, c10::nullopt, 2}, v);},
        // 创建一个带有相同值的 c10::List<std::optional<int64_t>>
        c10::List<::std::optional<int64_t>>(c10::List<::std::optional<int64_t>>({3, c10::nullopt, 2})),
        // 匿名函数，验证转换后的值是否与期望列表相同
        [] (const IValue& v) {expectListEquals<std::optional<int64_t>>({3, c10::nullopt, 2}, v.to<c10::List<::std::optional<int64_t>>>());},
        "(int?[] a) -> int?[]");

    // 字典类型
    c10::Dict<std::string, std::string> str_dict;
    // 向字典插入键值对
    str_dict.insert("key1", "value1");
    str_dict.insert("key2", "value2");
    testArgTypes<c10::Dict<std::string, std::string>>::test(
        // 使用 str_dict 进行测试
        str_dict,
        // 匿名函数，验证传入的字典是否符合预期
        [] (c10::Dict<std::string, std::string> v) {
            EXPECT_EQ(2, v.size());
            EXPECT_EQ("value1", v.at("key1"));
            EXPECT_EQ("value2", v.at("key2"));
        },
        // 使用 str_dict 进行测试
        str_dict,
        // 匿名函数，验证转换后的字典是否符合预期
        [] (const IValue& v) {
            // 将通用字典转换为特定类型的字典
            c10::Dict<std::string, std::string> dict = c10::impl::toTypedDict<std::string, std::string>(v.toGenericDict());
            EXPECT_EQ(2, dict.size());
            EXPECT_EQ("value1", dict.at("key1"));
            EXPECT_EQ("value2", dict.at("key2"));
        },
        "(Dict(str, str) a) -> Dict(str, str)");

    // Tensor 字典类型
    c10::Dict<int64_t, Tensor> tensor_dict;
    // 向 Tensor 字典插入键值对
    tensor_dict.insert(1, dummyTensor(c10::DispatchKey::CPU));
    tensor_dict.insert(2, dummyTensor(c10::DispatchKey::CUDA));
    testArgTypes<c10::Dict<int64_t, Tensor>>::test(
        // 使用 tensor_dict 进行测试
        tensor_dict,
        // 匿名函数，验证传入的字典是否符合预期
        [] (c10::Dict<int64_t, Tensor> v) {
            EXPECT_EQ(2, v.size());
            EXPECT_EQ(c10::DispatchKey::CPU, extractDispatchKey(v.at(1)));
            EXPECT_EQ(c10::DispatchKey::CUDA, extractDispatchKey(v.at(2)));
        },
        // 使用 tensor_dict 进行测试
        tensor_dict,
        // 匿名函数，验证转换后的字典是否符合预期
        [] (const IValue& v) {
            // 将通用字典转换为特定类型的字典
            c10::Dict<int64_t, Tensor> dict = c10::impl::toTypedDict<int64_t, Tensor>(v.toGenericDict());
            EXPECT_EQ(2, dict.size());
            EXPECT_EQ(c10::DispatchKey::CPU, extractDispatchKey(dict.at(1)));
            EXPECT_EQ(c10::DispatchKey::CUDA, extractDispatchKey(dict.at(2)));
        },
        "(Dict(int, Tensor) a) -> Dict(int, Tensor)");

    // 废弃的字典类型
    std::unordered_map<std::string, std::string> str_map;
    // 向未排序的 map 中插入键值对
    str_map.emplace("key1", "value1");
    str_map.emplace("key2", "value2");
    testArgTypes<std::unordered_map<std::string, std::string>>::test<TestLegacyAPI>(
        // 使用 str_map 进行测试
        str_map,
        // 匿名函数，验证传入的 map 是否符合预期
        [] (std::unordered_map<std::string, std::string> v) {
            EXPECT_EQ(2, v.size());
            EXPECT_EQ("value1", v.at("key1"));
            EXPECT_EQ("value2", v.at("key2"));
        },
  // 定义一个匿名 Lambda 函数，参数为 const IValue& v，返回类型为 void
  str_map, [] (const IValue& v) {
    // 将 v 转换为 c10::Dict<std::string, std::string> 类型的字典
    c10::Dict<std::string, std::string> dict = c10::impl::toTypedDict<std::string, std::string>(v.toGenericDict());
    // 断言字典的大小为 2
    EXPECT_EQ(2, dict.size());
    // 断言键 "key1" 的值为 "value1"
    EXPECT_EQ("value1", dict.at("key1"));
    // 断言键 "key2" 的值为 "value2"
    EXPECT_EQ("value2", dict.at("key2"));
  },
  // 设置 Lambda 函数的类型签名为 "(Dict(str, str) a) -> Dict(str, str)"
  "(Dict(str, str) a) -> Dict(str, str)");

// 创建一个空的 unordered_map<int64_t, Tensor> 类型的变量 tensor_map
std::unordered_map<int64_t, Tensor> tensor_map;
// 向 tensor_map 中插入键值对 (1, dummyTensor(c10::DispatchKey::CPU))
tensor_map.emplace(1, dummyTensor(c10::DispatchKey::CPU));
// 向 tensor_map 中插入键值对 (2, dummyTensor(c10::DispatchKey::CUDA))
tensor_map.emplace(2, dummyTensor(c10::DispatchKey::CUDA));
// 使用 testArgTypes 对 std::unordered_map<int64_t, Tensor> 进行类型测试
testArgTypes<std::unordered_map<int64_t, Tensor>>::test<TestLegacyAPI>(
  // 传入参数 tensor_map
  tensor_map, [] (std::unordered_map<int64_t, Tensor> v) {
    // 断言 v 的大小为 2
    EXPECT_EQ(2, v.size());
    // 断言键 1 对应的 Tensor 的 DispatchKey 为 CPU
    EXPECT_EQ(c10::DispatchKey::CPU, extractDispatchKey(v.at(1)));
    // 断言键 2 对应的 Tensor 的 DispatchKey 为 CUDA
    EXPECT_EQ(c10::DispatchKey::CUDA, extractDispatchKey(v.at(2)));
  },
  // 传入参数 tensor_map
  tensor_map, [] (const IValue& v) {
    // 将 v 转换为 c10::Dict<int64_t, Tensor> 类型的字典
    c10::Dict<int64_t, Tensor> dict = c10::impl::toTypedDict<int64_t, Tensor>(v.toGenericDict());
    // 断言字典的大小为 2
    EXPECT_EQ(2, dict.size());
    // 断言键 1 对应的 Tensor 的 DispatchKey 为 CPU
    EXPECT_EQ(c10::DispatchKey::CPU, extractDispatchKey(dict.at(1)));
    // 断言键 2 对应的 Tensor 的 DispatchKey 为 CUDA
    EXPECT_EQ(c10::DispatchKey::CUDA, extractDispatchKey(dict.at(2)));
  },
  // 设置 Lambda 函数的类型签名为 "(Dict(int, Tensor) a) -> Dict(int, Tensor)"
  "(Dict(int, Tensor) a) -> Dict(int, Tensor)");

// 定义一个类型别名 DeeplyNestedType，表示一个嵌套复杂的数据结构
using DeeplyNestedType = c10::List<c10::Dict<std::string, c10::List<::std::optional<c10::Dict<int64_t, std::string>>>>>;
// 创建一个 Lambda 函数 makeDeeplyNestedObject，返回类型为 DeeplyNestedType
auto makeDeeplyNestedObject = [] () -> DeeplyNestedType {
  // 创建内部的 c10::Dict<int64_t, std::string> 类型的变量 inner3
  c10::Dict<int64_t, std::string> inner3;
  // 向 inner3 中插入键值对 (1, "1")
  inner3.insert(1, "1");
  // 创建内部的 c10::List<::std::optional<c10::Dict<int64_t, std::string>>> 类型的变量 inner2
  c10::List<::std::optional<c10::Dict<int64_t, std::string>>> inner2;
  // 将 inner3 移动到 inner2 中
  inner2.push_back(std::move(inner3));
  // 创建内部的 c10::Dict<std::string, c10::List<::std::optional<c10::Dict<int64_t, std::string>>>> 类型的变量 inner1
  c10::Dict<std::string, c10::List<::std::optional<c10::Dict<int64_t, std::string>>>> inner1;
  // 向 inner1 中插入键值对 ("key", std::move(inner2))
  inner1.insert("key", std::move(inner2));
  // 创建一个 c10::List<c10::Dict<std::string, c10::List<::std::optional<c10::Dict<int64_t, std::string>>>>> 类型的变量 result
  c10::List<c10::Dict<std::string, c10::List<::std::optional<c10::Dict<int64_t, std::string>>>>> result;
  // 将 inner1 添加到 result 中
  result.push_back(inner1);
  // 返回 result
  return result;
};
// 使用 testArgTypes 对 DeeplyNestedType 进行类型测试
testArgTypes<DeeplyNestedType>::test(
  // 调用 makeDeeplyNestedObject 函数得到的结果作为参数
  makeDeeplyNestedObject(), [] (const DeeplyNestedType& v) {EXPECT_EQ("1", v.get(0).at("key").get(0).value().at(1));},
  // 调用 makeDeeplyNestedObject 函数得到的结果作为参数
  makeDeeplyNestedObject(), [] (const IValue& v) {EXPECT_EQ("1", v.to<DeeplyNestedType>().get(0).at("key").get(0).value().at(1));},
  // 设置 Lambda 函数的类型签名为 "(Dict(str, Dict(int, str)?[])[] a) -> Dict(str, Dict(int, str)?[])[]"
  "(Dict(str, Dict(int, str)?[])[] a) -> Dict(str, Dict(int, str)?[])[]");
TEST(NewOperatorRegistrationTest, erroroutwithinvalidoverloadname) {
  // 创建名为 m 的 Torch 库对象，用于注册运算符
  auto m = MAKE_TORCH_LIBRARY(_test);
  // 预期捕获 c10::Error 异常，因为 "default" 不是合法的操作重载名称
  expectThrows<c10::Error>([&] {
   // 尝试注册一个操作，但使用了不合法的重载名称 "dummy.default(Tensor self) -> Tensor"
   m.def("dummy.default(Tensor self) -> Tensor");
  }, "default is not a legal overload name for aten operators");
  // 预期捕获 c10::Error 异常，因为 "__name__" 不是合法的操作重载名称
  expectThrows<c10::Error>([&] {
   // 尝试注册一个操作，但使用了不合法的重载名称 "dummy.__name__(Tensor self) -> Tensor"
   m.def("dummy.__name__(Tensor self) -> Tensor");
  }, "__name__ is not a legal overload name for aten operators");
}

TEST(NewOperatorRegistrationTest, testBasics) {
  // 创建名为 m 的 Torch 库对象，用于注册运算符
  auto m = MAKE_TORCH_LIBRARY(_test);
  // 注册一个名为 "dummy" 的操作，接受一个 Tensor 参数并返回 Tensor
  m.def("dummy(Tensor self) -> Tensor");
  // 注册一个名为 "dummy1" 的操作，接受一个 Tensor 参数并返回 Tensor
  m.def("dummy1(Tensor self) -> Tensor");
  // 注册一个名为 "dummy2" 的操作，接受一个 Tensor 参数并返回 Tensor
  m.def("dummy2(Tensor self) -> Tensor");
  // 注册一个名为 "dummy3" 的操作，接受两个 Tensor 参数并返回 Tensor，使用 lambda 表达式定义
  m.def("dummy3(Tensor self, Tensor other) -> Tensor", [](const Tensor& self, const Tensor& other) { return self; });
  // 注册一个名为 "dummy4" 的操作，接受两个 Tensor 参数并返回 Tensor，使用 lambda 表达式定义
  m.def("dummy4", [](const Tensor& self, const Tensor& other) { return other; });
  // 为 "dummy" 操作在 CPU 设备上注册实现，使用 lambda 表达式定义
  m.impl("dummy", c10::DeviceType::CPU, [](const Tensor& self) { return self; });
  // 为 "dummy" 操作在 XLA 设备上注册实现，使用 lambda 表达式定义
  m.impl("dummy", c10::DeviceType::XLA, [](const Tensor& self) { return self; });
  // 为 "dummy" 操作在 Lazy 设备上注册实现，使用 lambda 表达式定义
  m.impl("dummy", c10::DeviceType::Lazy, [](const Tensor& self) { return self; });
  // 内部 API：为 "dummy2" 操作在 CPU 调度键上注册实现，使用 lambda 表达式定义
  m.impl("dummy2", c10::DispatchKey::CPU, [](const Tensor& self) { return self; });
  // 内部 API：为 "dummy2" 操作在 XLA 调度键上注册实现，使用 lambda 表达式定义
  m.impl("dummy2", c10::DispatchKey::XLA, [](const Tensor& self) { return self; });
  // 内部 API：为 "dummy2" 操作在 Lazy 调度键上注册实现，使用 lambda 表达式定义
  m.impl("dummy2", c10::DispatchKey::Lazy, [](const Tensor& self) { return self; });

  // 断言：确保 Dispatcher 单例能够找到包含给定名称的操作架构
  ASSERT_TRUE(Dispatcher::singleton().findSchema({"_test::dummy", ""}).has_value());
  // 断言：即使没有实现，确保 Dispatcher 单例能够找到包含给定名称的操作架构
  ASSERT_TRUE(Dispatcher::singleton().findSchema({"_test::dummy1", ""}).has_value());
  ASSERT_TRUE(Dispatcher::singleton().findSchema({"_test::dummy2", ""}).has_value());
  ASSERT_TRUE(Dispatcher::singleton().findSchema({"_test::dummy3", ""}).has_value());
  ASSERT_TRUE(Dispatcher::singleton().findSchema({"_test::dummy4", ""}).has_value());
}

TEST(NewOperatorRegistrationTest, importTopLevel) {
  // 创建名为 m 的 Torch 库对象，用于注册运算符
  auto m = MAKE_TORCH_LIBRARY(test);
  // 注册一个名为 "def1" 的操作，接受一个 Tensor 参数并返回 Tensor
  m.def("def1(Tensor self) -> Tensor");
  // 注册一个名为 "def2" 的操作，接受一个 Tensor 参数并返回 Tensor，使用 lambda 表达式定义
  m.def("def2(Tensor self) -> Tensor", [](const Tensor& x) { return x; });
  // 注册一个名为 "def3" 的操作，接受一个 Tensor 参数并返回 Tensor，使用 lambda 表达式定义
  m.def("def3", [](const Tensor& x) { return x; });

  // 创建名为 m2 的 Torch 库对象，用于注册运算符的实现
  auto m2 = MAKE_TORCH_LIBRARY_IMPL(test, CatchAll);
  // 为 "impl1" 操作注册实现，接受一个 Tensor 参数并返回 Tensor，使用 lambda 表达式定义
  m2.impl("impl1", [](const Tensor& x) { return x; });

  // 断言：确保 Dispatcher 单例能够找到包含给定名称的操作架构
  ASSERT_TRUE(Dispatcher::singleton().findSchema({"test::def1", ""}).has_value());
  ASSERT_TRUE(Dispatcher::singleton().findSchema({"test::def2", ""}).has_value());
  ASSERT_TRUE(Dispatcher::singleton().findSchema({"test::def3", ""}).has_value());
  // 断言：确保 Dispatcher 单例能够找到包含给定名称的操作
  ASSERT_TRUE(Dispatcher::singleton().findOp({"test::def1", ""}).has_value());
  ASSERT_TRUE(Dispatcher::singleton().findOp({"test::def2", ""}).has_value());
  ASSERT_TRUE(Dispatcher::singleton().findOp({"test::def3", ""}).has_value());
  ASSERT_TRUE(Dispatcher::singleton().findOp({"test::impl1", ""}).has_value());
}
TEST(NewOperatorRegistrationTest, overload) {
  // 创建名为 'test' 的 Torch 库命名空间，并将其赋给变量 m
  auto m = MAKE_TORCH_LIBRARY(test);
  // 在 Torch 库中注册函数 'fn'，接受一个 Tensor 类型参数并返回一个 Tensor 类型结果
  m.def("fn(Tensor self) -> Tensor");
  // 在函数 'fn' 的重载 'overload1' 中注册，接受两个 Tensor 类型参数并返回一个 Tensor 类型结果
  m.def("fn.overload1(Tensor self, Tensor other) -> Tensor");
  // 在函数 'fn' 的重载 'overload2' 中注册，接受三个 Tensor 类型参数并返回一个 Tensor 类型结果
  m.def("fn.overload2(Tensor self, Tensor other, Tensor alpha) -> Tensor");

  // 断言是否能通过 Dispatcher 单例找到名为 'test::fn' 的函数模式
  ASSERT_TRUE(Dispatcher::singleton().findSchema({"test::fn", ""}).has_value());
  // 断言是否能通过 Dispatcher 单例找到名为 'test::fn' 的重载 'overload1' 的函数模式
  ASSERT_TRUE(Dispatcher::singleton().findSchema({"test::fn", "overload1"}).has_value());
  // 断言是否能通过 Dispatcher 单例找到名为 'test::fn' 的重载 'overload2' 的函数模式
  ASSERT_TRUE(Dispatcher::singleton().findSchema({"test::fn", "overload2"}).has_value());
}

TEST(NewOperatorRegistrationTest, importNamespace) {
  // 创建名为 'test' 的 Torch 库命名空间，并将其赋给变量 m
  auto m = MAKE_TORCH_LIBRARY(test);
  // 在 Torch 库中注册函数 'def1'，接受一个 Tensor 类型参数并返回一个 Tensor 类型结果
  m.def("def1(Tensor self) -> Tensor");
  // 在 Torch 库中注册函数 'def2'，接受一个 Tensor 类型参数并返回一个 Tensor 类型结果，使用 Lambda 表达式
  m.def("def2(Tensor self) -> Tensor", [](const Tensor& x) { return x; });
  // 在 Torch 库中注册函数 'def3'，不接受参数，返回一个 Tensor 类型结果，使用 Lambda 表达式
  m.def("def3", [](const Tensor& x) { return x; });
  // 在 Torch 库中注册函数 'impl1'，接受一个 Tensor 类型参数并返回一个 Tensor 类型结果，使用 Lambda 表达式
  m.impl("impl1", [](const Tensor& x) { return x; });
  // 期望抛出 c10::Error 异常，当试图在命名空间 'retest' 下注册函数 'def1' 时
  expectThrows<c10::Error>([&] {
    m.def("retest::def1(Tensor self) -> Tensor");
  }, "");

  // 断言是否能通过 Dispatcher 单例找到名为 'test::def1' 的函数模式
  ASSERT_TRUE(Dispatcher::singleton().findSchema({"test::def1", ""}).has_value());
  // 断言是否能通过 Dispatcher 单例找到名为 'test::def2' 的函数模式
  ASSERT_TRUE(Dispatcher::singleton().findSchema({"test::def2", ""}).has_value());
  // 断言是否能通过 Dispatcher 单例找到名为 'test::def3' 的函数模式
  ASSERT_TRUE(Dispatcher::singleton().findSchema({"test::def3", ""}).has_value());
  // 断言是否能通过 Dispatcher 单例找到名为 'test::impl1' 的操作模式
  ASSERT_TRUE(Dispatcher::singleton().findOp({"test::impl1", ""}).has_value());
}

TEST(NewOperatorRegistrationTest, schema) {
  // 创建名为 'test' 的 Torch 库命名空间，并将其赋给变量 m
  auto m = MAKE_TORCH_LIBRARY(test);
  // 在 Torch 库中注册函数 'def1'，接受一个 Tensor 类型参数并返回一个 Tensor 类型结果
  m.def("def1(Tensor self) -> Tensor");
  // 在 Torch 库中注册函数 'def2'，接受一个 Tensor 类型参数并返回一个 Tensor 类型结果，使用 torch::schema 创建
  m.def(torch::schema("def2(Tensor self) -> Tensor"));
  // 在 Torch 库中注册函数 'def3'，接受一个 Tensor 类型参数并返回一个 Tensor 类型结果，
  // 并指定其别名分析为 PURE_FUNCTION，使用 torch::schema 创建
  m.def(torch::schema("def3(Tensor self) -> Tensor", c10::AliasAnalysisKind::PURE_FUNCTION));
  // 在 Torch 库中注册函数 'def4'，接受一个 Tensor 类型参数并返回一个 Tensor 类型结果，
  // 使用 torch::jit::parseSchema 解析函数签名并注册
  m.def(torch::jit::parseSchema("def4(Tensor self) -> Tensor"));

  // 断言是否能通过 Dispatcher 单例找到名为 'test::def1' 的函数模式
  ASSERT_TRUE(Dispatcher::singleton().findSchema({"test::def1", ""}).has_value());
  // 断言是否能通过 Dispatcher 单例找到名为 'test::def2' 的函数模式
  ASSERT_TRUE(Dispatcher::singleton().findSchema({"test::def2", ""}).has_value());
  // 断言是否能通过 Dispatcher 单例找到名为 'test::def3' 的函数模式
  ASSERT_TRUE(Dispatcher::singleton().findSchema({"test::def3", ""}).has_value());
  // 断言是否能通过 Dispatcher 单例找到名为 'test::def4' 的函数模式
  ASSERT_TRUE(Dispatcher::singleton().findSchema({"test::def4", ""}).has_value());

  // 验证名为 'test::def1' 的函数的别名分析是否为 FROM_SCHEMA 类型
  EXPECT_EQ(Dispatcher::singleton().findSchema({"test::def1", ""})->schema().aliasAnalysis(), c10::AliasAnalysisKind::FROM_SCHEMA);
  // 验证名为 'test::def2' 的函数的别名分析是否为 FROM_SCHEMA 类型
  EXPECT_EQ(Dispatcher::singleton().findSchema({"test::def2", ""})->schema().aliasAnalysis(), c10::AliasAnalysisKind::FROM_SCHEMA);
  // 验证名为 'test::def3' 的函数的别名分析是否为 PURE_FUNCTION 类型
  EXPECT_EQ(Dispatcher::singleton().findSchema({"test::def3", ""})->schema().aliasAnalysis(), c10::AliasAnalysisKind::PURE_FUNCTION);
  // 验证名为 'test::def4' 的函数的别名分析是否为默认的别名分析类型
  ASSERT_TRUE(Dispatcher::singleton().findSchema({"test::def4", ""})->schema().isDefaultAliasAnalysisKind());
}
TEST(NewOperatorRegistrationTest, whenRegisteringBackendFallbackKernelAndCatchallKernelForSameBackend_thenCallsFallbackKernel) {
  // 创建一个新的 Torch 库实现对象 m1，使用 CPU 作为后端
  auto m1 = MAKE_TORCH_LIBRARY_IMPL(_, CPU);
  // 将后端回退内核注册到 m1 中
  m1.fallback(CppFunction::makeFromBoxedFunction<&backend_fallback_kernel>());

  // 初始化一个布尔变量 called，用于检测是否调用了函数
  bool called = false;
  // 创建一个新的 Torch 库对象 m，用于测试
  auto m = MAKE_TORCH_LIBRARY(test);
  // 定义一个名为 fn 的函数，接受一个 Tensor 和一个字符串作为输入，无返回值
  m.def("fn(Tensor t, str input) -> ()");
  // 将一个 lambda 函数作为 fn 函数的实现，用于设置 called 变量为 true
  m.impl("fn", [&] (Tensor, std::string) { called = true; });

  // 查找名为 "test::fn" 的操作模式
  auto op = Dispatcher::singleton().findSchema({"test::fn", ""});
  // 断言操作模式存在
  ASSERT_TRUE(op.has_value());

  // 重置 called 变量为 false
  called = false;
  // 调用 callOp 函数，传入 op 操作模式和一个虚拟的 Tensor 对象和字符串
  auto stack = callOp(*op, dummyTensor(c10::DispatchKey::CPU), "hello ");
  // 现在 CatchAll 映射到 CompositeImplicitAutograd，并且优先级高于后端回退
  EXPECT_TRUE(called);
}

TEST(NewOperatorRegistrationTest, whenRegisteringAutogradKernelWithRegularKernel_thenCanCallRegularKernel) {
  // 创建一个新的 Torch 库对象 m，用于测试
  auto m = MAKE_TORCH_LIBRARY(test);
  // 定义一个名为 fn 的函数，接受一个 Tensor 作为输入，无返回值
  m.def("fn(Tensor dummy) -> ()");
  // 将一个非自动求导的内核函数注册为 fn 函数的实现，使用 CPU 作为分发键
  m.impl("fn", c10::DispatchKey::CPU, nonautograd_kernel);
  // 将一个自动求导的内核函数注册为 fn 函数的实现，使用 Autograd 作为分发键
  m.impl("fn", c10::DispatchKey::Autograd, autograd_kernel);

  // 查找名为 "test::fn" 的操作模式
  auto op = Dispatcher::singleton().findSchema({"test::fn", ""});
  // 断言操作模式存在
  ASSERT_TRUE(op.has_value());

  // 初始化 called_nonautograd 和 called_autograd 变量为 false
  called_nonautograd = called_autograd = false;
  // 调用 callOp 函数，传入 op 操作模式和一个虚拟的 Tensor 对象
  callOp(*op, dummyTensor(DispatchKey::CPU));
  // 断言非自动求导内核被调用
  EXPECT_TRUE(called_nonautograd);
  // 断言自动求导内核未被调用
  EXPECT_FALSE(called_autograd);
}

TEST(NewOperatorRegistrationTest, dispatchWithCompositeImplicitAutogradKernel) {
  // 初始化 math_called 变量为 false
  bool math_called = false;
  // 创建一个新的 Torch 库对象 m，用于测试
  auto m = MAKE_TORCH_LIBRARY(test);
  // 定义一个名为 fn 的函数，使用 CompositeImplicitAutograd 作为分发键，接受一个 Tensor 作为输入
  m.def("fn", torch::dispatch(c10::DispatchKey::CompositeImplicitAutograd, [&](const Tensor& x) { math_called = true; return x; }));

  // 查找名为 "test::fn" 的操作模式
  auto op = Dispatcher::singleton().findSchema({"test::fn", ""});
  // 断言操作模式存在
  ASSERT_TRUE(op.has_value());

  {
    // 断言 math_called 变量为 false
    ASSERT_FALSE(math_called);
    // 调用 callOp 函数，传入 op 操作模式和一个虚拟的 Tensor 对象，使用 CPU 作为分发键
    callOp(*op, dummyTensor(c10::DispatchKey::CPU));
    // 断言 math_called 变量为 true
    ASSERT_TRUE(math_called);
  }

  {
    // 将 math_called 变量重置为 false
    math_called = false;
    // 调用 callOp 函数，传入 op 操作模式和一个虚拟的 Tensor 对象，使用 CPU 作为分发键，并指定需要梯度
    callOp(*op, dummyTensor(c10::DispatchKey::CPU, /*requires_grad=*/true));
    // 断言 math_called 变量为 true
    ASSERT_TRUE(math_called);
  }

  // 遍历分发键列表 {XLA, Lazy}
  for (c10::DispatchKey key : {c10::DispatchKey::XLA, c10::DispatchKey::Lazy}) {
    // 将 math_called 变量重置为 false
    math_called = false;
    // 调用 callOp 函数，传入 op 操作模式和一个虚拟的 Tensor 对象，使用当前分发键
    callOp(*op, dummyTensor(key));
    // 断言 math_called 变量为 true
    ASSERT_TRUE(math_called);
  }

  // 遍历分发键列表 {XLA, Lazy}
  for (c10::DispatchKey key : {c10::DispatchKey::XLA, c10::DispatchKey::Lazy}) {
    // 将 math_called 变量重置为 false
    math_called = false;
    // 调用 callOp 函数，传入 op 操作模式和一个虚拟的 Tensor 对象，使用当前分发键，并指定需要梯度
    callOp(*op, dummyTensor(key, /*requires_grad=*/true));
    // 断言 math_called 变量为 true
    ASSERT_TRUE(math_called);
  }

  {
    // 将 math_called 变量重置为 false
    math_called = false;
    // 调用 callOp 函数，传入 op 操作模式和一个虚拟的 Tensor 对象，使用 SparseCPU 作为分发键
    callOp(*op, dummyTensor(c10::DispatchKey::SparseCPU));
    // 断言 math_called 变量为 true
    ASSERT_TRUE(math_called);
  }

  {
    // 将 math_called 变量重置为 false
    math_called = false;
    // 调用 callOp 函数，传入 op 操作模式和一个虚拟的 Tensor 对象，使用 SparseCPU 作为分发键，并指定需要梯度
    callOp(*op, dummyTensor(c10::DispatchKey::SparseCPU, /*requires_grad=*/true));
    // 断言 math_called 变量为 true
    ASSERT_TRUE(math_called);
  }
}
TEST(NewOperatorRegistrationTest, dispatchWithCompositeImplicitAutogradAndAutogradKernel) {
  bool math_called = false;  // 定义一个布尔变量，用于标记数学运算是否被调用
  bool autograd_called = false;  // 定义一个布尔变量，用于标记自动微分运算是否被调用
  auto m = MAKE_TORCH_LIBRARY(test);  // 创建一个名为 m 的 Torch 库
  m.def("fn", torch::dispatch(c10::DispatchKey::CompositeImplicitAutograd, [&](const Tensor& x) { math_called = true; return x; }));  // 在库 m 中注册一个函数 "fn"，使用 CompositeImplicitAutograd 调度键，当调用时执行数学运算，并更新 math_called 标志
  m.impl("fn", c10::DispatchKey::Autograd, [&](const Tensor& x) { autograd_called = true; return x; });  // 在库 m 中注册 "fn" 函数的 Autograd 实现，当调用时执行自动微分运算，并更新 autograd_called 标志

  auto op = Dispatcher::singleton().findSchema({"test::fn", ""});  // 获取名称为 "test::fn" 的操作模式
  ASSERT_TRUE(op.has_value());  // 断言操作模式存在

  // CompositeImplicitAutograd 优先级高于 Autograd
  {
    math_called = autograd_called = false;  // 重置 math_called 和 autograd_called 标志
    callOp(*op, dummyTensor(c10::DispatchKey::CPU, /*requires_grad=*/true));  // 调用 op 操作模式，传入一个具有 CPU 调度键和需要梯度的虚拟张量
    ASSERT_TRUE(math_called);  // 断言数学运算被调用
    ASSERT_FALSE(autograd_called);  // 断言自动微分未被调用
  }

  {
    math_called = autograd_called = false;  // 重置 math_called 和 autograd_called 标志
    callOp(*op, dummyTensor(c10::DispatchKey::CPU));  // 再次调用 op 操作模式，传入一个具有 CPU 调度键的虚拟张量
    ASSERT_TRUE(math_called);  // 断言数学运算被调用
    ASSERT_FALSE(autograd_called);  // 断言自动微分未被调用
  }
}

TEST(NewOperatorRegistrationTest, dispatchWithCompositeImplicitAutogradAndCatchAllKernel) {
  bool math_called = false;  // 定义一个布尔变量，用于标记数学运算是否被调用
  bool catchall_called = false;  // 定义一个布尔变量，用于标记 catch-all 函数是否被调用
  auto m = MAKE_TORCH_LIBRARY(test);  // 创建一个名为 m 的 Torch 库
  m.def("fn", torch::dispatch(c10::DispatchKey::CompositeImplicitAutograd, [&](const Tensor& x) { math_called = true; return x; }));  // 在库 m 中注册一个函数 "fn"，使用 CompositeImplicitAutograd 调度键，当调用时执行数学运算，并更新 math_called 标志
  m.impl("fn", [&](const Tensor& x) { catchall_called = true; return x; });  // 在库 m 中注册 "fn" 函数的 catch-all 实现，当调用时执行并捕获所有情况，并更新 catchall_called 标志

  auto op = Dispatcher::singleton().findSchema({"test::fn", ""});  // 获取名称为 "test::fn" 的操作模式
  ASSERT_TRUE(op.has_value());  // 断言操作模式存在

  // catchAll 现在映射到 CompositeImplicitAutograd，这意味着我们有两个注册到 CompositeImplicitAutograd 键的注册。
  // 使用最后一个注册。
  {
    catchall_called = math_called = false;  // 重置 catchall_called 和 math_called 标志
    callOp(*op, dummyTensor(c10::DispatchKey::CPU));  // 调用 op 操作模式，传入一个具有 CPU 调度键的虚拟张量
    ASSERT_FALSE(math_called);  // 断言数学运算未被调用
    ASSERT_TRUE(catchall_called);  // 断言 catch-all 函数被调用
  }

  {
    catchall_called = math_called = false;  // 重置 catchall_called 和 math_called 标志
    callOp(*op, dummyTensor(c10::DispatchKey::CPU, /*requires_grad=*/true));  // 再次调用 op 操作模式，传入一个具有 CPU 调度键和需要梯度的虚拟张量
    ASSERT_FALSE(math_called);  // 断言数学运算未被调用
    ASSERT_TRUE(catchall_called);  // 断言 catch-all 函数被调用
  }
}

TEST(NewOperatorRegistrationTest, AutogradBackendOverridesCompositeImplicitAutogradKernel) {
  bool math_called = false;  // 定义一个布尔变量，用于标记数学运算是否被调用
  bool autograd_called = false;  // 定义一个布尔变量，用于标记自动微分运算是否被调用
  auto m = MAKE_TORCH_LIBRARY(test);  // 创建一个名为 m 的 Torch 库
  m.def("fn", torch::dispatch(c10::DispatchKey::CompositeImplicitAutograd, [&](const Tensor& x) { math_called = true; return x; }));  // 在库 m 中注册一个函数 "fn"，使用 CompositeImplicitAutograd 调度键，当调用时执行数学运算，并更新 math_called 标志
  m.impl("fn", c10::DispatchKey::AutogradCPU, [&](const Tensor& x) { autograd_called = true; return x; });  // 在库 m 中注册 "fn" 函数的 AutogradCPU 实现，当调用时执行自动微分运算，并更新 autograd_called 标志

  auto op = Dispatcher::singleton().findSchema({"test::fn", ""});  // 获取名称为 "test::fn" 的操作模式
  ASSERT_TRUE(op.has_value());  // 断言操作模式存在

  {
    math_called = autograd_called = false;  // 重置 math_called 和 autograd_called 标志
    callOp(*op, dummyTensor(c10::DispatchKey::CPU));  // 调用 op 操作模式，传入一个具有 CPU 调度键的虚拟张量
    ASSERT_TRUE(math_called);  // 断言数学运算被调用
    ASSERT_FALSE(autograd_called);  // 断言自动微分未被调用
  }

  {
    math_called = autograd_called = false;  // 重置 math_called 和 autograd_called 标志
    callOp(*op, dummyTensor(c10::DispatchKey::CPU, /*requires_grad=*/true));  // 再次调用 op 操作模式，传入一个具有 CPU 调度键和需要梯度的虚拟张量
    ASSERT_TRUE(autograd_called);  // 断言自动微分被调用
    ASSERT_FALSE(math_called);  // 断言数学运算未被调用
  }

  {
    math_called = autograd_called = false;  // 重置 math_called 和 autograd_called 标志
    callOp(*op, dummyTensor(c10::DispatchKey::CUDA));  // 再次调用 op 操作模式，传入一个具有 CUDA 调度键的虚拟张量
    # 断言确保 `math_called` 为真
    ASSERT_TRUE(math_called);
    # 断言确保 `autograd_called` 为假
    ASSERT_FALSE(autograd_called);
  }

  {
    # 将 `math_called` 和 `autograd_called` 初始化为假
    math_called = autograd_called = false;
    # 调用 `callOp` 函数，传入操作符 `*op` 和一个虚拟张量，指定其 CUDA 分发键并启用梯度跟踪
    callOp(*op, dummyTensor(c10::DispatchKey::CUDA, /*requires_grad=*/true));
    # 断言确保 `math_called` 为真
    ASSERT_TRUE(math_called);
    # 断言确保 `autograd_called` 为假
    ASSERT_FALSE(autograd_called);
  }
TEST(NewOperatorRegistrationTest, BackendOverridesCompositeImplicitAutogradKernel) {
  // 标志变量，用于记录是否调用了数学后端和通用后端
  bool math_called = false;
  bool backend_called = false;
  // 创建一个 Torch 库命名空间 'test'
  auto m = MAKE_TORCH_LIBRARY(test);
  // 定义函数 'fn'，并分派给复合隐式自动微分的调度键
  m.def("fn", torch::dispatch(c10::DispatchKey::CompositeImplicitAutograd, [&](const Tensor& x) { 
    math_called = true; 
    return x; 
  }));
  // 实现函数 'fn'，在 CPU 调度键上调用后端实现
  m.impl("fn", c10::DispatchKey::CPU, [&](const Tensor& x) { 
    backend_called = true; 
    return x; 
  });

  // 查找名为 "test::fn" 的操作模式
  auto op = Dispatcher::singleton().findSchema({"test::fn", ""});
  ASSERT_TRUE(op.has_value());

  {
    // 重置标志变量
    math_called = backend_called = false;
    // 调用 'fn' 操作，传递一个假的 CPU 张量
    callOp(*op, dummyTensor(c10::DispatchKey::CPU));
    // 断言后端实现已调用且数学实现未被调用
    ASSERT_TRUE(backend_called);
    ASSERT_FALSE(math_called);
  }

  {
    // 对于具有自动微分要求的 CPU 张量，自动微分 CPU 是默认路径，最终调用 CPU 核心
    math_called = backend_called = false;
    callOp(*op, dummyTensor(c10::DispatchKey::CPU, /*requires_grad=*/true));
    ASSERT_TRUE(backend_called);
    ASSERT_FALSE(math_called);
  }

  {
    // 调用 'fn' 操作，传递一个假的 CUDA 张量
    math_called = backend_called = false;
    callOp(*op, dummyTensor(c10::DispatchKey::CUDA));
    // 断言数学实现已调用且后端实现未被调用
    ASSERT_TRUE(math_called);
    ASSERT_FALSE(backend_called);
  }

  {
    // 对于具有自动微分要求的 CUDA 张量，自动微分 CUDA 是默认路径，最终调用 CUDA 核心
    math_called = backend_called = false;
    callOp(*op, dummyTensor(c10::DispatchKey::CUDA, /*requires_grad=*/true));
    ASSERT_TRUE(math_called);
    ASSERT_FALSE(backend_called);
  }
}

TEST(NewOperatorRegistrationTest, dispatchWithCompositeExplicitAutogradKernel) {
  // 标志变量，用于记录是否调用了函数 'fn'
  bool called = false;
  // 创建一个 Torch 库命名空间 'test'
  auto m = MAKE_TORCH_LIBRARY(test);
  // 定义函数 'fn'，并分派给复合显式自动微分的调度键
  m.def("fn", torch::dispatch(c10::DispatchKey::CompositeExplicitAutograd, [&](const Tensor& x) { 
    called = true; 
    return x; 
  }));

  // 查找名为 "test::fn" 的操作模式
  auto op = Dispatcher::singleton().findSchema({"test::fn", ""});
  ASSERT_TRUE(op.has_value());

  {
    // 断言函数 'fn' 尚未被调用
    ASSERT_FALSE(called);
    // 调用 'fn' 操作，传递一个假的 CPU 张量
    callOp(*op, dummyTensor(c10::DispatchKey::CPU));
    // 断言函数 'fn' 已被调用
    ASSERT_TRUE(called);
  }

  {
    // 对于具有自动微分要求的 CPU 张量，自动微分 CPU 是默认路径，最终调用 CPU 核心
    called = false;
    callOp(*op, dummyTensor(c10::DispatchKey::CPU, /*requires_grad=*/true));
    ASSERT_TRUE(called);
  }

  // 对于每个给定的调度键，调用 'fn' 操作
  for (c10::DispatchKey key : {c10::DispatchKey::XLA, c10::DispatchKey::Lazy}) {
    called = false;
    callOp(*op, dummyTensor(key));
    ASSERT_TRUE(called);
  }

  // 对于每个给定的具有自动微分要求的调度键，调用 'fn' 操作
  for (c10::DispatchKey key : {c10::DispatchKey::XLA, c10::DispatchKey::Lazy}) {
    called = false;
    // 自动微分 {XLA, Lazy} 是默认路径，最终调用 XLA / Lazy 核心
    callOp(*op, dummyTensor(key, /*requires_grad=*/true));
    ASSERT_TRUE(called);
  }

  {
    // 调用 'fn' 操作，传递一个假的 SparseCPU 张量
    called = false;
    callOp(*op, dummyTensor(c10::DispatchKey::SparseCPU));
    ASSERT_TRUE(called);
  }

  {
    // 对于具有自动微分要求的 SparseCPU 张量，自动微分 CPU 是默认路径，最终调用 CPU 核心
    called = false;
    callOp(*op, dummyTensor(c10::DispatchKey::SparseCPU, /*requires_grad=*/true));
    ASSERT_TRUE(called);
  }
}
TEST(NewOperatorRegistrationTest, dispatchWithCompositeExplicitAutogradAndCompositeImplicitAutogradKernel) {
  bool backend_called = false;  // 初始化一个布尔变量，表示后端调用的状态
  bool math_called = false;  // 初始化一个布尔变量，表示数学调用的状态
  auto m = MAKE_TORCH_LIBRARY(test);  // 创建一个 Torch 库，用于注册函数
  m.def("fn", torch::dispatch(c10::DispatchKey::CompositeExplicitAutograd, [&](const Tensor& x) { backend_called = true; return x; }));  // 定义函数 "fn"，使用 CompositeExplicitAutograd 分发策略调用后端函数
  m.impl("fn", c10::DispatchKey::CompositeImplicitAutograd, [&](const Tensor& x) { math_called = true; return x; });  // 在 "fn" 上实现 CompositeImplicitAutograd 分发策略，调用数学函数

  auto op = Dispatcher::singleton().findSchema({"test::fn", ""});  // 查找注册的操作模式
  ASSERT_TRUE(op.has_value());  // 确保找到了操作模式

  {
    backend_called = math_called = false;  // 重置后端调用和数学调用的状态
    callOp(*op, dummyTensor(c10::DispatchKey::CPU));  // 调用操作 op，并传入一个 CPU 张量
    ASSERT_TRUE(backend_called);  // 断言后端函数被调用
    ASSERT_FALSE(math_called);  // 断言数学函数未被调用
  }

  {
    backend_called = math_called = false;  // 重置后端调用和数学调用的状态
    // AutogradCPU 是向后兼容，调用 CPU 内核
    callOp(*op, dummyTensor(c10::DispatchKey::CPU, /*requires_grad=*/true));  // 调用操作 op，并传入一个需要梯度的 CPU 张量
    ASSERT_FALSE(math_called);  // 断言数学函数未被调用
    ASSERT_TRUE(backend_called);  // 断言后端函数被调用
  }

  for (c10::DispatchKey key : {c10::DispatchKey::XLA, c10::DispatchKey::Lazy}) {
    backend_called = math_called = false;  // 重置后端调用和数学调用的状态
    callOp(*op, dummyTensor(key));  // 调用操作 op，并传入指定的 key 对应的张量
    ASSERT_TRUE(backend_called);  // 断言后端函数被调用
    ASSERT_FALSE(math_called);  // 断言数学函数未被调用
  }

  for (c10::DispatchKey key : {c10::DispatchKey::XLA, c10::DispatchKey::Lazy}) {
    backend_called = math_called = false;  // 重置后端调用和数学调用的状态
    // Autograd{XLA, Lazy} 是向后兼容，调用对应的 XLA / Lazy 内核
    callOp(*op, dummyTensor(key, /*requires_grad=*/true));  // 调用操作 op，并传入需要梯度的指定 key 对应的张量
    ASSERT_FALSE(math_called);  // 断言数学函数未被调用
    ASSERT_TRUE(backend_called);  // 断言后端函数被调用
  }

  {
    backend_called = math_called = false;  // 重置后端调用和数学调用的状态
    callOp(*op, dummyTensor(c10::DispatchKey::SparseCPU));  // 调用操作 op，并传入一个 SparseCPU 张量
    ASSERT_TRUE(backend_called);  // 断言后端函数被调用
    ASSERT_FALSE(math_called);  // 断言数学函数未被调用
  }

  {
    backend_called = math_called = false;  // 重置后端调用和数学调用的状态
    // AutogradOther 是向后兼容，调用 SparseCPU 内核
    callOp(*op, dummyTensor(c10::DispatchKey::SparseCPU, /*requires_grad=*/true));  // 调用操作 op，并传入一个需要梯度的 SparseCPU 张量
    ASSERT_FALSE(math_called);  // 断言数学函数未被调用
    ASSERT_TRUE(backend_called);  // 断言后端函数被调用
  }
}

TEST(NewOperatorRegistrationTest, BackendOverridesCompositeExplicitAutogradKernel) {
  bool default_called = false;  // 初始化一个布尔变量，表示默认调用的状态
  bool backend_called = false;  // 初始化一个布尔变量，表示后端调用的状态
  auto m = MAKE_TORCH_LIBRARY(test);  // 创建一个 Torch 库，用于注册函数
  m.def("fn", torch::dispatch(c10::DispatchKey::CompositeExplicitAutograd, [&](const Tensor& x) { default_called = true; return x; }));  // 定义函数 "fn"，使用 CompositeExplicitAutograd 分发策略调用默认函数
  m.impl("fn", c10::DispatchKey::CPU, [&](const Tensor& x) { backend_called = true; return x; });  // 在 "fn" 上实现 CPU 分发策略，调用后端函数

  auto op = Dispatcher::singleton().findSchema({"test::fn", ""});  // 查找注册的操作模式
  ASSERT_TRUE(op.has_value());  // 确保找到了操作模式

  {
    default_called = backend_called = false;  // 重置默认调用和后端调用的状态
    callOp(*op, dummyTensor(c10::DispatchKey::CPU));  // 调用操作 op，并传入一个 CPU 张量
    ASSERT_TRUE(backend_called);  // 断言后端函数被调用
    ASSERT_FALSE(default_called);  // 断言默认函数未被调用
  }

  {
    default_called = backend_called = false;  // 重置默认调用和后端调用的状态
    // AutogradCPU 是向后兼容，调用 CPU 内核
    callOp(*op, dummyTensor(c10::DispatchKey::CPU, /*requires_grad=*/true));  // 调用操作 op，并传入一个需要梯度的 CPU 张量
    ASSERT_TRUE(backend_called);  // 断言后端函数被调用
    ASSERT_FALSE(default_called);  // 断言默认函数未被调用
  }

  // 未完全展示的代码部分没有包含在注释中
}
    # 断言确保 default_called 为真
    ASSERT_TRUE(default_called);
    # 断言确保 backend_called 为假
    ASSERT_FALSE(backend_called);
  }

  {
    # 重置 default_called 和 backend_called 的值为 false
    default_called = backend_called = false;
    # 调用 callOp 函数，传入参数 op 和一个模拟的张量 dummyTensor
    # dummyTensor 具有 CUDA 的 DispatchKey，且设置 requires_grad 为 true
    // AutogradCUDA is fallthrough, calls CUDA kernel
    callOp(*op, dummyTensor(c10::DispatchKey::CUDA, /*requires_grad=*/true));
    # 断言确保 default_called 为真
    ASSERT_TRUE(default_called);
    # 断言确保 backend_called 为假
    ASSERT_FALSE(backend_called);
  }
}

// 定义测试用例 NewOperatorRegistrationTest 中的 dispatch 函数
TEST(NewOperatorRegistrationTest, dispatch) {
  // 标志变量，用于记录是否调用了不同的计算设备
  bool cpu_called = false;
  bool cuda_called = false;
  bool autograd_called = false;
  // 创建一个 Torch 库命名空间 m
  auto m = MAKE_TORCH_LIBRARY(test);
  // 定义在 CPU 上执行的函数 fn_cpu
  m.def("fn_cpu", torch::dispatch(c10::DispatchKey::CPU, [&](const Tensor& x) { cpu_called = true; return x; }));
  // 定义在 CUDA 上执行的函数 fn_cuda
  m.def("fn_cuda", torch::dispatch(c10::kCUDA, [&](const Tensor& x) { cuda_called = true; return x; }));
  // 定义在自动求导模式下执行的函数 fn_autograd
  m.def("fn_autograd", torch::dispatch(c10::kAutograd, [&](const Tensor& x) { autograd_called = true; return x; }));

  {
    // 查找并验证 fn_cpu 的操作模式是否存在
    auto op = Dispatcher::singleton().findSchema({"test::fn_cpu", ""});
    ASSERT_TRUE(op.has_value());
    // 确保在调用前 cpu_called 标志为假
    ASSERT_FALSE(cpu_called);
    // 调用 fn_cpu 并验证 cpu_called 是否被设置为真
    callOp(*op, dummyTensor(c10::DispatchKey::CPU));
    ASSERT_TRUE(cpu_called);
  }

  {
    // 查找并验证 fn_cuda 的操作模式是否存在
    auto op = Dispatcher::singleton().findSchema({"test::fn_cuda", ""});
    ASSERT_TRUE(op.has_value());
    // 确保在调用前 cuda_called 标志为假
    ASSERT_FALSE(cuda_called);
    // 调用 fn_cuda 并验证 cuda_called 是否被设置为真
    callOp(*op, dummyTensor(c10::DispatchKey::CUDA));
    ASSERT_TRUE(cuda_called);
  }

  {
    // 查找并验证 fn_autograd 的操作模式是否存在
    auto op = Dispatcher::singleton().findSchema({"test::fn_autograd", ""});
    ASSERT_TRUE(op.has_value());
    // 确保在调用前 autograd_called 标志为假
    ASSERT_FALSE(autograd_called);
    // 调用 fn_autograd 并验证 autograd_called 是否被设置为真
    callOp(*op, dummyTensor(c10::DispatchKey::CPU, /*requires_grad=*/true));
    ASSERT_TRUE(autograd_called);
  }

  // 循环测试其他分发键的情况
  for (c10::DispatchKey key : {c10::DispatchKey::XLA, c10::DispatchKey::Lazy}) {
    autograd_called = false;
    // 再次查找并验证 fn_autograd 的操作模式是否存在
    auto op = Dispatcher::singleton().findSchema({"test::fn_autograd", ""});
    ASSERT_TRUE(op.has_value());
    // 调用 fn_autograd 并验证 autograd_called 是否被设置为真
    callOp(*op, dummyTensor(key, /*requires_grad=*/true));
    ASSERT_TRUE(autograd_called);
  }
}

// 定义测试用例 NewOperatorRegistrationTest 中的 dispatchAutogradPrecedence 函数
TEST(NewOperatorRegistrationTest, dispatchAutogradPrecedence) {
  // 标志变量，用于记录是否调用了 CPU 计算设备
  bool cpu_called = false;
  auto m = MAKE_TORCH_LIBRARY(test);
  // 定义在 CPU 上执行的函数 fn
  m.def("fn", torch::dispatch(c10::DispatchKey::CPU, [&](const Tensor& x) { cpu_called = true; return x; }));

  {
    // 查找并验证 fn 的操作模式是否存在
    auto op = Dispatcher::singleton().findSchema({"test::fn", ""});
    ASSERT_TRUE(op.has_value());
    // 确保在调用前 cpu_called 标志为假
    ASSERT_FALSE(cpu_called);
    // 调用 fn 并验证 cpu_called 是否被设置为真
    callOp(*op, dummyTensor(c10::DispatchKey::CPU));
    ASSERT_TRUE(cpu_called);
  }

  {
    // 在自动求导模式下，使用 CPU 内核作为默认
    cpu_called = false;
    // 再次查找 fn 的操作模式是否存在
    auto op = Dispatcher::singleton().findSchema({"test::fn", ""});
    // 调用 fn 并验证 cpu_called 是否被设置为真
    callOp(*op, dummyTensor(c10::DispatchKey::CPU, /*requires_grad=*/true));
    ASSERT_TRUE(cpu_called);
  }

  // 标志变量，用于记录是否调用了自动求导模式下的函数
  bool autograd_called = false;
  // 在自动求导模式下实现 fn 函数的逻辑
  m.impl("fn", c10::kAutograd, [&](const Tensor& x) { autograd_called = true; return x; });

  {
    // 再次查找 fn 的操作模式是否存在
    auto op = Dispatcher::singleton().findSchema({"test::fn", ""});
    // 调用 fn 并验证 autograd_called 是否被设置为真
    callOp(*op, dummyTensor(c10::DispatchKey::CPU, /*requires_grad=*/true));
    ASSERT_TRUE(autograd_called);
  }

  // 自动求导后端内核比自动求导别名具有更高的优先级
  // 标志变量，用于记录是否调用了自动求导 CPU 模式下的函数
  bool autogradcpu_called = false;
  // 在自动求导 CPU 模式下实现 fn 函数的逻辑
  m.impl("fn", c10::DispatchKey::AutogradCPU, [&](const Tensor& x) { autogradcpu_called = true; return x; });

  {
    // 再次查找 fn 的操作模式是否存在
    auto op = Dispatcher::singleton().findSchema({"test::fn", ""});
    // 调用 fn 并验证 autogradcpu_called 是否被设置为真
    callOp(*op, dummyTensor(c10::DispatchKey::CPU, /*requires_grad=*/true));
    ASSERT_TRUE(autogradcpu_called);
  }
}
TEST(NewOperatorRegistrationTest, throwsWhenRegisterToBackendMapsToAutogradOther) {
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  // 声明并初始化两个布尔变量，fpga_called 为 false，math_called 为 false
  bool fpga_called, math_called = false;
  // 创建名为 m 的 Torch 库，并注册名为 "test" 的库
  auto m = MAKE_TORCH_LIBRARY(test);
  // 定义名为 "fn" 的函数，在 FPGA 分发键上注册一个 Lambda 表达式
  m.def("fn", torch::dispatch(c10::DispatchKey::FPGA, [&](const Tensor& x) { fpga_called = true; return x; }));
  // 使用 CompositeImplicitAutograd 分发键注册名为 "fn" 的函数的 Lambda 实现
  m.impl("fn", c10::DispatchKey::CompositeImplicitAutograd, [&](const Tensor& x) { math_called = true; return x; });

  // 查找名为 "test::fn" 的操作模式，并断言其存在
  auto op = Dispatcher::singleton().findSchema({"test::fn", ""});
  ASSERT_TRUE(op.has_value());

  {
    // 调用名为 "op" 的操作，使用 FPGA 分发键和一个虚拟的张量
    callOp(*op, dummyTensor(c10::DispatchKey::FPGA));
    // 断言 fpga_called 变为 true
    ASSERT_TRUE(fpga_called);
  }

  {
    // 期望抛出 C10 错误类型的异常
    expectThrows<c10::Error>([&] {
      // 调用名为 "op" 的操作，使用 FPGA 分发键和一个要求梯度的虚拟张量
      callOp(*op, dummyTensor(c10::DispatchKey::FPGA, /*requires_grad=*/true));
    }, "test::fn has kernels registered to both CompositeImplicitAutograd and a backend mapped to AutogradOther.");
  }
}

TEST(NewOperatorRegistrationTest, dispatchMultipleTensors) {
  // 声明并初始化两个布尔变量，privateuse1_called 和 catchall_called 都为 false
  bool privateuse1_called = false;
  bool catchall_called = false;
  // 创建名为 m1 的 Torch 库，注册到 AutogradPrivateUse1 分发键上，并设置后备函数为 CppFunction::makeFallthrough()
  auto m1 = MAKE_TORCH_LIBRARY_IMPL(_, AutogradPrivateUse1);
  m1.fallback(CppFunction::makeFallthrough());

  // 创建名为 m 的 Torch 库，并注册名为 "test" 的库
  auto m = MAKE_TORCH_LIBRARY(test);
  // 定义名为 "fn" 的函数，在 PrivateUse1 分发键上注册一个 Lambda 表达式
  m.def("fn", torch::dispatch(c10::DispatchKey::PrivateUse1, [&](const Tensor& x, const Tensor& y) { privateuse1_called = true; return x; }));
  // 使用默认分发键注册名为 "fn" 的函数的 Lambda 实现
  m.impl("fn", [&](const Tensor& x, const Tensor& y) { catchall_called = true; return x; });

  {
    // 查找名为 "test::fn" 的操作模式，并断言其存在
    auto op = Dispatcher::singleton().findOp({"test::fn", ""});
    ASSERT_TRUE(op.has_value());
    // 调用名为 "op" 的操作，使用 PrivateUse1 和 CPU 分发键的虚拟张量
    callOp(*op, dummyTensor(c10::DispatchKey::PrivateUse1), dummyTensor(c10::DispatchKey::CPU));
    // 断言 privateuse1_called 变为 true
    ASSERT_TRUE(privateuse1_called);
  }

  {
    // 查找名为 "test::fn" 的操作模式，并断言其存在
    auto op = Dispatcher::singleton().findOp({"test::fn", ""});
    ASSERT_TRUE(op.has_value());
    // 断言 catchall_called 为 false
    ASSERT_FALSE(catchall_called);
    // 调用名为 "op" 的操作，使用 CPU 分发键的两个虚拟张量
    callOp(*op, dummyTensor(c10::DispatchKey::CPU), dummyTensor(c10::DispatchKey::CPU));
    // 断言 catchall_called 变为 true
    ASSERT_TRUE(catchall_called);
  }

  {
    // 查找名为 "test::fn" 的操作模式，并断言其存在
    auto op = Dispatcher::singleton().findOp({"test::fn", ""});
    ASSERT_TRUE(op.has_value());
    // 将 catchall_called 置为 false
    catchall_called = false;
    // 调用名为 "op" 的操作，使用 CPU 分发键的两个要求梯度的虚拟张量
    callOp(*op,
           dummyTensor(c10::DispatchKey::CPU, /*requires_grad=*/true),
           dummyTensor(c10::DispatchKey::CPU, /*requires_grad=*/true));
    // 断言 catchall_called 变为 true
    ASSERT_TRUE(catchall_called);
  }

  {
    // 查找名为 "test::fn" 的操作模式，并断言其存在
    auto op = Dispatcher::singleton().findOp({"test::fn", ""});
    ASSERT_TRUE(op.has_value());
    // 将 catchall_called 和 privateuse1_called 都置为 false
    catchall_called = false;
    privateuse1_called = false;
    // 调用名为 "op" 的操作，使用 PrivateUse1 和 CPU 分发键的虚拟张量
    callOp(*op,
           dummyTensor(c10::DispatchKey::PrivateUse1, /*requires_grad=*/true),
           dummyTensor(c10::DispatchKey::CPU, /*requires_grad=*/true));
    // 断言 catchall_called 为 false
    ASSERT_FALSE(catchall_called);
    // 断言 privateuse1_called 变为 true
    ASSERT_TRUE(privateuse1_called);
  }

  // 使用 AutogradPrivateUse1 分发键注册名为 "fn" 的函数的 Lambda 实现
  m.impl("fn", c10::DispatchKey::AutogradPrivateUse1, [&](const Tensor& x, const Tensor& y) { privateuse1_called = true; return x; });

  {
    // 查找名为 "test::fn" 的操作模式，并断言其存在
    auto op = Dispatcher::singleton().findOp({"test::fn", ""});


这段代码主要是测试 Torch 框架中的操作注册和分发功能，包括注册不同的分发键和处理不同张量情况下的操作调用。
    # 确保 op 包含有效值，即 op.has_value() 返回 true
    ASSERT_TRUE(op.has_value());
    
    # 将 privateuse1_called 标志设置为 false
    privateuse1_called = false;
    
    # 调用 op 所指向的函数，传入两个张量参数：
    # - 第一个张量使用私有调度键 PrivateUse1，设置 requires_grad 为 true
    # - 第二个张量使用 CPU 调度键，设置 requires_grad 为 true
    callOp(*op,
           dummyTensor(c10::DispatchKey::PrivateUse1, /*requires_grad=*/true),
           dummyTensor(c10::DispatchKey::CPU, /*requires_grad=*/true));
    
    # 确保 privateuse1_called 已经被设置为 true
    ASSERT_TRUE(privateuse1_called);
TEST(NewOperatorRegistrationTest, registerCompositeImplicitAutogradWithCPUKernel_andCallAutogradOtherKernel_callsComposite) {
  // 标记数学和 CPU 被调用的状态为 false
  bool math_called = false;
  bool cpu_called = false;
  // 创建一个 Torch 库的实例 m
  auto m = MAKE_TORCH_LIBRARY(test);
  // 定义一个名为 "fn" 的函数，接受一个 Tensor 参数并返回 Tensor
  m.def("fn(Tensor dummy) -> Tensor");
  // 给 "fn" 函数注册一个在 CPU DispatchKey 下执行的实现，设置 CPU 被调用的标志位
  m.impl("fn", c10::DispatchKey::CPU, [&](const Tensor& x) { cpu_called = true; return x; });
  // 给 "fn" 函数注册一个在 CompositeImplicitAutograd DispatchKey 下执行的实现，设置数学被调用的标志位
  m.impl("fn", c10::DispatchKey::CompositeImplicitAutograd, [&](const Tensor& x) { math_called = true; return x; });

  // 查找名为 "test::fn" 的操作的模式
  auto op = Dispatcher::singleton().findSchema({"test::fn", ""});
  // 断言操作 op 的值存在
  ASSERT_TRUE(op.has_value());

  {
    // 重置数学和 CPU 被调用的状态
    math_called = cpu_called = false;
    // Meta DispatchKey 应重新调度到 AutogradOther 后端，
    // 其中复合核心应注册。
    callOp(*op, dummyTensor(c10::DispatchKey::Meta, /*requires_grad=*/true));
    // 断言数学被调用为 true
    ASSERT_TRUE(math_called);
    // 断言 CPU 未被调用为 false
    ASSERT_FALSE(cpu_called);
  }
}

TEST(NewOperatorRegistrationTest, dispatchMultiple) {
  // 标记 CPU、CUDA 和 Autograd 被调用的状态为 false
  bool cpu_called = false;
  bool cuda_called = false;
  bool autograd_called = false;
  // 创建一个 Torch 库的实例 m
  auto m = MAKE_TORCH_LIBRARY(test);
  // 定义一个名为 "fn" 的函数，接受一个 self Tensor 参数并返回 Tensor
  m.def("fn(Tensor self) -> Tensor");
  // 给 "fn" 函数注册一个在 CPU DispatchKey 下执行的实现，设置 CPU 被调用的标志位
  m.impl("fn", c10::DispatchKey::CPU, [&](const Tensor& x) { cpu_called = true; return x; });
  // 给 "fn" 函数注册一个在 CUDA DispatchKey 下执行的实现，设置 CUDA 被调用的标志位
  m.impl("fn", c10::kCUDA, [&](const Tensor& x) { cuda_called = true; return x; });
  // 给 "fn" 函数注册一个在 Autograd DispatchKey 下执行的实现，设置 Autograd 被调用的标志位
  m.impl("fn", c10::kAutograd, [&](const Tensor& x) { autograd_called = true; return x; });

  // 查找名为 "test::fn" 的操作的模式
  auto op = Dispatcher::singleton().findSchema({"test::fn", ""});
  // 断言操作 op 的值存在
  ASSERT_TRUE(op.has_value());

  {
    // 断言 CPU 未被调用为 false
    ASSERT_FALSE(cpu_called);
    // 调用带有 CPU DispatchKey 的 dummyTensor
    callOp(*op, dummyTensor(c10::DispatchKey::CPU));
    // 断言 CPU 被调用为 true
    ASSERT_TRUE(cpu_called);

    // 断言 CUDA 未被调用为 false
    ASSERT_FALSE(cuda_called);
    // 调用带有 CUDA DispatchKey 的 dummyTensor
    callOp(*op, dummyTensor(c10::DispatchKey::CUDA));
    // 断言 CUDA 被调用为 true
    ASSERT_TRUE(cuda_called);
  }

  {
    // 断言 Autograd 未被调用为 false
    ASSERT_FALSE(autograd_called);
    // 调用带有 CPU DispatchKey 和 requires_grad=true 的 dummyTensor
    callOp(*op, dummyTensor(c10::DispatchKey::CPU, /*requires_grad=*/true));
    // 断言 Autograd 被调用为 true
    ASSERT_TRUE(autograd_called);

    // 重置 Autograd 被调用的状态
    autograd_called = false;
    // 调用带有 CUDA DispatchKey 和 requires_grad=true 的 dummyTensor
    callOp(*op, dummyTensor(c10::DispatchKey::CUDA, /*requires_grad=*/true));
    // 断言 Autograd 被调用为 true
    ASSERT_TRUE(autograd_called);
  }
}

TEST(NewOperatorRegistrationTest, fallback) {
  // 创建一个 Torch 库的实例 m，指定后端为 CPU
  auto m = MAKE_TORCH_LIBRARY_IMPL(_, CPU);
  // 注册一个后备函数为 "_test::dummy"
  m.fallback(CppFunction::makeFromBoxedFunction<&backend_fallback_kernel>());

  // 注册操作为 "_test::dummy(Tensor dummy, str input) -> ()"
  auto registrar1 = c10::RegisterOperators().op("_test::dummy(Tensor dummy, str input) -> ()");

  // 查找名为 "_test::dummy" 的操作的模式
  auto op = Dispatcher::singleton().findSchema({"_test::dummy", ""});
  // 断言操作 op 的值存在
  ASSERT_TRUE(op.has_value());
  // 调用带有 CPU DispatchKey 的 dummyTensor，并传入字符串 "hello "
  auto stack = callOp(*op, dummyTensor(c10::DispatchKey::CPU), "hello ");
  // 断言 stack 的第二个元素的字符串值为 "hello _test::dummy"
  EXPECT_EQ("hello _test::dummy", stack[1].toStringRef());
}
TEST(NewOperatorRegistrationTest, BackendSelectRedispatchesToCPU) {
  // 初始化标记，用于检测是否调用了 CPU 相关的函数
  bool cpu_called = false;
  // 初始化标记，用于检测是否调用了后端选择的通用函数
  bool backend_generic_called = false;
  // 创建名为 m 的 Torch 库
  auto m = MAKE_TORCH_LIBRARY(test);
  // 定义一个接受 Tensor 类型参数并返回 Tensor 类型结果的函数 "fn"
  m.def("fn(Tensor self) -> Tensor");
  // 在 CPU 后端实现函数 "fn"
  m.impl("fn", c10::kCPU, [&](const Tensor& x) { cpu_called = true; return x; });
  // 在后端选择处实现函数 "fn"
  m.impl("fn", c10::DispatchKey::BackendSelect, [&](c10::DispatchKeySet ks, const Tensor& x) {
     backend_generic_called = true;
     // 查找函数 "test::fn" 的操作模式
     auto op = c10::Dispatcher::singleton().findSchema({"test::fn", ""}).value().typed<Tensor (const Tensor&)>();
     // 使用后端选择后的调度器重新分发操作
     return c10::Dispatcher::singleton().redispatch<Tensor, const Tensor&>(op, ks & after_backend_select, x);
   });

  // 查找操作模式 "test::fn" 并断言其存在
  auto op = Dispatcher::singleton().findSchema({"test::fn", ""});
  ASSERT_TRUE(op.has_value());
  // 调用操作 "op"，传入一个虚拟的 Tensor 类型参数并断言调用了 CPU 后端实现
  callOp(*op, dummyTensor(c10::DispatchKey::CPU));
  ASSERT_TRUE(cpu_called);
  ASSERT_TRUE(backend_generic_called);
}

TEST(NewOperatorRegistrationTest, TorchLibraryTwiceIsError) {
  {
    // 创建 Torch 库 "test"，期望抛出 c10::Error 异常，因为只能注册单个 TORCH_LIBRARY
    auto m = MAKE_TORCH_LIBRARY(test);
    expectThrows<c10::Error>([] {
      // 再次创建 Torch 库 "test"，预期会抛出异常
      auto m2 = MAKE_TORCH_LIBRARY(test);
    }, "Only a single TORCH_LIBRARY");
  }
  // 确保在注销后能够正常创建
  auto m = MAKE_TORCH_LIBRARY(test);
}

Tensor dummy_fn(const Tensor& x) {
  return x;
}

TEST(NewOperatorRegistrationTest, CppFunction) {
  // 展示注册函数的多种可能方式
  auto m = MAKE_TORCH_LIBRARY(test);
  // 注册函数 "dummy_fn" 到函数名 "fn1"
  m.def("fn1", &dummy_fn);
  // C++ 会隐式地将函数转换为函数指针
  // 参考：https://en.cppreference.com/w/cpp/language/implicit_conversion#Function_to_pointer
  m.def("fn2", dummy_fn);
  // 使用 Lambda 表达式注册函数 "fn3"
  m.def("fn3", [](const Tensor& x) { return x; });
  // 这些需要显式的模式
  m.def("fn4(Tensor x) -> Tensor", CppFunction::makeFallthrough());
  m.def("fn5(Tensor x) -> Tensor", CppFunction::makeFromUnboxedFunction(dummy_fn));
  m.def("fn6(Tensor x) -> Tensor", CppFunction::makeFromBoxedFunction<&backend_fallback_kernel>());
}

// 一些必须从 C++ 执行的内部测试

struct OpRegistrationListenerForDelayedListenerTest : public c10::OpRegistrationListener {
  int64_t num_registers_ = 0;
  int64_t num_deregisters_ = 0;
  // 当操作被注册时调用
  void onOperatorRegistered(const OperatorHandle& op) override {
    num_registers_++;
  }
  // 当操作被注销时调用
  void onOperatorDeregistered(const OperatorHandle& op) override {
    num_deregisters_++;
  }
};
TEST(NewOperatorRegistrationTest, testDelayedListener) {
  // 创建一个延迟监听器测试用例的操作注册监听器
  auto listener = std::make_unique<OpRegistrationListenerForDelayedListenerTest>();
  // 获取监听器的原始指针
  auto listener_ptr = listener.get();
  // 向调度器注册监听器，并获取注册表
  auto registry = Dispatcher::singleton().addRegistrationListener(std::move(listener));
  // 获取监听器初始时注册的数量
  int64_t initial_num_registers = listener_ptr->num_registers_;
  // 获取监听器初始时注销的数量
  int64_t initial_num_deregisters = listener_ptr->num_deregisters_;
  // 查找名为 "_test::dummy" 的操作，断言其不存在
  auto op = Dispatcher::singleton().findOp({"_test::dummy", ""});
  ASSERT_FALSE(op.has_value());
  // 创建名为 "_test" 的 Torch 库，并注册名为 "dummy" 的操作
  auto m1 = MAKE_TORCH_LIBRARY_IMPL(_test, CPU);
  m1.impl("dummy", [](const Tensor& self) { return self; });
  // 断言注册数量未变化
  EXPECT_EQ(initial_num_registers, listener_ptr->num_registers_);
  {
    // 创建名为 "_test" 的 Torch 库，并定义 "dummy" 操作
    auto m2 = MAKE_TORCH_LIBRARY(_test);
    m2.def("dummy(Tensor self) -> Tensor");
    // 断言注册数量增加了一个
    EXPECT_EQ(initial_num_registers + 1, listener_ptr->num_registers_);
  }
  // 断言注销数量增加了一个
  EXPECT_EQ(initial_num_deregisters + 1, listener_ptr->num_deregisters_);
}

TEST(NewOperatorRegistrationTest, testImplNoDefGetsCaught) {
  // 查找未定义的实现
  auto danglingImpls = Dispatcher::singleton().findDanglingImpls();
  // 错误消息字符串，用于报告未明确指定模式的已注册操作
  std::string error_str = "Discovered operators that have been registered through the dispatcher"
                          " without explicitly specifying their schemas. Please do so using"
                          " the TORCH_LIBRARY macro. Suspect operators:\n";
  for (auto& op : danglingImpls) {
      auto& op_name = op.operator_name();
      // 添加悬挂操作的名称到错误消息中
      error_str += "\t" + op_name.name;
      if (op_name.overload_name != "") {
          error_str += "." + op_name.overload_name;
      }
      error_str += "\n";
  }
  // 断言没有发现未定义的实现
  ASSERT_EQ(danglingImpls.size(), 0) << error_str;
}

bool called_kernel_cpu = false;
bool called_kernel_autograd = false;
bool called_kernel_tracing = false;

void cpu_kernel(Tensor) {
  // 标记 CPU 内核被调用
  called_kernel_cpu = true;
}

// autograd kernel that redispatches. Explicitly takes in and updates the DispatchKeySet
void autograd_kernel_redispatching_with_DispatchKeySet(c10::DispatchKeySet ks, Tensor a) {
  // 标记自动微分内核被调用
  called_kernel_autograd = true;
  // 查找名为 "test::fn" 的模式
  auto op = Dispatcher::singleton().findSchema({"test::fn", ""});
  // 更新调度键集合
  auto updatedDispatchKeySet = ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::AutogradOther);
  // 使用预计算的调度键集合调用未装箱的操作
  callOpUnboxedWithPrecomputedDispatchKeySet<void, Tensor>(*op, updatedDispatchKeySet, a);
}

// autograd kernel that redispatches. Does not take in a DispatchKeySet
void autograd_kernel_redispatching_without_DispatchKeySet(c10::DispatchKeySet ks, Tensor a) {
  // 标记自动微分内核被调用
  called_kernel_autograd = true;
  // 查找名为 "test::fn" 的模式
  auto op = Dispatcher::singleton().findSchema({"test::fn", ""});
  // 更新调度键集合
  auto updatedDispatchKeySet = ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::AutogradOther);
  // 使用预计算的调度键集合调用未装箱的操作
  callOpUnboxedWithPrecomputedDispatchKeySet<void, Tensor>(*op, updatedDispatchKeySet, a);
}

// tracing kernel that redispatches. Explicitly takes in and updates the DispatchKeySet
// 标记内核追踪已被调用
void tracing_kernel_redispatching_with_DispatchKeySet(c10::DispatchKeySet ks, Tensor a) {
  called_kernel_tracing = true; // 将标志变量设置为 true，表示内核追踪已被调用
  // 查找名称为 {"test::fn", ""} 的模式并返回操作符
  auto op = Dispatcher::singleton().findSchema({"test::fn", ""});
  // 更新 DispatchKeySet，仅保留 Tracer DispatchKey
  auto updatedDispatchKeySet = ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, c10::DispatchKey::Tracer);
  // 调用 op 操作符，使用预计算的 DispatchKeySet，传入张量 a，返回 void
  callOpUnboxedWithPrecomputedDispatchKeySet<void, Tensor>(*op, updatedDispatchKeySet, a);
}

TEST(OperatorRegistrationTest, callKernelsWithDispatchKeySetConvention_call_redispatchesToLowerPriorityKernels) {
  // 创建名为 "test" 的 Torch 库
  auto m = MAKE_TORCH_LIBRARY(test);
  // 定义函数 "fn"，接受一个名为 dummy 的张量，返回空
  m.def("fn(Tensor dummy) -> ()");
  // 实现函数 "fn" 的 CPU 版本，使用 cpu_kernel
  m.impl("fn", c10::DispatchKey::CPU, cpu_kernel);
  // 实现函数 "fn" 的 AutogradCPU 版本，使用 autograd_kernel_redispatching_with_DispatchKeySet
  m.impl("fn", c10::DispatchKey::AutogradCPU, autograd_kernel_redispatching_with_DispatchKeySet);
  // 实现函数 "fn" 的 Tracer 版本，使用 tracing_kernel_redispatching_with_DispatchKeySet
  m.impl("fn", c10::DispatchKey::Tracer, tracing_kernel_redispatching_with_DispatchKeySet);

  // 查找名称为 {"test::fn", ""} 的模式并返回操作符
  auto op = Dispatcher::singleton().findSchema({"test::fn", ""});
  // 断言操作符 op 的值存在
  ASSERT_TRUE(op.has_value());

  // 将标志变量初始化为 false，表示三种内核都未被调用
  called_kernel_cpu = called_kernel_autograd = called_kernel_tracing = false;
  // 创建包含 Tracer、AutogradCPU、CPU DispatchKey 的 DispatchKeySet
  auto tracing_autograd_cpu_set = c10::DispatchKeySet()
                                    .add(c10::DispatchKey::Tracer)
                                    .add(c10::DispatchKey::AutogradCPU)
                                    .add(c10::DispatchKey::CPU);

  // 调用 op 操作符，传入使用 tracing_autograd_cpu_set 生成的 dummyTensor，预计追踪 -> 自动求导 -> CPU
  callOpUnboxed<void, Tensor>(*op, dummyTensor(tracing_autograd_cpu_set, true));
  // 断言内核追踪已被调用
  EXPECT_TRUE(called_kernel_tracing);
  // 断言自动求导内核已被调用
  EXPECT_TRUE(called_kernel_autograd);
  // 断言 CPU 内核已被调用
  EXPECT_TRUE(called_kernel_cpu);
}

TEST(OperatorRegistrationTest, callKernelsWithDispatchKeySetConvention_callBoxed_redispatchesToLowerPriorityKernels) {
  // 创建名为 "test" 的 Torch 库
  auto m = MAKE_TORCH_LIBRARY(test);
  // 定义函数 "fn"，接受一个名为 dummy 的张量，返回空
  m.def("fn(Tensor dummy) -> ()");
  // 实现函数 "fn" 的 CPU 版本，使用 cpu_kernel
  m.impl("fn", c10::DispatchKey::CPU, cpu_kernel);
  // 实现函数 "fn" 的 AutogradCPU 版本，使用 autograd_kernel_redispatching_with_DispatchKeySet
  m.impl("fn", c10::DispatchKey::AutogradCPU, autograd_kernel_redispatching_with_DispatchKeySet);
  // 实现函数 "fn" 的 Tracer 版本，使用 tracing_kernel_redispatching_with_DispatchKeySet
  m.impl("fn", c10::DispatchKey::Tracer, tracing_kernel_redispatching_with_DispatchKeySet);

  // 查找名称为 {"test::fn", ""} 的模式并返回操作符
  auto op = Dispatcher::singleton().findSchema({"test::fn", ""});
  // 断言操作符 op 的值存在
  ASSERT_TRUE(op.has_value());

  // 将标志变量初始化为 false，表示三种内核都未被调用
  called_kernel_cpu = called_kernel_autograd = called_kernel_tracing = false;
  // 创建包含 Tracer、AutogradCPU、CPU DispatchKey 的 DispatchKeySet
  auto tracing_autograd_cpu_set = c10::DispatchKeySet()
                                    .add(c10::DispatchKey::Tracer)
                                    .add(c10::DispatchKey::AutogradCPU)
                                    .add(c10::DispatchKey::CPU);

  // 调用 op 操作符，传入使用 tracing_autograd_cpu_set 生成的 dummyTensor，追踪 -> 自动求导 -> CPU
  callOp<Tensor>(*op, dummyTensor(tracing_autograd_cpu_set, true));
  // 断言内核追踪已被调用
  EXPECT_TRUE(called_kernel_tracing);
  // 断言自动求导内核已被调用
  EXPECT_TRUE(called_kernel_autograd);
  // 断言 CPU 内核已被调用
  EXPECT_TRUE(called_kernel_cpu);
}
TEST(OperatorRegistrationTest, callKernelsWithDispatchKeySetConvention_mixedCallingConventions_redispatchesToLowerPriorityKernels) {
  auto m = MAKE_TORCH_LIBRARY(test);
  // 在 test 库中注册一个名为 "fn" 的函数，接受一个 Tensor 类型的参数，返回空
  m.def("fn(Tensor dummy) -> ()");
  // 将 CPU DispatchKey 关联到对应的 kernel 函数
  m.impl("fn", c10::DispatchKey::CPU, cpu_kernel);
  // 将 AutogradCPU DispatchKey 关联到对应的 kernel 函数
  // dispatcher 应正确处理将其 DispatchKeySet 传递给追踪（tracing）而不是自动微分（autograd）。
  m.impl("fn", c10::DispatchKey::AutogradCPU, autograd_kernel_redispatching_without_DispatchKeySet);
  // 将 Tracer DispatchKey 关联到对应的 kernel 函数
  m.impl("fn", c10::DispatchKey::Tracer, tracing_kernel_redispatching_with_DispatchKeySet);

  // 查找名称为 "test::fn" 的操作的架构，并断言其存在
  auto op = Dispatcher::singleton().findSchema({"test::fn", ""});
  ASSERT_TRUE(op.has_value());

  // 初始化标志为 false
  called_kernel_cpu = called_kernel_autograd = called_kernel_tracing = false;
  // 创建 DispatchKeySet，包含 Tracer、AutogradCPU 和 CPU DispatchKey
  auto tracing_autograd_cpu_set = c10::DispatchKeySet()
                                    .add(c10::DispatchKey::Tracer)
                                    .add(c10::DispatchKey::AutogradCPU)
                                    .add(c10::DispatchKey::CPU);

  // 调用操作，传入包含指定 DispatchKeySet 的 dummyTensor
  callOpUnboxed<void, Tensor>(*op, dummyTensor(tracing_autograd_cpu_set, true));
  // 断言 kernel 被调用
  EXPECT_TRUE(called_kernel_tracing);
  EXPECT_TRUE(called_kernel_autograd);
  EXPECT_TRUE(called_kernel_cpu);
}

TEST(OperatorRegistrationTest, getRegistrationsForDispatchKey) {
  // 获取所有注册的操作
  auto all_ops = Dispatcher::singleton().getRegistrationsForDispatchKey(c10::nullopt);
  // 获取所有注册了 CPU DispatchKey 的操作
  auto cpu_ops = Dispatcher::singleton().getRegistrationsForDispatchKey(c10::DispatchKey::CPU);
  ASSERT_TRUE(all_ops.size() > 0);
  ASSERT_TRUE(cpu_ops.size() > 0);

  // 比较 lambda 函数，用于排序操作名称
  auto cmp_lambda = [](const c10::OperatorName a, const c10::OperatorName& b) -> bool {
      return c10::toString(a) < c10::toString(b);
  };

  // 对所有操作和 CPU 操作按操作名称排序
  std::sort(all_ops.begin(), all_ops.end(), cmp_lambda);
  std::sort(cpu_ops.begin(), cpu_ops.end(), cmp_lambda);
  // 断言 CPU 操作是所有操作的子集
  ASSERT_TRUE(std::includes(all_ops.begin(), all_ops.end(), cpu_ops.begin(), cpu_ops.end(), cmp_lambda));
}

// 定义一个操作 symint_op，接受一个 Tensor 和一个 int64_t 类型的参数，返回一个 Tensor
Tensor symint_op(const Tensor& self, int64_t length) {
  return self.clone();
}

TEST(OperatorRegistrationTest, TestSymNonSymCompatibility) {
  auto m = MAKE_TORCH_LIBRARY(_test);
  // 在 _test 库中注册一个名为 "_test::symint_op" 的函数，接受一个 Tensor 和 SymInt 类型的参数，返回 Tensor
  m.def("_test::symint_op(Tensor self, SymInt length) -> Tensor");
  auto m_cpu = MAKE_TORCH_LIBRARY_IMPL(_test, CPU);
  // 将 CPU DispatchKey 关联到 symint_op 函数
  m_cpu.impl("symint_op", c10::DispatchKey::CPU, TORCH_FN(symint_op));

  // 查找 "_test::symint_op" 的操作架构，并获取其操作句柄
  auto opHandle = c10::Dispatcher::singleton().findSchemaOrThrow(
      "_test::symint_op", "");

  // 调用带有 DispatchKey::CPU 的符号化 Tensor 的操作
  opHandle.typed<Tensor(const Tensor&, int64_t)>().call(dummyTensor(c10::DispatchKey::CPU), 4);
  // 调用带有 SymInt DispatchKey 的符号化 Tensor 的操作
  opHandle.typed<Tensor(const Tensor&, c10::SymInt)>().call(dummyTensor(c10::DispatchKey::CPU), c10::SymInt(4));

  // 断言尝试使用错误签名调用操作会抛出异常
  expectThrows<c10::Error>([&] {
    opHandle.typed<Tensor(const Tensor&, const c10::SymInt&)>().call(dummyTensor(c10::DispatchKey::CPU), c10::SymInt(4));
  }, "Tried to access or call an operator with a wrong signature");
}
}

Tensor symint_op2(const Tensor& self, c10::SymInt length) {
  return self.clone();
}

TEST(OperatorRegistrationTest, TestSymSymCompatibility) {
  // 创建名为 _test 的 Torch 库
  auto m = MAKE_TORCH_LIBRARY(_test);
  // 定义 _test::symint_op 函数，接受 Tensor self 和 SymInt length，返回 Tensor
  m.def("_test::symint_op(Tensor self, SymInt length) -> Tensor");
  // 在 CPU 上创建名为 _test 的 Torch 库的实现
  auto m_cpu = MAKE_TORCH_LIBRARY_IMPL(_test, CPU);
  // 将 symint_op 函数的 CPU 实现注册为 symint_op2 函数
  m_cpu.impl("symint_op", c10::DispatchKey::CPU, TORCH_FN(symint_op2));

  // 查找 _test::symint_op 操作的架构句柄
  auto opHandle = c10::Dispatcher::singleton().findSchemaOrThrow(
      "_test::symint_op", "");

  // 调用 _test::symint_op 的双参数模板，使用 dummyTensor 创建一个 CPU 分发键的 Tensor，长度为 4
  opHandle.typed<Tensor(const Tensor&, int64_t)>().call(dummyTensor(c10::DispatchKey::CPU), 4);
  // 调用 _test::symint_op 的双参数模板，使用 dummyTensor 创建一个 CPU 分发键的 Tensor，长度为 SymInt(4)
  opHandle.typed<Tensor(const Tensor&, c10::SymInt)>().call(dummyTensor(c10::DispatchKey::CPU), c10::SymInt(4));
  // TODO: We should reject this on principle, but today it accidentally works
  // due to going through the boxed calling convention.
  //
  // First, we attempt to test if const SymInt& has SymInt. It does not,
  // because we only accept something as SymInt if it has exactly SymInt in
  // its signature. So we check if there is a non-symint kernel. But there is
  // no non-SymInt kernel, because we only registered a real SymInt kernel.
  // When this occurs, we fall back to the boxed calling convention.  And the
  // boxed calling convention can deal with const SymInt& fine, as during
  // boxing it will just create a SymInt to push onto the argument stack and
  // everything is fine.
  // TODO: 原则上我们应该拒绝这个，但是今天它出乎意料地有效，因为它通过箱式调用约定。
  //
  // 首先，我们尝试测试 const SymInt& 是否有 SymInt。没有，因为我们只接受在其签名中完全具有 SymInt 的东西作为 SymInt。所以我们检查是否有非 SymInt 内核。但是没有非 SymInt 内核，因为我们只注册了一个真正的 SymInt 内核。
  // 当发生这种情况时，我们退回到箱式调用约定。箱式调用约定可以很好地处理 const SymInt&，因为在打包过程中它将只创建一个 SymInt 来推送到参数堆栈上，一切正常。
  opHandle.typed<Tensor(const Tensor&, const c10::SymInt&)>().call(dummyTensor(c10::DispatchKey::CPU), c10::SymInt(4));
}

Tensor symint_op3(const Tensor& self, const c10::SymInt& length) {
  return self.clone();
}

TEST(OperatorRegistrationTest, TestSymSymRefCompatibility) {
  // 创建名为 _test 的 Torch 库
  auto m = MAKE_TORCH_LIBRARY(_test);
  // 定义 _test::symint_op 函数，接受 Tensor self 和 SymInt length，返回 Tensor
  m.def("_test::symint_op(Tensor self, SymInt length) -> Tensor");
  // 在 CPU 上创建名为 _test 的 Torch 库的实现
  auto m_cpu = MAKE_TORCH_LIBRARY_IMPL(_test, CPU);

  // 预期引发 c10::Error 异常，因为 symint_op3 的签名不匹配预期的函数模式
  expectThrows<c10::Error>([&] {
    m_cpu.impl("symint_op", c10::DispatchKey::CPU, TORCH_FN(symint_op3));
  }, "doesn't match the expected function schema");
}

}

#pragma GCC diagnostic pop
```