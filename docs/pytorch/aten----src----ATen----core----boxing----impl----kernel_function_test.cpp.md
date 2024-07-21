# `.\pytorch\aten\src\ATen\core\boxing\impl\kernel_function_test.cpp`

```py
#include <gtest/gtest.h>
#include <ATen/core/boxing/impl/test_helpers.h>

#include <ATen/core/op_registration/op_registration.h>
#include <ATen/core/Tensor.h>
#include <torch/csrc/jit/frontend/function_schema_parser.h>
#include <torch/library.h>

#include <ATen/core/LegacyTypeDispatch.h>

using c10::RegisterOperators;
using c10::DispatchKey;
using c10::Stack;
using std::make_unique;
using c10::intrusive_ptr;
using c10::Dict;
using at::Tensor;
using std::string;
using std::unique_ptr;

namespace {

// 定义一个总是返回错误的内核函数，用于测试
int64_t errorKernel(const Tensor& tensor, int64_t input) {
  EXPECT_TRUE(false); // this kernel should never be called
  return 0;
}

// 增加输入值的内核函数
int64_t incrementKernel(const Tensor& tensor, int64_t input) {
  return input + 1;
}

// 减少输入值的内核函数
int64_t decrementKernel(const Tensor& tensor, int64_t input) {
  return input - 1;
}

// 预期调用增加操作的函数
void expectCallsIncrement(DispatchKey dispatch_key) {
  at::AutoDispatchBelowAutograd mode;

  // 查找操作的函数模式，并确保 CPU 内核存在
  auto op = c10::Dispatcher::singleton().findSchema({"_test::my_op", ""});
  ASSERT_TRUE(op.has_value());
  // 调用操作并验证结果
  auto result = callOp(*op, dummyTensor(dispatch_key), 5);
  EXPECT_EQ(1, result.size());  // 确保返回结果集的大小为 1
  EXPECT_EQ(6, result[0].toInt());  // 确保返回结果为预期的增加后的值
}

// 预期调用减少操作的函数
void expectCallsDecrement(DispatchKey dispatch_key) {
  at::AutoDispatchBelowAutograd mode;

  // 查找操作的函数模式，并确保 CPU 内核存在
  auto op = c10::Dispatcher::singleton().findSchema({"_test::my_op", ""});
  ASSERT_TRUE(op.has_value());
  // 调用操作并验证结果
  auto result = callOp(*op, dummyTensor(dispatch_key), 5);
  EXPECT_EQ(1, result.size());  // 确保返回结果集的大小为 1
  EXPECT_EQ(4, result[0].toInt());  // 确保返回结果为预期的减少后的值
}

// 测试用例：当基于函数的内核注册时，确保能够成功调用
TEST(OperatorRegistrationTestFunctionBasedKernel, givenKernel_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators().op("_test::my_op(Tensor dummy, int input) -> int", RegisterOperators::options().kernel<decltype(incrementKernel), &incrementKernel>(DispatchKey::CPU));
  expectCallsIncrement(DispatchKey::CPU);
}

// 测试用例：当基于 Torch 库和 Torch 函数注册时，确保能够成功调用
TEST(OperatorRegistrationTestFunctionBasedKernel, givenKernel_whenRegisteredWithTorchLibraryAndTorchFn_thenCanBeCalled) {
  auto m = MAKE_TORCH_LIBRARY(_test);
  m.def("my_op(Tensor dummy, int input) -> int");
  m.impl("my_op", DispatchKey::CPU, TORCH_FN(incrementKernel));
  expectCallsIncrement(DispatchKey::CPU);
}

// 测试用例：当基于 Torch 库和 Torch 函数注册（使用通用内核）时，确保能够成功调用
TEST(OperatorRegistrationTestFunctionBasedKernel, givenCatchAllKernel_whenRegisteredWithTorchLibraryAndTorchFn_thenCanBeCalled) {
  auto m = MAKE_TORCH_LIBRARY(_test);
  m.def("my_op(Tensor dummy, int input) -> int", TORCH_FN(incrementKernel));
  expectCallsIncrement(DispatchKey::CPU);
}

} // namespace
TEST(OperatorRegistrationTestFunctionBasedKernel, givenMultipleOperatorsAndKernels_whenRegisteredInOneRegistrar_thenCallsRightKernel) {
  // 创建操作符注册器 registrar，注册两个操作符 "_test::my_op" 和 "_test::error"
  auto registrar = RegisterOperators()
      // 注册 "_test::my_op" 操作符，指定 CPU 和 CUDA 的内核函数
      .op("_test::my_op(Tensor dummy, int input) -> int", RegisterOperators::options()
          .kernel<decltype(incrementKernel), &incrementKernel>(DispatchKey::CPU)
          .kernel<decltype(errorKernel), &errorKernel>(DispatchKey::CUDA))
      // 注册 "_test::error" 操作符，指定 CPU 和 CUDA 的内核函数
      .op("_test::error(Tensor dummy, int input) -> int", RegisterOperators::options()
          .kernel<decltype(errorKernel), &errorKernel>(DispatchKey::CPU)
          .kernel<decltype(errorKernel), &errorKernel>(DispatchKey::CUDA));
  // 验证调用 DispatchKey::CPU 时是否调用了 incrementKernel
  expectCallsIncrement(DispatchKey::CPU);
}

TEST(OperatorRegistrationTestFunctionBasedKernel, givenMultipleOperatorsAndKernels_whenRegisteredInMultipleRegistrars_thenCallsRightKernel) {
  // 创建第一个操作符注册器 registrar1，注册 "_test::my_op" 操作符，指定 CPU 和 CUDA 的内核函数
  auto registrar1 = RegisterOperators()
      .op("_test::my_op(Tensor dummy, int input) -> int", RegisterOperators::options()
          .kernel<decltype(incrementKernel), &incrementKernel>(DispatchKey::CPU)
          .kernel<decltype(errorKernel), &errorKernel>(DispatchKey::CUDA));
  // 创建第二个操作符注册器 registrar2，注册 "_test::error" 操作符，指定 CPU 和 CUDA 的内核函数
  auto registrar2 = RegisterOperators()
      .op("_test::error(Tensor dummy, int input) -> int", RegisterOperators::options()
          .kernel<decltype(errorKernel), &errorKernel>(DispatchKey::CPU)
          .kernel<decltype(errorKernel), &errorKernel>(DispatchKey::CUDA));
  // 验证调用 DispatchKey::CPU 时是否调用了 incrementKernel
  expectCallsIncrement(DispatchKey::CPU);
}

TEST(NewOperatorRegistrationTestFunctionBasedKernel, givenKernel_whenRegistrationRunsOutOfScope_thenCannotBeCalledAnymore) {
  {
    // 创建 Torch 库 _test 的注册对象 m
    auto m = MAKE_TORCH_LIBRARY(_test);
    // 定义 "_test::my_op" 的操作
    m.def("_test::my_op(Tensor dummy, int input) -> int");
    // 在 CPU 环境下注册 "_test::my_op" 的实现函数 incrementKernel
    auto m_cpu = MAKE_TORCH_LIBRARY_IMPL(_test, CPU);
    m_cpu.impl("my_op", DispatchKey::CPU, TORCH_FN(incrementKernel));
    {
      // 在 CUDA 环境下注册 "_test::my_op" 的实现函数 decrementKernel
      auto m_cuda = MAKE_TORCH_LIBRARY_IMPL(_test, CUDA);
      m_cuda.impl("my_op", DispatchKey::CUDA, TORCH_FN(decrementKernel));

      // 断言：验证在 DispatchKey::CPU 调用时调用了 incrementKernel，在 DispatchKey::CUDA 调用时调用了 decrementKernel
      expectCallsIncrement(DispatchKey::CPU);
      expectCallsDecrement(DispatchKey::CUDA);
    }

    // 现在 m_cuda 超出作用域。断言：验证在 DispatchKey::CPU 调用时仍调用了 incrementKernel，但在 DispatchKey::CUDA 调用时未找到 kernel
    expectCallsIncrement(DispatchKey::CPU);
    expectDoesntFindKernel("_test::my_op", DispatchKey::CUDA);
  }

  // 现在两个注册器都超出作用域。断言：验证整个 schema 已经不存在
  expectDoesntFindOperator("_test::my_op");
}

bool was_called = false;

void kernelWithoutOutput(const Tensor&) {
  was_called = true;
}
TEST(OperatorRegistrationTestFunctionBasedKernel, givenKernelWithoutOutput_whenRegistered_thenCanBeCalled) {
  // 注册一个没有输出的内核函数，接受一个名为dummy的张量作为参数
  auto registrar = RegisterOperators().op("_test::no_return(Tensor dummy) -> ()", RegisterOperators::options().kernel<decltype(kernelWithoutOutput), &kernelWithoutOutput>(DispatchKey::CPU));

  // 查找并验证已注册的操作模式
  auto op = c10::Dispatcher::singleton().findSchema({"_test::no_return", ""});
  ASSERT_TRUE(op.has_value());

  // 初始化调用标志
  was_called = false;
  // 调用注册的操作，并传入一个虚拟的CPU张量
  auto result = callOp(*op, dummyTensor(DispatchKey::CPU));
  EXPECT_TRUE(was_called);
  EXPECT_EQ(0, result.size());
}

std::tuple<> kernelWithZeroOutputs(const Tensor&) {
  // 标记此内核已被调用
  was_called = true;
  return std::make_tuple();  // 返回空元组
}

TEST(OperatorRegistrationTestFunctionBasedKernel, givenKernelWithZeroOutputs_whenRegistered_thenCanBeCalled) {
  // 注册一个无输出的内核函数，接受一个名为dummy的张量作为参数
  auto registrar = RegisterOperators().op("_test::zero_outputs(Tensor dummy) -> ()", RegisterOperators::options().kernel<decltype(kernelWithZeroOutputs), &kernelWithZeroOutputs>(DispatchKey::CPU));

  // 查找并验证已注册的操作模式
  auto op = c10::Dispatcher::singleton().findSchema({"_test::zero_outputs", ""});
  ASSERT_TRUE(op.has_value());

  // 初始化调用标志
  was_called = false;
  // 调用注册的操作，并传入一个虚拟的CPU张量
  auto result = callOp(*op, dummyTensor(DispatchKey::CPU));
  EXPECT_TRUE(was_called);
  EXPECT_EQ(0, result.size());
}

int64_t kernelWithIntOutput(Tensor, int64_t a, int64_t b) {
  // 返回两个整数的和作为输出
  return a + b;
}

TEST(OperatorRegistrationTestFunctionBasedKernel, givenKernelWithIntOutput_whenRegistered_thenCanBeCalled) {
  // 注册一个输出为整数的内核函数，接受一个名为dummy的张量和两个整数作为参数
  auto registrar = RegisterOperators()
      .op("_test::int_output(Tensor dummy, int a, int b) -> int", RegisterOperators::options().kernel<decltype(kernelWithIntOutput), &kernelWithIntOutput>(DispatchKey::CPU));

  // 查找并验证已注册的操作模式
  auto op = c10::Dispatcher::singleton().findSchema({"_test::int_output", ""});
  ASSERT_TRUE(op.has_value());

  // 调用注册的操作，并传入一个虚拟的CPU张量以及两个整数参数
  auto result = callOp(*op, dummyTensor(DispatchKey::CPU), 3, 6);
  EXPECT_EQ(1, result.size());
  EXPECT_EQ(9, result[0].toInt());
}

Tensor kernelWithTensorOutput(const Tensor& input) {
  // 返回与输入张量相同的张量作为输出
  return input;
}

TEST(OperatorRegistrationTestFunctionBasedKernel, givenKernelWithTensorOutput_whenRegistered_thenCanBeCalled) {
  // 注册一个输出为张量的内核函数，接受一个名为input的张量作为参数，并且可以在CPU和CUDA上执行
  auto registrar = RegisterOperators()
      .op("_test::returning_tensor(Tensor input) -> Tensor", RegisterOperators::options().kernel<decltype(kernelWithTensorOutput), &kernelWithTensorOutput>(DispatchKey::CPU)
                                                                                         .kernel<decltype(kernelWithTensorOutput), &kernelWithTensorOutput>(DispatchKey::CUDA));

  // 查找并验证已注册的操作模式
  auto op = c10::Dispatcher::singleton().findSchema({"_test::returning_tensor", ""});
  ASSERT_TRUE(op.has_value());

  // 调用注册的操作，并传入一个虚拟的CPU张量
  auto result = callOp(*op, dummyTensor(DispatchKey::CPU));
  EXPECT_EQ(1, result.size());
  EXPECT_EQ(DispatchKey::CPU, extractDispatchKey(result[0].toTensor()));

  // 再次调用注册的操作，并传入一个虚拟的CUDA张量
  result = callOp(*op, dummyTensor(DispatchKey::CUDA));
  EXPECT_EQ(1, result.size());
  EXPECT_EQ(DispatchKey::CUDA, extractDispatchKey(result[0].toTensor()));
}
// 定义一个函数，接收三个张量作为输入，并返回一个包含这三个张量的张量列表
c10::List<Tensor> kernelWithTensorListOutput(const Tensor& input1, const Tensor& input2, const Tensor& input3) {
  // 使用输入张量构造一个张量列表并返回
  return c10::List<Tensor>({input1, input2, input3});
}

// 测试函数，验证具有张量列表输出的核函数在注册后可以被调用
TEST(OperatorRegistrationTestFunctionBasedKernel, givenKernelWithTensorListOutput_whenRegistered_thenCanBeCalled) {
    // 注册一个名为 "_test::list_output" 的操作符，其输入为三个张量，输出为张量列表
    auto registrar = RegisterOperators()
      .op("_test::list_output(Tensor input1, Tensor input2, Tensor input3) -> Tensor[]", RegisterOperators::options().kernel<decltype(kernelWithTensorListOutput), &kernelWithTensorListOutput>(DispatchKey::CUDA));

  // 查找刚注册的操作符
  auto op = c10::Dispatcher::singleton().findSchema({"_test::list_output", ""});
  ASSERT_TRUE(op.has_value());

  // 调用操作符，并检查返回的结果
  auto result = callOp(*op, dummyTensor(DispatchKey::CPU), dummyTensor(DispatchKey::CUDA), dummyTensor(DispatchKey::CPU));
  EXPECT_EQ(1, result.size());
  EXPECT_EQ(3, result[0].toTensorVector().size());
  EXPECT_EQ(DispatchKey::CPU, extractDispatchKey(result[0].toTensorVector()[0]));
  EXPECT_EQ(DispatchKey::CUDA, extractDispatchKey(result[0].toTensorVector()[1]));
  EXPECT_EQ(DispatchKey::CPU, extractDispatchKey(result[0].toTensorVector()[2]));
}

// 定义一个函数，接收一个张量和三个整数作为输入，并返回一个包含这三个整数的整数列表
c10::List<int64_t> kernelWithIntListOutput(const Tensor&, int64_t input1, int64_t input2, int64_t input3) {
  // 使用输入整数构造一个整数列表并返回
  return c10::List<int64_t>({input1, input2, input3});
}

// 测试函数，验证具有整数列表输出的核函数在注册后可以被调用
TEST(OperatorRegistrationTestFunctionBasedKernel, givenKernelWithIntListOutput_whenRegistered_thenCanBeCalled) {
  // 注册一个名为 "_test::list_output" 的操作符，其输入为一个张量和三个整数，输出为整数列表
  auto registrar = RegisterOperators()
      .op("_test::list_output(Tensor dummy, int input1, int input2, int input3) -> int[]", RegisterOperators::options().kernel<decltype(kernelWithIntListOutput), &kernelWithIntListOutput>(DispatchKey::CPU));

  // 查找刚注册的操作符
  auto op = c10::Dispatcher::singleton().findSchema({"_test::list_output", ""});
  ASSERT_TRUE(op.has_value());

  // 调用操作符，并检查返回的结果
  auto result = callOp(*op, dummyTensor(DispatchKey::CPU), 2, 4, 6);
  EXPECT_EQ(1, result.size());
  EXPECT_EQ(3, result[0].toIntVector().size());
  EXPECT_EQ(2, result[0].toIntVector()[0]);
  EXPECT_EQ(4, result[0].toIntVector()[1]);
  EXPECT_EQ(6, result[0].toIntVector()[2]);
}

// 定义一个函数，接收一个张量作为输入，并返回一个包含多个不同类型输出的元组
std::tuple<Tensor, int64_t, c10::List<Tensor>, std::optional<int64_t>, Dict<string, Tensor>> kernelWithMultipleOutputs(Tensor) {
  // 创建一个字典，包含两个键值对，分别对应 "first" 和 "second"，值为具有不同分派键的虚拟张量
  Dict<string, Tensor> dict;
  dict.insert("first", dummyTensor(DispatchKey::CPU));
  dict.insert("second", dummyTensor(DispatchKey::CUDA));
  
  // 返回一个包含不同类型数据的元组
  return std::tuple<Tensor, int64_t, c10::List<Tensor>, std::optional<int64_t>, Dict<string, Tensor>>(
    dummyTensor(DispatchKey::CUDA),  // 返回一个虚拟张量，分派键为 CUDA
    5,                               // 返回一个整数 5
    c10::List<Tensor>({dummyTensor(DispatchKey::CPU), dummyTensor(DispatchKey::CUDA)}),  // 返回一个包含两个虚拟张量的张量列表
    std::optional<int64_t>(std::in_place, 0),  // 返回一个包含值为 0 的可选整数
    dict  // 返回一个包含两个键值对的字典，每个值为虚拟张量
  );
}
TEST(OperatorRegistrationTestFunctionBasedKernel, givenKernelWithMultipleOutputs_whenRegistered_thenCanBeCalled) {
  // 注册自定义操作符 "_test::multiple_outputs"，指定使用 kernelWithMultipleOutputs 函数处理，限定在 CPU 上执行
  auto registrar = RegisterOperators()
     .op("_test::multiple_outputs(Tensor dummy) -> (Tensor, int, Tensor[], int?, Dict(str, Tensor))", RegisterOperators::options().kernel<decltype(kernelWithMultipleOutputs), &kernelWithMultipleOutputs>(DispatchKey::CPU));

  // 查找已注册的操作符 "_test::multiple_outputs"
  auto op = c10::Dispatcher::singleton().findSchema({"_test::multiple_outputs", ""});
  // 断言确保找到该操作符
  ASSERT_TRUE(op.has_value());

  // 调用操作符，传入一个 CPU 上的虚拟张量，并获取结果
  auto result = callOp(*op, dummyTensor(DispatchKey::CPU));
  // 断言结果的大小为 5
  EXPECT_EQ(5, result.size());
  // 断言第一个结果张量的分发键为 CUDA
  EXPECT_EQ(DispatchKey::CUDA, extractDispatchKey(result[0].toTensor()));
  // 断言第二个结果为整数 5
  EXPECT_EQ(5, result[1].toInt());
  // 断言第三个结果为张量向量，其大小为 2
  EXPECT_EQ(2, result[2].toTensorVector().size());
  // 断言第三个结果张量向量中第一个张量的分发键为 CPU
  EXPECT_EQ(DispatchKey::CPU, extractDispatchKey(result[2].toTensorVector()[0]));
  // 断言第三个结果张量向量中第二个张量的分发键为 CUDA
  EXPECT_EQ(DispatchKey::CUDA, extractDispatchKey(result[2].toTensorVector()[1]));
  // 断言第四个结果为整数 0
  EXPECT_EQ(0, result[3].toInt());
  // 将第五个结果转换为具名张量字典，并断言其大小为 2
  auto result_dict = c10::impl::toTypedDict<string, Tensor>(result[4].toGenericDict());
  EXPECT_EQ(2, result_dict.size());
  // 断言字典中名为 "first" 的张量的分发键为 CPU
  EXPECT_EQ(DispatchKey::CPU, extractDispatchKey(result_dict.at("first")));
  // 断言字典中名为 "second" 的张量的分发键为 CUDA
  EXPECT_EQ(DispatchKey::CUDA, extractDispatchKey(result_dict.at("second")));
}

Tensor kernelWithTensorInputByReferenceWithOutput(const Tensor& input1) {
  // 直接返回输入张量的引用
  return input1;
}

Tensor kernelWithTensorInputByValueWithOutput(Tensor input1) {
  // 直接返回输入张量的副本
  return input1;
}

TEST(OperatorRegistrationTestFunctionBasedKernel, givenKernelWithTensorInputByReference_withOutput_whenRegistered_thenCanBeCalled) {
  // 注册自定义操作符 "_test::tensor_input"，指定在 CPU 和 CUDA 上使用 kernelWithTensorInputByReferenceWithOutput 函数处理
  auto registrar = RegisterOperators()
      .op("_test::tensor_input(Tensor input) -> Tensor", RegisterOperators::options().kernel<decltype(kernelWithTensorInputByReferenceWithOutput), &kernelWithTensorInputByReferenceWithOutput>(DispatchKey::CPU)
                                                                                     .kernel<decltype(kernelWithTensorInputByReferenceWithOutput), &kernelWithTensorInputByReferenceWithOutput>(DispatchKey::CUDA));

  // 查找已注册的操作符 "_test::tensor_input"
  auto op = c10::Dispatcher::singleton().findSchema({"_test::tensor_input", ""});
  // 断言确保找到该操作符
  ASSERT_TRUE(op.has_value());

  // 调用操作符，传入一个 CPU 上的虚拟张量，并获取结果
  auto result = callOp(*op, dummyTensor(DispatchKey::CPU));
  // 断言结果的大小为 1
  EXPECT_EQ(1, result.size());
  // 断言结果张量的分发键为 CPU
  EXPECT_EQ(DispatchKey::CPU, extractDispatchKey(result[0].toTensor()));

  // 再次调用操作符，传入一个 CUDA 上的虚拟张量，并获取结果
  result = callOp(*op, dummyTensor(DispatchKey::CUDA));
  // 断言结果的大小为 1
  EXPECT_EQ(1, result.size());
  // 断言结果张量的分发键为 CUDA
  EXPECT_EQ(DispatchKey::CUDA, extractDispatchKey(result[0].toTensor()));
}
TEST(OperatorRegistrationTestFunctionBasedKernel, givenKernelWithTensorInputByValue_withOutput_whenRegistered_thenCanBeCalled) {
  // 创建操作注册器，注册具有张量输入和输出的函数内核
  auto registrar = RegisterOperators()
      .op("_test::tensor_input(Tensor input) -> Tensor", RegisterOperators::options()
          .kernel<decltype(kernelWithTensorInputByValueWithOutput), &kernelWithTensorInputByValueWithOutput>(DispatchKey::CPU)
          .kernel<decltype(kernelWithTensorInputByValueWithOutput), &kernelWithTensorInputByValueWithOutput>(DispatchKey::CUDA));

  // 查找并验证操作的架构
  auto op = c10::Dispatcher::singleton().findSchema({"_test::tensor_input", ""});
  ASSERT_TRUE(op.has_value());

  // 调用操作并检查结果
  auto result = callOp(*op, dummyTensor(DispatchKey::CPU));
  EXPECT_EQ(1, result.size());
  EXPECT_EQ(DispatchKey::CPU, extractDispatchKey(result[0].toTensor()));

  // 使用不同的调度键再次调用操作并检查结果
  result = callOp(*op, dummyTensor(DispatchKey::CUDA));
  EXPECT_EQ(1, result.size());
  EXPECT_EQ(DispatchKey::CUDA, extractDispatchKey(result[0].toTensor()));
}

Tensor captured_input;

void kernelWithTensorInputByReferenceWithoutOutput(const Tensor& input1) {
  // 将输入张量通过引用赋值给全局变量 captured_input
  captured_input = input1;
}

void kernelWithTensorInputByValueWithoutOutput(Tensor input1) {
  // 将输入张量通过值赋值给全局变量 captured_input
  captured_input = input1;
}

TEST(OperatorRegistrationTestFunctionBasedKernel, givenKernelWithTensorInputByReference_withoutOutput_whenRegistered_thenCanBeCalled) {
  // 创建操作注册器，注册具有通过引用传递张量输入但无输出的函数内核
  auto registrar = RegisterOperators()
      .op("_test::tensor_input(Tensor input) -> ()", RegisterOperators::options()
          .kernel<decltype(kernelWithTensorInputByReferenceWithoutOutput), &kernelWithTensorInputByReferenceWithoutOutput>(DispatchKey::CPU)
          .kernel<decltype(kernelWithTensorInputByReferenceWithoutOutput), &kernelWithTensorInputByReferenceWithoutOutput>(DispatchKey::CUDA));

  // 查找并验证操作的架构
  auto op = c10::Dispatcher::singleton().findSchema({"_test::tensor_input", ""});
  ASSERT_TRUE(op.has_value());

  // 调用操作并检查输出结果为空
  auto outputs = callOp(*op, dummyTensor(DispatchKey::CPU));
  EXPECT_EQ(0, outputs.size());
  EXPECT_EQ(DispatchKey::CPU, extractDispatchKey(captured_input));

  // 使用不同的调度键再次调用操作并检查输出结果为空
  outputs = callOp(*op, dummyTensor(DispatchKey::CUDA));
  EXPECT_EQ(0, outputs.size());
  EXPECT_EQ(DispatchKey::CUDA, extractDispatchKey(captured_input));
}
// 定义一个测试函数 OperatorRegistrationTestFunctionBasedKernel，验证注册的基于函数的操作符核心功能
TEST(OperatorRegistrationTestFunctionBasedKernel, givenKernelWithTensorInputByValue_withoutOutput_whenRegistered_thenCanBeCalled) {
  // 创建操作符注册器 registrar
  auto registrar = RegisterOperators()
      // 注册名为 "_test::tensor_input" 的操作符，接受 Tensor 输入，无输出
      .op("_test::tensor_input(Tensor input) -> ()", RegisterOperators::options()
          // 指定 CPU 和 CUDA 上的核心函数为 kernelWithTensorInputByValueWithoutOutput
          .kernel<decltype(kernelWithTensorInputByValueWithoutOutput), &kernelWithTensorInputByValueWithoutOutput>(DispatchKey::CPU)
          .kernel<decltype(kernelWithTensorInputByValueWithoutOutput), &kernelWithTensorInputByValueWithoutOutput>(DispatchKey::CUDA));

  // 查找已注册的操作符 "_test::tensor_input" 的模式
  auto op = c10::Dispatcher::singleton().findSchema({"_test::tensor_input", ""});
  // 断言操作符模式存在
  ASSERT_TRUE(op.has_value());

  // 调用操作符 op，并传入一个虚拟的 CPU 张量，获取输出结果
  auto outputs = callOp(*op, dummyTensor(DispatchKey::CPU));
  // 验证输出结果为空
  EXPECT_EQ(0, outputs.size());
  // 验证捕获到的输入分发键为 CPU
  EXPECT_EQ(DispatchKey::CPU, extractDispatchKey(captured_input));

  // 再次调用操作符 op，传入一个虚拟的 CUDA 张量，获取输出结果
  outputs = callOp(*op, dummyTensor(DispatchKey::CUDA));
  // 验证输出结果为空
  EXPECT_EQ(0, outputs.size());
  // 验证捕获到的输入分发键为 CUDA
  EXPECT_EQ(DispatchKey::CUDA, extractDispatchKey(captured_input));
}

// 定义一个全局变量 captured_int_input，用于捕获整数输入
int64_t captured_int_input = 0;

// 定义一个核心函数 kernelWithIntInputWithoutOutput，接受 Tensor 和 int64_t 输入，无输出
void kernelWithIntInputWithoutOutput(Tensor, int64_t input1) {
  // 将 input1 的值赋给 captured_int_input
  captured_int_input = input1;
}

// 测试函数 OperatorRegistrationTestFunctionBasedKernel，验证整数输入的核心函数注册和调用
TEST(OperatorRegistrationTestFunctionBasedKernel, givenKernelWithIntInput_withoutOutput_whenRegistered_thenCanBeCalled) {
  // 创建操作符注册器 registrar
  auto registrar = RegisterOperators()
      // 注册名为 "_test::int_input" 的操作符，接受 Tensor 和 int 输入，无输出
      .op("_test::int_input(Tensor dummy, int input) -> ()", RegisterOperators::options()
          // 指定 CPU 上的核心函数为 kernelWithIntInputWithoutOutput
          .kernel<decltype(kernelWithIntInputWithoutOutput), &kernelWithIntInputWithoutOutput>(DispatchKey::CPU));

  // 查找已注册的操作符 "_test::int_input" 的模式
  auto op = c10::Dispatcher::singleton().findSchema({"_test::int_input", ""});
  // 断言操作符模式存在
  ASSERT_TRUE(op.has_value());

  // 将 captured_int_input 初始化为 0
  captured_int_input = 0;
  // 调用操作符 op，传入一个虚拟的 CPU 张量和整数 3，获取输出结果
  auto outputs = callOp(*op, dummyTensor(DispatchKey::CPU), 3);
  // 验证输出结果为空
  EXPECT_EQ(0, outputs.size());
  // 验证捕获到的整数输入为 3
  EXPECT_EQ(3, captured_int_input);
}

// 定义一个核心函数 kernelWithIntInputWithOutput，接受 Tensor 和 int64_t 输入，并返回 int64_t 输出
int64_t kernelWithIntInputWithOutput(Tensor, int64_t input1) {
  // 返回 input1 加上 1 的值
  return input1 + 1;
}

// 测试函数 OperatorRegistrationTestFunctionBasedKernel，验证整数输入和输出的核心函数注册和调用
TEST(OperatorRegistrationTestFunctionBasedKernel, givenKernelWithIntInput_withOutput_whenRegistered_thenCanBeCalled) {
  // 创建操作符注册器 registrar
  auto registrar = RegisterOperators()
      // 注册名为 "_test::int_input" 的操作符，接受 Tensor 和 int 输入，并返回 int 输出
      .op("_test::int_input(Tensor dummy, int input) -> int", RegisterOperators::options()
          // 指定 CPU 上的核心函数为 kernelWithIntInputWithOutput
          .kernel<decltype(kernelWithIntInputWithOutput), &kernelWithIntInputWithOutput>(DispatchKey::CPU));

  // 查找已注册的操作符 "_test::int_input" 的模式
  auto op = c10::Dispatcher::singleton().findSchema({"_test::int_input", ""});
  // 断言操作符模式存在
  ASSERT_TRUE(op.has_value());

  // 调用操作符 op，传入一个虚拟的 CPU 张量和整数 3，获取输出结果
  auto outputs = callOp(*op, dummyTensor(DispatchKey::CPU), 3);
  // 验证输出结果的大小为 1
  EXPECT_EQ(1, outputs.size());
  // 验证输出结果的第一个元素为 4
  EXPECT_EQ(4, outputs[0].toInt());
}

// 定义一个全局变量 captured_input_list_size，用于捕获整数列表的大小
int64_t captured_input_list_size = 0;

// 定义一个核心函数 kernelWithIntListInputWithoutOutput，接受 Tensor 和整数列表输入，无输出
void kernelWithIntListInputWithoutOutput(Tensor, const c10::List<int64_t>& input1) {
  // 将 input1 的大小赋给 captured_input_list_size
  captured_input_list_size = input1.size();
}
TEST(OperatorRegistrationTestFunctionBasedKernel, givenKernelWithIntListInput_withoutOutput_whenRegistered_thenCanBeCalled) {
  // 创建一个操作注册器，并注册一个没有输出的具有整数列表输入的核函数
  auto registrar = RegisterOperators()
      .op("_test::int_list_input(Tensor dummy, int[] input) -> ()", RegisterOperators::options().kernel<decltype(kernelWithIntListInputWithoutOutput), &kernelWithIntListInputWithoutOutput>(DispatchKey::CPU));

  // 查找已注册的操作模式
  auto op = c10::Dispatcher::singleton().findSchema({"_test::int_list_input", ""});
  // 确保找到相应的操作模式
  ASSERT_TRUE(op.has_value());

  // 设置捕获的输入列表大小为零
  captured_input_list_size = 0;
  // 调用注册的操作，并传入一个整数列表作为输入
  auto outputs = callOp(*op, dummyTensor(DispatchKey::CPU), c10::List<int64_t>({2, 4, 6}));
  // 断言输出的大小为零
  EXPECT_EQ(0, outputs.size());
  // 断言捕获的输入列表大小为3
  EXPECT_EQ(3, captured_input_list_size);
}

int64_t kernelWithIntListInputWithOutput(Tensor, const c10::List<int64_t>& input1) {
  // 返回输入列表的大小作为输出
  return input1.size();
}

TEST(OperatorRegistrationTestFunctionBasedKernel, givenKernelWithIntListInput_withOutput_whenRegistered_thenCanBeCalled) {
  // 创建一个操作注册器，并注册一个具有输出的具有整数列表输入的核函数
  auto registrar = RegisterOperators()
      .op("_test::int_list_input(Tensor dummy, int[] input) -> int", RegisterOperators::options().kernel<decltype(kernelWithIntListInputWithOutput), &kernelWithIntListInputWithOutput>(DispatchKey::CPU));

  // 查找已注册的操作模式
  auto op = c10::Dispatcher::singleton().findSchema({"_test::int_list_input", ""});
  // 确保找到相应的操作模式
  ASSERT_TRUE(op.has_value());

  // 调用注册的操作，并传入一个整数列表作为输入
  auto outputs = callOp(*op, dummyTensor(DispatchKey::CPU), c10::List<int64_t>({2, 4, 6}));
  // 断言输出的大小为1
  EXPECT_EQ(1, outputs.size());
  // 断言输出的第一个元素为输入列表的大小，即3
  EXPECT_EQ(3, outputs[0].toInt());
}

void kernelWithTensorListInputWithoutOutput(const c10::List<Tensor>& input1) {
  // 设置捕获的输入列表大小为输入列表的大小
  captured_input_list_size = input1.size();
}

TEST(OperatorRegistrationTestFunctionBasedKernel, givenKernelWithTensorListInput_withoutOutput_whenRegistered_thenCanBeCalled) {
  // 创建一个操作注册器，并注册一个没有输出的具有张量列表输入的核函数
  auto registrar = RegisterOperators()
      .op("_test::tensor_list_input(Tensor[] input) -> ()", RegisterOperators::options().kernel<decltype(kernelWithTensorListInputWithoutOutput), &kernelWithTensorListInputWithoutOutput>(DispatchKey::CPU));

  // 查找已注册的操作模式
  auto op = c10::Dispatcher::singleton().findSchema({"_test::tensor_list_input", ""});
  // 确保找到相应的操作模式
  ASSERT_TRUE(op.has_value());

  // 设置捕获的输入列表大小为零
  captured_input_list_size = 0;
  // 调用注册的操作，并传入一个张量列表作为输入
  auto outputs = callOp(*op, c10::List<Tensor>({dummyTensor(DispatchKey::CPU), dummyTensor(DispatchKey::CPU)}));
  // 断言输出的大小为零
  EXPECT_EQ(0, outputs.size());
  // 断言捕获的输入列表大小为2
  EXPECT_EQ(2, captured_input_list_size);
}

int64_t kernelWithTensorListInputWithOutput(const c10::List<Tensor>& input1) {
  // 返回输入列表的大小作为输出
  return input1.size();
}
// 定义测试函数 OperatorRegistrationTestFunctionBasedKernel，测试带有张量列表输入和输出的内核函数注册
TEST(OperatorRegistrationTestFunctionBasedKernel, givenKernelWithTensorListInput_withOutput_whenRegistered_thenCanBeCalled) {
  // 注册操作符，指定函数签名和内核函数，关联到 CPU 分发键
  auto registrar = RegisterOperators()
      .op("_test::tensor_list_input(Tensor[] input) -> int", RegisterOperators::options().kernel<decltype(kernelWithTensorListInputWithOutput), &kernelWithTensorListInputWithOutput>(DispatchKey::CPU));

  // 查找并验证注册的操作符 schema
  auto op = c10::Dispatcher::singleton().findSchema({"_test::tensor_list_input", ""});
  ASSERT_TRUE(op.has_value());

  // 调用操作符，传入两个虚拟张量，并期望返回一个整数结果
  auto outputs = callOp(*op, c10::List<Tensor>({dummyTensor(DispatchKey::CPU), dummyTensor(DispatchKey::CPU)}));
  EXPECT_EQ(1, outputs.size());
  EXPECT_EQ(2, outputs[0].toInt());
}

// 定义全局变量，用于捕获字典大小
int captured_dict_size = 0;

// 定义不带输出的字典输入内核函数
void kernelWithDictInputWithoutOutput(Dict<string, Tensor> input1) {
  // 将输入字典的大小捕获到全局变量中
  captured_dict_size = input1.size();
}

// 测试函数，验证带有字典输入但没有输出的内核函数注册
TEST(OperatorRegistrationTestFunctionBasedKernel, givenKernelWithDictInput_withoutOutput_whenRegistered_thenCanBeCalled) {
  // 注册操作符，指定函数签名和内核函数
  auto registrar = RegisterOperators()
      .op("_test::dict_input(Dict(str, Tensor) input) -> ()", RegisterOperators::options().catchAllKernel<decltype(kernelWithDictInputWithoutOutput), &kernelWithDictInputWithoutOutput>());

  // 查找并验证注册的操作符 schema
  auto op = c10::Dispatcher::singleton().findSchema({"_test::dict_input", ""});
  ASSERT_TRUE(op.has_value());

  // 初始化全局变量为0，创建测试用的输入字典，调用操作符，并期望返回空结果
  captured_dict_size = 0;
  Dict<string, Tensor> dict;
  dict.insert("key1", dummyTensor(DispatchKey::CPU));
  dict.insert("key2", dummyTensor(DispatchKey::CUDA));
  auto outputs = callOp(*op, dict);
  EXPECT_EQ(0, outputs.size());
  EXPECT_EQ(2, captured_dict_size);
}

// 定义带有字典输入和输出的内核函数
string kernelWithDictInputWithOutput(Dict<string, string> input1) {
  // 返回输入字典中 "key2" 对应的值
  return input1.at("key2");
}

// 测试函数，验证带有字典输入和输出的内核函数注册
TEST(OperatorRegistrationTestFunctionBasedKernel, givenKernelWithDictInput_withOutput_whenRegistered_thenCanBeCalled) {
  // 注册操作符，指定函数签名和内核函数
  auto registrar = RegisterOperators()
      .op("_test::dict_input(Dict(str, str) input) -> str", RegisterOperators::options().catchAllKernel<decltype(kernelWithDictInputWithOutput), &kernelWithDictInputWithOutput>());

  // 查找并验证注册的操作符 schema
  auto op = c10::Dispatcher::singleton().findSchema({"_test::dict_input", ""});
  ASSERT_TRUE(op.has_value());

  // 创建测试用的输入字典，调用操作符，并期望返回包含一个字符串结果
  Dict<string, string> dict;
  dict.insert("key1", "value1");
  dict.insert("key2", "value2");
  auto outputs = callOp(*op, dict);
  EXPECT_EQ(1, outputs.size());
  EXPECT_EQ("value2", outputs[0].toStringRef());
}

// 定义带有字典输入和输出的内核函数，直接返回输入字典
Dict<string, string> kernelWithDictOutput(Dict<string, string> input) {
  // 返回与输入相同的字典
  return input;
}
TEST(OperatorRegistrationTestFunctionBasedKernel, givenKernelWithDictOutput_whenRegistered_thenCanBeCalled) {
  // 创建一个操作注册器，注册一个接受字典输入并返回字典输出的操作
  auto registrar = RegisterOperators()
      .op("_test::dict_output(Dict(str, str) input) -> Dict(str, str)", RegisterOperators::options().catchAllKernel<decltype(kernelWithDictOutput), &kernelWithDictOutput>());

  // 查找注册的操作模式，确保操作已成功注册
  auto op = c10::Dispatcher::singleton().findSchema({"_test::dict_output", ""});
  ASSERT_TRUE(op.has_value());

  // 创建一个测试用的输入字典
  Dict<string, string> dict;
  dict.insert("key1", "value1");
  dict.insert("key2", "value2");

  // 调用注册的操作，传入输入字典，并获取输出
  auto outputs = callOp(*op, dict);
  
  // 断言输出的大小为1
  EXPECT_EQ(1, outputs.size());

  // 将输出转换为具体类型的字典，然后断言其大小和键值对
  auto output = c10::impl::toTypedDict<string, string>(outputs[0].toGenericDict());
  EXPECT_EQ(2, output.size());
  EXPECT_EQ("value1", output.at("key1"));
  EXPECT_EQ("value2", output.at("key2"));
}

bool called = false;

void kernelWithoutInputs() {
  called = true;
}

TEST(OperatorRegistrationTestFunctionBasedKernel, givenFallbackKernelWithoutAnyArguments_whenRegistered_thenCanBeCalled) {
  // 注意：没有张量参数的非回退内核不能正常工作，因为无法获取分派键。
  // 对于只有回退内核的操作符，这对向后兼容性非常重要。
  // 创建一个操作注册器，注册一个没有输入参数的回退内核操作
  auto registrar = RegisterOperators()
      .op("_test::no_tensor_args() -> ()", RegisterOperators::options().catchAllKernel<decltype(kernelWithoutInputs), &kernelWithoutInputs>());

  // 查找注册的操作模式，确保操作已成功注册
  auto op = c10::Dispatcher::singleton().findSchema({"_test::no_tensor_args", ""});
  ASSERT_TRUE(op.has_value());

  // 在调用操作之前重置调用标志
  called = false;

  // 调用注册的操作，并断言调用标志已设置为true
  auto outputs = callOp(*op);
  EXPECT_TRUE(called);
}

int64_t kernelWithoutTensorInputs(int64_t arg) {
  return arg + 1;
}

TEST(OperatorRegistrationTestFunctionBasedKernel, givenFallbackKernelWithoutTensorArguments_whenRegistered_thenCanBeCalled) {
  // 注意：没有张量参数的非回退内核不能正常工作，因为无法获取分派键。
  // 对于只有回退内核的操作符，这对向后兼容性非常重要。
  // 创建一个操作注册器，注册一个接受整数参数并返回整数的回退内核操作
  auto registrar = RegisterOperators()
      .op("_test::no_tensor_args(int arg) -> int", RegisterOperators::options().catchAllKernel<decltype(kernelWithoutTensorInputs), &kernelWithoutTensorInputs>());

  // 查找注册的操作模式，确保操作已成功注册
  auto op = c10::Dispatcher::singleton().findSchema({"_test::no_tensor_args", ""});
  ASSERT_TRUE(op.has_value());

  // 调用注册的操作，传入参数3，并断言输出的值为4
  auto outputs = callOp(*op, 3);
  EXPECT_EQ(1, outputs.size());
  EXPECT_EQ(4, outputs[0].toInt());
}

std::optional<Tensor> called_arg2 = c10::nullopt;
std::optional<int64_t> called_arg3 = c10::nullopt;
std::optional<std::string> called_arg4 = c10::nullopt;

void kernelWithOptInputWithoutOutput(Tensor arg1, const std::optional<Tensor>& arg2, std::optional<int64_t> arg3, std::optional<std::string> arg4) {
  // 设置调用标志为true，并记录传入的可选参数值
  called = true;
  called_arg2 = arg2;
  called_arg3 = arg3;
  called_arg4 = arg4;
}
TEST(OperatorRegistrationTestFunctionBasedKernel, givenKernelWithOptionalInputs_withoutOutput_whenRegistered_thenCanBeCalled) {
  // 创建操作符注册器，注册一个具有可选输入但无输出的内核函数
  auto registrar = RegisterOperators().op("_test::opt_input(Tensor arg1, Tensor? arg2, int? arg3, str? arg4) -> ()", RegisterOperators::options().kernel<decltype(kernelWithOptInputWithoutOutput), &kernelWithOptInputWithoutOutput>(DispatchKey::CPU));
  // 查找名称为 "_test::opt_input" 的操作符模式
  auto op = c10::Dispatcher::singleton().findSchema({"_test::opt_input", ""});
  // 断言找到了对应的操作符模式
  ASSERT_TRUE(op.has_value());

  // 设置标志为假
  called = false;
  // 调用操作符，并传入参数
  auto outputs = callOp(*op, dummyTensor(DispatchKey::CPU), dummyTensor(DispatchKey::CPU), c10::IValue(), std::string("text"));
  // 预期输出结果为空
  EXPECT_EQ(0, outputs.size());

  // 验证已调用标志为真
  EXPECT_TRUE(called);
  // 验证已调用的可选参数 arg2 存在
  EXPECT_TRUE(called_arg2.has_value());
  // 验证调用的 arg2 的分发键是 CPU
  EXPECT_EQ(extractDispatchKey(*called_arg2), DispatchKey::CPU);
  // 验证调用的可选参数 arg3 不存在
  EXPECT_FALSE(called_arg3.has_value());
  // 验证调用的可选参数 arg4 存在
  EXPECT_TRUE(called_arg4.has_value());
  // 验证调用的 arg4 的值为 "text"
  EXPECT_EQ(*called_arg4, "text");

  // 重新设置标志为假
  called = false;
  // 再次调用操作符，传入参数
  outputs = callOp(*op, dummyTensor(DispatchKey::CPU), c10::IValue(), 4, c10::IValue());
  // 预期输出结果为空
  EXPECT_EQ(0, outputs.size());

  // 验证已调用标志为真
  EXPECT_TRUE(called);
  // 验证已调用的可选参数 arg2 不存在
  EXPECT_FALSE(called_arg2.has_value());
  // 验证调用的可选参数 arg3 存在
  EXPECT_TRUE(called_arg3.has_value());
  // 验证调用的 arg3 的值为 4
  EXPECT_EQ(*called_arg3, 4);
  // 验证调用的可选参数 arg4 不存在
  EXPECT_FALSE(called_arg4.has_value());
}

std::optional<Tensor> kernelWithOptInputWithOutput(Tensor arg1, const std::optional<Tensor>& arg2, std::optional<int64_t> arg3, std::optional<std::string> arg4) {
  // 设置标志为真
  called = true;
  // 存储传入的 arg2 到全局变量 called_arg2
  called_arg2 = arg2;
  // 存储传入的 arg3 到全局变量 called_arg3
  called_arg3 = arg3;
  // 存储传入的 arg4 到全局变量 called_arg4
  called_arg4 = arg4;
  // 返回 arg2
  return arg2;
}

TEST(OperatorRegistrationTestFunctionBasedKernel, givenKernelWithOptionalInputs_withOutput_whenRegistered_thenCanBeCalled) {
  // 创建操作符注册器，注册一个具有可选输入和输出的内核函数
  auto registrar = RegisterOperators().op("_test::opt_input(Tensor arg1, Tensor? arg2, int? arg3, str? arg4) -> Tensor?", RegisterOperators::options().kernel<decltype(kernelWithOptInputWithOutput), &kernelWithOptInputWithOutput>(DispatchKey::CPU));
  // 查找名称为 "_test::opt_input" 的操作符模式
  auto op = c10::Dispatcher::singleton().findSchema({"_test::opt_input", ""});
  // 断言找到了对应的操作符模式
  ASSERT_TRUE(op.has_value());

  // 设置标志为假
  called = false;
  // 调用操作符，并传入参数
  auto outputs = callOp(*op, dummyTensor(DispatchKey::CPU), dummyTensor(DispatchKey::CPU), c10::IValue(), std::string("text"));
  // 预期输出结果为一个元素
  EXPECT_EQ(1, outputs.size());
  // 预期输出的第一个元素的分发键为 CPU
  EXPECT_EQ(DispatchKey::CPU, extractDispatchKey(outputs[0].toTensor()));

  // 验证已调用标志为真
  EXPECT_TRUE(called);
  // 验证已调用的可选参数 arg2 存在
  EXPECT_TRUE(called_arg2.has_value());
  // 验证调用的 arg2 的分发键是 CPU
  EXPECT_EQ(extractDispatchKey(*called_arg2), DispatchKey::CPU);
  // 验证调用的可选参数 arg3 不存在
  EXPECT_FALSE(called_arg3.has_value());
  // 验证调用的可选参数 arg4 存在
  EXPECT_TRUE(called_arg4.has_value());
  // 验证调用的 arg4 的值为 "text"
  EXPECT_EQ(*called_arg4, "text");

  // 重新设置标志为假
  called = false;
  // 再次调用操作符，传入参数
  outputs = callOp(*op, dummyTensor(DispatchKey::CPU), c10::IValue(), 4, c10::IValue());
  // 预期输出结果为一个元素
  EXPECT_EQ(1, outputs.size());
  // 预期输出的第一个元素为 None
  EXPECT_TRUE(outputs[0].isNone());

  // 验证已调用标志为真
  EXPECT_TRUE(called);
  // 验证已调用的可选参数 arg2 不存在
  EXPECT_FALSE(called_arg2.has_value());
  // 验证调用的可选参数 arg3 存在
  EXPECT_TRUE(called_arg3.has_value());
  // 验证调用的 arg3 的值为 4
  EXPECT_EQ(*called_arg3, 4);
  // 验证调用的可选参数 arg4 不存在
  EXPECT_FALSE(called_arg4.has_value());
}
// 定义一个函数 kernelWithOptInputWithMultipleOutputs，接受四个参数：
// arg1 是一个 Tensor 类型的参数
// arg2 是一个可选的 Tensor 类型的参数
// arg3 是一个可选的 int64_t 类型的参数
// arg4 是一个可选的 std::string 类型的参数
// 函数返回一个 std::tuple，包含 arg2, arg3, arg4 这三个参数
kernelWithOptInputWithMultipleOutputs(Tensor arg1, const std::optional<Tensor>& arg2, std::optional<int64_t> arg3, std::optional<std::string> arg4) {
  return std::make_tuple(arg2, arg3, arg4);
}

// 定义一个单元测试函数 OperatorRegistrationTestFunctionBasedKernel，验证注册的带有多个可选输入的核心函数。
// 使用 RegisterOperators().op 注册一个操作 "_test::opt_input"，指定了输入参数类型和核心函数的绑定。
// 确认操作已注册成功，并调用该操作，验证输出结果。
TEST(OperatorRegistrationTestFunctionBasedKernel, givenKernelWithOptionalInputs_withMultipleOutputs_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators().op("_test::opt_input(Tensor arg1, Tensor? arg2, int? arg3, str? arg4) -> (Tensor?, int?, str?)", RegisterOperators::options().kernel<decltype(kernelWithOptInputWithMultipleOutputs), &kernelWithOptInputWithMultipleOutputs>(DispatchKey::CPU));
  auto op = c10::Dispatcher::singleton().findSchema({"_test::opt_input", ""});
  ASSERT_TRUE(op.has_value());

  // 调用注册的操作，使用给定的参数调用 callOp 函数，并验证输出结果。
  auto outputs = callOp(*op, dummyTensor(DispatchKey::CPU), dummyTensor(DispatchKey::CPU), c10::IValue(), std::string("text"));
  EXPECT_EQ(3, outputs.size());
  EXPECT_EQ(DispatchKey::CPU, extractDispatchKey(outputs[0].toTensor()));
  EXPECT_TRUE(outputs[1].isNone());
  EXPECT_EQ("text", outputs[2].toStringRef());

  // 再次调用注册的操作，使用不同的参数调用 callOp 函数，并验证输出结果。
  outputs = callOp(*op, dummyTensor(DispatchKey::CPU), c10::IValue(), 4, c10::IValue());
  EXPECT_EQ(3, outputs.size());
  EXPECT_TRUE(outputs[0].isNone());
  EXPECT_EQ(4, outputs[1].toInt());
  EXPECT_TRUE(outputs[2].isNone());
}

// 定义一个核心函数 concatKernel，接受四个参数：
// tensor1 是一个 Tensor 类型的参数
// a 是一个 std::string 类型的参数
// b 是一个 const std::string& 类型的参数
// c 是一个 int64_t 类型的参数
// 函数返回一个 std::string 类型的字符串，将参数 a, b 和 c 拼接成一个字符串返回。
std::string concatKernel(const Tensor& tensor1, std::string a, const std::string& b, int64_t c) {
  return a + b + std::to_string(c);
}

// 定义一个单元测试函数 OperatorRegistrationTestFunctionBasedKernel，验证注册的字符串拼接核心函数。
// 使用 RegisterOperators().op 注册一个操作 "_test::my_op"，指定输入参数类型和核心函数的绑定。
// 调用 expectCallsConcatUnboxed 函数，验证调用注册的操作后的返回结果。
void expectCallsConcatUnboxed(DispatchKey dispatch_key) {
  at::AutoDispatchBelowAutograd mode;

  // 确认指定的操作 "_test::my_op" 已注册，并调用 callOpUnboxed 函数，验证返回结果。
  auto op = c10::Dispatcher::singleton().findSchema({"_test::my_op", ""});
  ASSERT_TRUE(op.has_value());
  std::string result = callOpUnboxed<std::string, const Tensor&, std::string, const std::string&, int64_t>(*op, dummyTensor(dispatch_key), "1", "2", 3);
  EXPECT_EQ("123", result);
}

// 定义一个单元测试函数 OperatorRegistrationTestFunctionBasedKernel，验证注册的核心函数在不指定模式的情况下推断模式。
// 使用 RegisterOperators().op 注册一个操作 "_test::no_schema_specified"，并使用 catchAllKernel 推断函数的核心。
// 确认指定的操作已注册，并验证推断的模式与给定的模式相匹配。
TEST(OperatorRegistrationTestFunctionBasedKernel, givenKernel_whenRegisteredWithoutSpecifyingSchema_thenInfersSchema) {
  auto registrar = RegisterOperators()
      .op("_test::no_schema_specified", RegisterOperators().options().catchAllKernel<decltype(kernelForSchemaInference), &kernelForSchemaInference>());

  auto op = c10::Dispatcher::singleton().findSchema({"_test::no_schema_specified", ""});
  ASSERT_TRUE(op.has_value());

  // 解析给定的模式，与推断的模式进行比较，确认二者一致。
  std::optional<std::string> differences = c10::findSchemaDifferences(torch::jit::parseSchema("_test::no_schema_specified(Tensor arg1, int arg2, Tensor[] arg3) -> (int, Tensor)"), op->schema());
  EXPECT_FALSE(differences.has_value());
}
template<class Return, class... Args> struct kernel_func final {
  // 模板结构体，用于定义具有返回类型和参数列表的 kernel_func
  static Return func(Args...) { return {}; }
};

template<class... Args> struct kernel_func<void, Args...> final {
  // 特化模板结构体，用于定义无返回类型的 kernel_func
  static void func(Args...) {}
};

TEST(OperatorRegistrationTestFunctionBasedKernel, givenMismatchedKernel_withDifferentNumArguments_whenRegistering_thenFails) {
  // 断言此处不会失败，因为匹配成功
  RegisterOperators()
      .op("_test::mismatch(Tensor arg) -> int", RegisterOperators::options().kernel<decltype(kernel_func<int64_t, Tensor>::func), &kernel_func<int64_t, Tensor>::func>(DispatchKey::CPU));

  // 现在是一组不匹配的模式
  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg, Tensor arg2) -> int", RegisterOperators::options().kernel<decltype(kernel_func<int64_t, Tensor>::func), &kernel_func<int64_t, Tensor>::func>(DispatchKey::CPU));
    }, "The number of arguments is different. 2 vs 1"
  );

  // 断言此处不会失败，因为匹配成功
  RegisterOperators()
      .op("_test::mismatch(Tensor arg, Tensor arg2) -> ()", RegisterOperators::options().kernel<decltype(kernel_func<void, Tensor, Tensor>::func), &kernel_func<void, Tensor, Tensor>::func>(DispatchKey::CPU));

  // 现在是一组不匹配的模式
  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch() -> ()", RegisterOperators::options().kernel<decltype(kernel_func<void, Tensor, Tensor>::func), &kernel_func<void, Tensor, Tensor>::func>(DispatchKey::CPU));
    }, "The number of arguments is different. 0 vs 2"
  );

  // 现在是一组不匹配的模式
  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> ()", RegisterOperators::options().kernel<decltype(kernel_func<void, Tensor, Tensor>::func), &kernel_func<void, Tensor, Tensor>::func>(DispatchKey::CPU));
    }, "The number of arguments is different. 1 vs 2"
  );

  // 现在是一组不匹配的模式
  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg, Tensor arg2, Tensor arg3) -> ()", RegisterOperators::options().kernel<decltype(kernel_func<void, Tensor, Tensor>::func), &kernel_func<void, Tensor, Tensor>::func>(DispatchKey::CPU));
    }, "The number of arguments is different. 3 vs 2"
  );
}

TEST(OperatorRegistrationTestFunctionBasedKernel, givenMismatchedKernel_withDifferentArgumentType_whenRegistering_thenFails) {
  // 断言此处不会失败，因为匹配成功
  RegisterOperators()
      .op("_test::mismatch(Tensor arg1, int arg2) -> int", RegisterOperators::options().kernel<decltype(kernel_func<int64_t, Tensor, int64_t>::func), &kernel_func<int64_t, Tensor, int64_t>::func>(DispatchKey::CPU));

  // 现在是一组不匹配的模式
  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg1, float arg2) -> int", RegisterOperators::options().kernel<decltype(kernel_func<int64_t, Tensor, int64_t>::func), &kernel_func<int64_t, Tensor, int64_t>::func>(DispatchKey::CPU));
  );



  // 关闭前面的 lambda 表达式
  );

  expectThrows<c10::Error>([] {
    // 在 RegisterOperators 上注册一个操作，定义了一个带有类型不匹配的操作
    RegisterOperators()
        // 注册一个操作 "_test::mismatch(int arg1, int arg2) -> int"
        // 使用 kernel_func<int64_t, Tensor, int64_t>::func 作为内核函数
        // 指定在 CPU 上进行分发
        .op("_test::mismatch(int arg1, int arg2) -> int", RegisterOperators::options().kernel<decltype(kernel_func<int64_t, Tensor, int64_t>::func), &kernel_func<int64_t, Tensor, int64_t>::func>(DispatchKey::CPU));
    // lambda 表达式结束
    }, "Type mismatch in argument 1: int vs Tensor"
  );



  // 关闭前面的 lambda 表达式
  );



// 该段代码主要是使用 C++ 的 RegisterOperators 类注册一个操作，并在此过程中捕获和期望的异常。
TEST(OperatorRegistrationTestFunctionBasedKernel, givenMismatchedKernel_withDifferentNumReturns_whenRegistering_thenFails) {
  // 注册一个操作符，匹配特定的内核函数，并指定CPU调度键
  RegisterOperators()
      .op("_test::mismatch(Tensor arg) -> int", RegisterOperators::options().kernel<decltype(kernel_func<int64_t, Tensor>::func), &kernel_func<int64_t, Tensor>::func>(DispatchKey::CPU));

  // 预期以下注册操作会抛出异常，因为返回值数量不同
  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> ()", RegisterOperators::options().kernel<decltype(kernel_func<int64_t, Tensor>::func), &kernel_func<int64_t, Tensor>::func>(DispatchKey::CPU));
    }, "The number of returns is different. 0 vs 1"
  );

  // 预期以下注册操作会抛出异常，因为返回值数量不同
  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> (int, int)", RegisterOperators::options().kernel<decltype(kernel_func<int64_t, Tensor>::func), &kernel_func<int64_t, Tensor>::func>(DispatchKey::CPU));
    }, "The number of returns is different. 2 vs 1"
  );

  // 注册一个操作符，匹配特定的内核函数，并指定CPU调度键
  RegisterOperators()
      .op("_test::mismatch(Tensor arg) -> ()", RegisterOperators::options().kernel<decltype(kernel_func<void, Tensor>::func), &kernel_func<void, Tensor>::func>(DispatchKey::CPU));

  // 预期以下注册操作会抛出异常，因为返回值数量不同
  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> Tensor", RegisterOperators::options().kernel<decltype(kernel_func<void, Tensor>::func), &kernel_func<void, Tensor>::func>(DispatchKey::CPU));
    }, "The number of returns is different. 1 vs 0"
  );

  // 预期以下注册操作会抛出异常，因为返回值数量不同
  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> (Tensor, Tensor)", RegisterOperators::options().kernel<decltype(kernel_func<void, Tensor>::func), &kernel_func<void, Tensor>::func>(DispatchKey::CPU));
    }, "The number of returns is different. 2 vs 0"
  );

  // 注册一个操作符，匹配特定的内核函数，并指定CPU调度键
  RegisterOperators()
      .op("_test::mismatch(Tensor arg) -> (Tensor, Tensor)", RegisterOperators::options().kernel<decltype(kernel_func<std::tuple<Tensor, Tensor>, Tensor>::func), &kernel_func<std::tuple<Tensor, Tensor>, Tensor>::func>(DispatchKey::CPU));

  // 预期以下注册操作会抛出异常，因为返回值数量不同
  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> ()", RegisterOperators::options().kernel<decltype(kernel_func<std::tuple<Tensor, Tensor>, Tensor>::func), &kernel_func<std::tuple<Tensor, Tensor>, Tensor>::func>(DispatchKey::CPU));
    }, "The number of returns is different. 0 vs 2"
  );

  // 预期以下注册操作会抛出异常，因为返回值数量不同
  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> Tensor", RegisterOperators::options().kernel<decltype(kernel_func<std::tuple<Tensor, Tensor>, Tensor>::func), &kernel_func<std::tuple<Tensor, Tensor>, Tensor>::func>(DispatchKey::CPU));
    }, "The number of returns is different. 1 vs 2"
  );


    // 抛出异常断言，验证代码块内操作是否抛出 c10::Error 异常
    expectThrows<c10::Error>([] {
        // 注册运算符，声明一个操作 "_test::mismatch(Tensor arg) -> (Tensor, Tensor, Tensor)"
        // 并指定其内核函数和分发键为 CPU
        RegisterOperators()
            .op("_test::mismatch(Tensor arg) -> (Tensor, Tensor, Tensor)", RegisterOperators::options()
                .kernel<decltype(kernel_func<std::tuple<Tensor, Tensor>, Tensor>::func), &kernel_func<std::tuple<Tensor, Tensor>, Tensor>::func>(DispatchKey::CPU));
    }, "The number of returns is different. 1 vs 2"
  );



  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> (Tensor, Tensor, Tensor)", RegisterOperators::options().kernel<decltype(kernel_func<std::tuple<Tensor, Tensor>, Tensor>::func), &kernel_func<std::tuple<Tensor, Tensor>, Tensor>::func>(DispatchKey::CPU));
    }, "The number of returns is different. 3 vs 2"
  );


  // 抛出异常断言，验证代码块内操作是否抛出 c10::Error 异常
  expectThrows<c10::Error>([] {
      // 注册运算符，声明一个操作 "_test::mismatch(Tensor arg) -> (Tensor, Tensor, Tensor)"
      // 并指定其内核函数和分发键为 CPU
      RegisterOperators()
          .op("_test::mismatch(Tensor arg) -> (Tensor, Tensor, Tensor)", RegisterOperators::options()
              .kernel<decltype(kernel_func<std::tuple<Tensor, Tensor>, Tensor>::func), &kernel_func<std::tuple<Tensor, Tensor>, Tensor>::func>(DispatchKey::CPU));
  }, "The number of returns is different. 3 vs 2"
);
}

TEST(OperatorRegistrationTestFunctionBasedKernel, givenMismatchedKernel_withDifferentReturnTypes_whenRegistering_thenFails) {
  // 注册一个操作符，使用特定的内核函数处理来自CPU的 _test::mismatch(Tensor arg) -> int 类型的操作符
  RegisterOperators()
      .op("_test::mismatch(Tensor arg) -> int", RegisterOperators::options().kernel<decltype(kernel_func<int64_t, Tensor>::func), &kernel_func<int64_t, Tensor>::func>(DispatchKey::CPU));

  // 现在是一组不匹配的模式
  expectThrows<c10::Error>([] {
    // 注册一个操作符，预期返回类型为 Tensor，但实际返回 int 类型，应该抛出异常
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> Tensor", RegisterOperators::options().kernel<decltype(kernel_func<int64_t, Tensor>::func), &kernel_func<int64_t, Tensor>::func>(DispatchKey::CPU));
    }, "Type mismatch in return 1: Tensor vs int"
  );

  expectThrows<c10::Error>([] {
    // 注册一个操作符，预期返回类型为 float，但实际返回 int 类型，应该抛出异常
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> float", RegisterOperators::options().kernel<decltype(kernel_func<int64_t, Tensor>::func), &kernel_func<int64_t, Tensor>::func>(DispatchKey::CPU));
    }, "Type mismatch in return 1: float vs int"
  );

  // 注册一个操作符，使用特定的内核函数处理来自CPU的 _test::mismatch(Tensor arg) -> Tensor 类型的操作符
  RegisterOperators()
      .op("_test::mismatch(Tensor arg) -> Tensor", RegisterOperators::options().kernel<decltype(kernel_func<Tensor, Tensor>::func), &kernel_func<Tensor, Tensor>::func>(DispatchKey::CPU));

  // 现在是一组不匹配的模式
  expectThrows<c10::Error>([] {
    // 注册一个操作符，预期返回类型为 float，但实际返回 Tensor 类型，应该抛出异常
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> float", RegisterOperators::options().kernel<decltype(kernel_func<Tensor, Tensor>::func), &kernel_func<Tensor, Tensor>::func>(DispatchKey::CPU));
    }, "Type mismatch in return 1: float vs Tensor"
  );

  // 注册一个操作符，使用特定的内核函数处理来自CPU的 _test::mismatch(Tensor arg) -> (Tensor, int) 类型的操作符
  RegisterOperators()
      .op("_test::mismatch(Tensor arg) -> (Tensor, int)", RegisterOperators::options().kernel<decltype(kernel_func<std::tuple<Tensor, int64_t>, Tensor>::func), &kernel_func<std::tuple<Tensor, int64_t>, Tensor>::func>(DispatchKey::CPU));

  // 现在是一组不匹配的模式
  expectThrows<c10::Error>([] {
    // 注册一个操作符，预期返回类型为 (Tensor, float)，但实际返回 (Tensor, int) 类型，应该抛出异常
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> (Tensor, float)", RegisterOperators::options().kernel<decltype(kernel_func<std::tuple<Tensor, int64_t>, Tensor>::func), &kernel_func<std::tuple<Tensor, int64_t>, Tensor>::func>(DispatchKey::CPU));
    }, "Type mismatch in return 2: float vs int"
  );

  expectThrows<c10::Error>([] {
    // 注册一个操作符，预期返回类型为 (int, int)，但实际返回 (Tensor, int) 类型，应该抛出异常
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> (int, int)", RegisterOperators::options().kernel<decltype(kernel_func<std::tuple<Tensor, int64_t>, Tensor>::func), &kernel_func<std::tuple<Tensor, int64_t>, Tensor>::func>(DispatchKey::CPU));
    }, "Type mismatch in return 1: int vs Tensor"
  );
}

}
```