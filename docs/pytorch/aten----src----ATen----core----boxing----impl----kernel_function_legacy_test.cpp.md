# `.\pytorch\aten\src\ATen\core\boxing\impl\kernel_function_legacy_test.cpp`

```py
// 包含 Google Test 框架的头文件
#include <gtest/gtest.h>

// 忽略使用了废弃声明的警告
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

// 包含 ATen 库的测试辅助函数头文件
#include <ATen/core/boxing/impl/test_helpers.h>
// 包含 ATen 库的操作注册相关头文件
#include <ATen/core/op_registration/op_registration.h>
// 包含 ATen 库的张量类定义头文件
#include <ATen/core/Tensor.h>
// 包含 Torch 的函数模式解析器头文件
#include <torch/csrc/jit/frontend/function_schema_parser.h>

// 包含 ATen 库的旧类型调度头文件
#include <ATen/core/LegacyTypeDispatch.h>

/**
 * 本文件用于测试使用函数式 API 注册内核的旧版 API。
 *
 * > namespace { Tensor kernel(Tensor a) {...} }
 * > static auto registry = c10::RegisterOperators()
 * >   .op("func(Tensor a) -> Tensor", &kernel);
 */

// 使用 c10 命名空间中的 RegisterOperators 类
using c10::RegisterOperators;
// 使用 c10 命名空间中的 DispatchKey 类型
using c10::DispatchKey;
// 使用 c10 命名空间中的 Stack 类型
using c10::Stack;
// 使用 std 命名空间中的 make_unique 函数
using std::make_unique;
// 使用 c10 命名空间中的 intrusive_ptr 类型
using c10::intrusive_ptr;
// 使用 c10 命名空间中的 Dict 类型
using c10::Dict;
// 使用 at 命名空间中的 Tensor 类
using at::Tensor;
// 使用 std 命名空间中的 string 类型
using std::string;
// 使用 std 命名空间中的 unique_ptr 类型
using std::unique_ptr;

// 匿名命名空间，用于隐藏内部实现细节
namespace {

// 错误内核函数，从给定张量和输入返回 0，并断言永远不应该调用此内核
int64_t errorKernel(const Tensor& tensor, int64_t input) {
  EXPECT_TRUE(false); // this kernel should never be called
  return 0;
}

// 增量内核函数，从给定张量和输入返回 input + 1
int64_t incrementKernel(const Tensor& tensor, int64_t input) {
  return input + 1;
}

// 预期调用增量函数，使用给定的调度键调用操作，并验证结果
void expectCallsIncrement(DispatchKey dispatch_key) {
  at::AutoDispatchBelowAutograd mode;

  // 断言找到指定模式下的模式和 CPU 内核
  auto op = c10::Dispatcher::singleton().findSchema({"_test::my_op", ""});
  ASSERT_TRUE(op.has_value());
  auto result = callOp(*op, dummyTensor(dispatch_key), 5);
  EXPECT_EQ(1, result.size());
  EXPECT_EQ(6, result[0].toInt());
}

// 测试用例，验证当注册增量内核后，可以成功调用该内核
TEST(OperatorRegistrationTestLegacyFunctionBasedKernel, givenKernel_whenRegistered_thenCanBeCalled) {
  auto registrar = RegisterOperators().op("_test::my_op(Tensor dummy, int input) -> int", &incrementKernel);
  expectCallsIncrement(DispatchKey::CPU);
}

// 测试用例，验证当在构造函数中注册增量内核后，可以成功调用该内核
TEST(OperatorRegistrationTestLegacyFunctionBasedKernel, givenKernel_whenRegisteredInConstructor_thenCanBeCalled) {
  auto registrar = RegisterOperators("_test::my_op(Tensor dummy, int input) -> int", &incrementKernel);
  expectCallsIncrement(DispatchKey::CPU);
}

// 测试用例，验证当注册多个操作和内核时，可以正确调用指定的内核
TEST(OperatorRegistrationTestLegacyFunctionBasedKernel, givenMultipleOperatorsAndKernels_whenRegisteredInOneRegistrar_thenCallsRightKernel) {
  auto registrar = RegisterOperators()
      .op("_test::my_op(Tensor dummy, int input) -> int", &incrementKernel)
      .op("_test::error(Tensor dummy, int input) -> int", &errorKernel);
  expectCallsIncrement(DispatchKey::CPU);
}

// 测试用例，验证当分别在多个注册器中注册不同的内核时，可以正确调用指定的内核
TEST(OperatorRegistrationTestLegacyFunctionBasedKernel, givenMultipleOperatorsAndKernels_whenRegisteredInMultipleRegistrars_thenCallsRightKernel) {
  auto registrar1 = RegisterOperators().op("_test::my_op(Tensor dummy, int input) -> int", &incrementKernel);
  auto registrar2 = RegisterOperators().op("_test::error(Tensor dummy, int input) -> int", &errorKernel);
  expectCallsIncrement(DispatchKey::CPU);
}

// 测试用例，验证当注册增量内核后，并且超出作用域后，无法再次成功调用该内核
TEST(OperatorRegistrationTestLegacyFunctionBasedKernel, givenKernel_whenRegistrationRunsOutOfScope_thenCannotBeCalledAnymore) {
  {
    auto registrar = RegisterOperators().op("_test::my_op(Tensor dummy, int input) -> int", &incrementKernel);
    expectCallsIncrement(DispatchKey::CPU);
  }

  // 确保在这之前使用 CPU 分发键增加了调用次数
  expectCallsIncrement(DispatchKey::CPU);

  // 现在注册器已被销毁。断言 _test::my_op 操作符已经不存在。
  expectDoesntFindOperator("_test::my_op");
}

// 设置一个全局变量，用于记录函数是否被调用过
bool was_called = false;

// 定义一个没有输出的内核函数，当被调用时将设置 `was_called` 为 true
void kernelWithoutOutput(const Tensor&) {
  was_called = true;
}

// 测试函数：注册无输出的内核函数，并验证其能否成功调用
TEST(OperatorRegistrationTestLegacyFunctionBasedKernel, givenKernelWithoutOutput_whenRegistered_thenCanBeCalled) {
  // 注册操作符并关联到对应的内核函数
  auto registrar = RegisterOperators().op("_test::no_return(Tensor dummy) -> ()", &kernelWithoutOutput);

  // 查找注册的操作符，并断言成功找到
  auto op = c10::Dispatcher::singleton().findSchema({"_test::no_return", ""});
  ASSERT_TRUE(op.has_value());
  
  // 初始化 `was_called` 为 false，调用操作符，检查 `was_called` 是否变为 true
  was_called = false;
  auto result = callOp(*op, dummyTensor(DispatchKey::CPU));
  EXPECT_TRUE(was_called);
  // 预期结果的大小为 0
  EXPECT_EQ(0, result.size());
}

// 定义一个返回空元组的内核函数，当被调用时将设置 `was_called` 为 true
std::tuple<> kernelWithZeroOutputs(const Tensor&) {
  was_called = true;
  return std::make_tuple();
}

// 测试函数：注册返回空元组的内核函数，并验证其能否成功调用
TEST(OperatorRegistrationTestLegacyFunctionBasedKernel, givenKernelWithZeroOutputs_whenRegistered_thenCanBeCalled) {
  // 注册操作符并关联到对应的内核函数
  auto registrar = RegisterOperators().op("_test::zero_outputs(Tensor dummy) -> ()", &kernelWithZeroOutputs);

  // 查找注册的操作符，并断言成功找到
  auto op = c10::Dispatcher::singleton().findSchema({"_test::zero_outputs", ""});
  ASSERT_TRUE(op.has_value());

  // 初始化 `was_called` 为 false，调用操作符，检查 `was_called` 是否变为 true
  was_called = false;
  auto result = callOp(*op, dummyTensor(DispatchKey::CPU));
  EXPECT_TRUE(was_called);
  // 预期结果的大小为 0
  EXPECT_EQ(0, result.size());
}

// 定义一个返回整数的内核函数，计算两个整数的和并返回
int64_t kernelWithIntOutput(Tensor, int64_t a, int64_t b) {
  return a + b;
}

// 测试函数：注册返回整数的内核函数，并验证其能否成功调用
TEST(OperatorRegistrationTestLegacyFunctionBasedKernel, givenKernelWithIntOutput_whenRegistered_thenCanBeCalled) {
  // 注册操作符并关联到对应的内核函数
  auto registrar = RegisterOperators()
      .op("_test::int_output(Tensor dummy, int a, int b) -> int", &kernelWithIntOutput);

  // 查找注册的操作符，并断言成功找到
  auto op = c10::Dispatcher::singleton().findSchema({"_test::int_output", ""});
  ASSERT_TRUE(op.has_value());

  // 调用操作符，传入参数并检查返回结果
  auto result = callOp(*op, dummyTensor(DispatchKey::CPU), 3, 6);
  // 预期结果的大小为 1
  EXPECT_EQ(1, result.size());
  // 检查返回的整数值是否为 9
  EXPECT_EQ(9, result[0].toInt());
}

// 定义一个返回张量的内核函数，直接返回输入的张量
Tensor kernelWithTensorOutput(const Tensor& input) {
  return input;
}

// 测试函数：注册返回张量的内核函数，并验证其能否成功调用
TEST(OperatorRegistrationTestLegacyFunctionBasedKernel, givenKernelWithTensorOutput_whenRegistered_thenCanBeCalled) {
  // 注册操作符并关联到对应的内核函数
  auto registrar = RegisterOperators()
      .op("_test::returning_tensor(Tensor input) -> Tensor", &kernelWithTensorOutput);

  // 查找注册的操作符，并断言成功找到
  auto op = c10::Dispatcher::singleton().findSchema({"_test::returning_tensor", ""});
  ASSERT_TRUE(op.has_value());

  // 调用操作符，传入不同的分发键（CPU 和 CUDA），并检查返回结果
  auto result = callOp(*op, dummyTensor(DispatchKey::CPU));
  // 预期结果的大小为 1
  EXPECT_EQ(1, result.size());
  // 检查返回的张量的分发键是否为 CPU
  EXPECT_EQ(DispatchKey::CPU, extractDispatchKey(result[0].toTensor()));

  result = callOp(*op, dummyTensor(DispatchKey::CUDA));
  // 预期结果的大小为 1
  EXPECT_EQ(1, result.size());
  // 检查返回的张量的分发键是否为 CUDA
  EXPECT_EQ(DispatchKey::CUDA, extractDispatchKey(result[0].toTensor()));
}

// 定义一个返回张量列表的内核函数，直接返回传入的三个张量组成的列表
std::vector<Tensor> kernelWithTensorListOutput(const Tensor& input1, const Tensor& input2, const Tensor& input3) {
  return {input1, input2, input3};
}
TEST(OperatorRegistrationTestLegacyFunctionBasedKernel, givenKernelWithTensorListOutput_whenRegistered_thenCanBeCalled) {
  // 创建操作注册器，并注册一个接受三个张量输入并返回张量数组的操作
  auto registrar = RegisterOperators()
      .op("_test::list_output(Tensor input1, Tensor input2, Tensor input3) -> Tensor[]", &kernelWithTensorListOutput);

  // 查找注册表中的操作模式，并确保成功找到
  auto op = c10::Dispatcher::singleton().findSchema({"_test::list_output", ""});
  ASSERT_TRUE(op.has_value());

  // 调用找到的操作，并获取返回结果
  auto result = callOp(*op, dummyTensor(DispatchKey::CPU), dummyTensor(DispatchKey::CUDA), dummyTensor(DispatchKey::CPU));
  // 断言返回结果的大小为1
  EXPECT_EQ(1, result.size());
  // 断言结果中第一个张量数组的大小为3
  EXPECT_EQ(3, result[0].toTensorVector().size());
  // 断言结果中第一个张量的调度键为CPU
  EXPECT_EQ(DispatchKey::CPU, extractDispatchKey(result[0].toTensorVector()[0]));
  // 断言结果中第二个张量的调度键为CUDA
  EXPECT_EQ(DispatchKey::CUDA, extractDispatchKey(result[0].toTensorVector()[1]));
  // 断言结果中第三个张量的调度键为CPU
  EXPECT_EQ(DispatchKey::CPU, extractDispatchKey(result[0].toTensorVector()[2]));
}

std::vector<int64_t> kernelWithIntListOutput(const Tensor&, int64_t input1, int64_t input2, int64_t input3) {
  // 返回一个包含输入整数的向量
  return {input1, input2, input3};
}

TEST(OperatorRegistrationTestLegacyFunctionBasedKernel, givenKernelWithIntListOutput_whenRegistered_thenCanBeCalled) {
  // 创建操作注册器，并注册一个接受一个张量和三个整数输入并返回整数数组的操作
  auto registrar = RegisterOperators()
      .op("_test::list_output(Tensor dummy, int input1, int input2, int input3) -> int[]", &kernelWithIntListOutput);

  // 查找注册表中的操作模式，并确保成功找到
  auto op = c10::Dispatcher::singleton().findSchema({"_test::list_output", ""});
  ASSERT_TRUE(op.has_value());

  // 调用找到的操作，并获取返回结果
  auto result = callOp(*op, dummyTensor(DispatchKey::CPU), 2, 4, 6);
  // 断言返回结果的大小为1
  EXPECT_EQ(1, result.size());
  // 断言结果中整数向量的大小为3
  EXPECT_EQ(3, result[0].toIntVector().size());
  // 断言结果中第一个整数为2
  EXPECT_EQ(2, result[0].toIntVector()[0]);
  // 断言结果中第二个整数为4
  EXPECT_EQ(4, result[0].toIntVector()[1]);
  // 断言结果中第三个整数为6
  EXPECT_EQ(6, result[0].toIntVector()[2]);
}

std::tuple<Tensor, int64_t, std::vector<Tensor>, std::optional<int64_t>, Dict<string, Tensor>> kernelWithMultipleOutputs(Tensor) {
  // 创建一个字典，将两个张量命名为"first"和"second"
  Dict<string, Tensor> dict;
  dict.insert("first", dummyTensor(DispatchKey::CPU));
  dict.insert("second", dummyTensor(DispatchKey::CUDA));
  // 返回一个包含多种类型输出的元组
  return std::tuple<Tensor, int64_t, std::vector<Tensor>, std::optional<int64_t>, Dict<string, Tensor>>(
    dummyTensor(DispatchKey::CUDA),  // 第一个张量输出，调度键为CUDA
    5,                               // 第二个整数输出为5
    {dummyTensor(DispatchKey::CPU), dummyTensor(DispatchKey::CUDA)},  // 第三个张量数组输出，包含两个张量，分别为CPU和CUDA
    std::optional<int64_t>(std::in_place, 0),  // 第四个可选整数输出，值为0
    dict  // 第五个字典输出，包含两个张量
  );
}
TEST(OperatorRegistrationTestLegacyFunctionBasedKernel, givenKernelWithMultipleOutputs_whenRegistered_thenCanBeCalled) {
  // 注册操作符，将多个输出的内核函数与操作符字符串关联起来
  auto registrar = RegisterOperators()
     .op("_test::multiple_outputs(Tensor dummy) -> (Tensor, int, Tensor[], int?, Dict(str, Tensor))", &kernelWithMultipleOutputs);

  // 查找注册的操作符，确保能够找到对应的操作符模式
  auto op = c10::Dispatcher::singleton().findSchema({"_test::multiple_outputs", ""});
  ASSERT_TRUE(op.has_value());

  // 调用操作符，传入虚拟的张量作为参数，获取返回结果
  auto result = callOp(*op, dummyTensor(DispatchKey::CPU));
  EXPECT_EQ(5, result.size());
  // 检查第一个返回值的分发键是否为 CUDA
  EXPECT_EQ(DispatchKey::CUDA, extractDispatchKey(result[0].toTensor()));
  // 检查第二个返回值是否为整数 5
  EXPECT_EQ(5, result[1].toInt());
  // 检查第三个返回值是否为包含两个张量的向量，并分别检查它们的分发键
  EXPECT_EQ(2, result[2].toTensorVector().size());
  EXPECT_EQ(DispatchKey::CPU, extractDispatchKey(result[2].toTensorVector()[0]));
  EXPECT_EQ(DispatchKey::CUDA, extractDispatchKey(result[2].toTensorVector()[1]));
  // 检查第四个返回值是否为整数 0
  EXPECT_EQ(0, result[3].toInt());
  // 将第五个返回值转换为字典，并检查字典的大小和其中键对应的张量的分发键
  auto result_dict = c10::impl::toTypedDict<string, Tensor>(result[4].toGenericDict());
  EXPECT_EQ(2, result_dict.size());
  EXPECT_EQ(DispatchKey::CPU, extractDispatchKey(result_dict.at("first")));
  EXPECT_EQ(DispatchKey::CUDA, extractDispatchKey(result_dict.at("second")));
}

Tensor kernelWithTensorInputByReferenceWithOutput(const Tensor& input1) {
  return input1;
}

Tensor kernelWithTensorInputByValueWithOutput(Tensor input1) {
  return input1;
}

TEST(OperatorRegistrationTestLegacyFunctionBasedKernel, givenKernelWithTensorInputByReference_withOutput_whenRegistered_thenCanBeCalled) {
  // 注册操作符，将接受引用输入的内核函数与操作符字符串关联起来
  auto registrar = RegisterOperators()
      .op("_test::tensor_input(Tensor input) -> Tensor", &kernelWithTensorInputByReferenceWithOutput);

  // 查找注册的操作符，确保能够找到对应的操作符模式
  auto op = c10::Dispatcher::singleton().findSchema({"_test::tensor_input", ""});
  ASSERT_TRUE(op.has_value());

  // 调用操作符，传入虚拟的张量作为参数，获取返回结果
  auto result = callOp(*op, dummyTensor(DispatchKey::CPU));
  EXPECT_EQ(1, result.size());
  // 检查返回的张量的分发键是否为 CPU
  EXPECT_EQ(DispatchKey::CPU, extractDispatchKey(result[0].toTensor()));

  // 使用 CUDA 分发键再次调用操作符，检查返回结果的分发键是否为 CUDA
  result = callOp(*op, dummyTensor(DispatchKey::CUDA));
  EXPECT_EQ(1, result.size());
  EXPECT_EQ(DispatchKey::CUDA, extractDispatchKey(result[0].toTensor()));
}

TEST(OperatorRegistrationTestLegacyFunctionBasedKernel, givenKernelWithTensorInputByValue_withOutput_whenRegistered_thenCanBeCalled) {
  // 注册操作符，将接受值输入的内核函数与操作符字符串关联起来
  auto registrar = RegisterOperators()
      .op("_test::tensor_input(Tensor input) -> Tensor", &kernelWithTensorInputByValueWithOutput);

  // 查找注册的操作符，确保能够找到对应的操作符模式
  auto op = c10::Dispatcher::singleton().findSchema({"_test::tensor_input", ""});
  ASSERT_TRUE(op.has_value());

  // 调用操作符，传入虚拟的张量作为参数，获取返回结果
  auto result = callOp(*op, dummyTensor(DispatchKey::CPU));
  EXPECT_EQ(1, result.size());
  // 检查返回的张量的分发键是否为 CPU
  EXPECT_EQ(DispatchKey::CPU, extractDispatchKey(result[0].toTensor()));

  // 使用 CUDA 分发键再次调用操作符，检查返回结果的分发键是否为 CUDA
  result = callOp(*op, dummyTensor(DispatchKey::CUDA));
  EXPECT_EQ(1, result.size());
  EXPECT_EQ(DispatchKey::CUDA, extractDispatchKey(result[0].toTensor()));
}

Tensor captured_input;

void kernelWithTensorInputByReferenceWithoutOutput(const Tensor& input1) {
  // 将输入张量赋值给全局变量 captured_input
  captured_input = input1;
}

void kernelWithTensorInputByValueWithoutOutput(Tensor input1) {
  // 将输入张量赋值给全局变量 captured_input
  captured_input = input1;
}
// 定义测试函数 OperatorRegistrationTestLegacyFunctionBasedKernel，测试使用引用传递的张量输入和无输出的内核函数注册
TEST(OperatorRegistrationTestLegacyFunctionBasedKernel, givenKernelWithTensorInputByReference_withoutOutput_whenRegistered_thenCanBeCalled) {
  // 注册操作符 "_test::tensor_input(Tensor input) -> ()"，关联到 kernelWithTensorInputByReferenceWithoutOutput 函数
  auto registrar = RegisterOperators()
      .op("_test::tensor_input(Tensor input) -> ()", &kernelWithTensorInputByReferenceWithoutOutput);

  // 查找注册的操作符 "_test::tensor_input"
  auto op = c10::Dispatcher::singleton().findSchema({"_test::tensor_input", ""});
  ASSERT_TRUE(op.has_value());

  // 调用操作符，传入一个虚拟的 CPU 张量，期望输出为空
  auto outputs = callOp(*op, dummyTensor(DispatchKey::CPU));
  EXPECT_EQ(0, outputs.size());
  // 检查捕获的输入的调度键是否为 CPU
  EXPECT_EQ(DispatchKey::CPU, extractDispatchKey(captured_input));

  // 再次调用操作符，传入一个虚拟的 CUDA 张量，期望输出为空
  outputs = callOp(*op, dummyTensor(DispatchKey::CUDA));
  EXPECT_EQ(0, outputs.size());
  // 检查捕获的输入的调度键是否为 CUDA
  EXPECT_EQ(DispatchKey::CUDA, extractDispatchKey(captured_input));
}

// 定义测试函数 OperatorRegistrationTestLegacyFunctionBasedKernel，测试使用值传递的张量输入和无输出的内核函数注册
TEST(OperatorRegistrationTestLegacyFunctionBasedKernel, givenKernelWithTensorInputByValue_withoutOutput_whenRegistered_thenCanBeCalled) {
  // 注册操作符 "_test::tensor_input(Tensor input) -> ()"，关联到 kernelWithTensorInputByValueWithoutOutput 函数
  auto registrar = RegisterOperators()
      .op("_test::tensor_input(Tensor input) -> ()", &kernelWithTensorInputByValueWithoutOutput);

  // 查找注册的操作符 "_test::tensor_input"
  auto op = c10::Dispatcher::singleton().findSchema({"_test::tensor_input", ""});
  ASSERT_TRUE(op.has_value());

  // 调用操作符，传入一个虚拟的 CPU 张量，期望输出为空
  auto outputs = callOp(*op, dummyTensor(DispatchKey::CPU));
  EXPECT_EQ(0, outputs.size());
  // 检查捕获的输入的调度键是否为 CPU
  EXPECT_EQ(DispatchKey::CPU, extractDispatchKey(captured_input));

  // 再次调用操作符，传入一个虚拟的 CUDA 张量，期望输出为空
  outputs = callOp(*op, dummyTensor(DispatchKey::CUDA));
  EXPECT_EQ(0, outputs.size());
  // 检查捕获的输入的调度键是否为 CUDA
  EXPECT_EQ(DispatchKey::CUDA, extractDispatchKey(captured_input));
}

// 定义一个全局变量，用于捕获整型输入
int64_t captured_int_input = 0;

// 定义一个接受整型输入的内核函数，没有输出
void kernelWithIntInputWithoutOutput(Tensor, int64_t input1) {
  // 将输入的整数值捕获到全局变量中
  captured_int_input = input1;
}

// 定义测试函数 OperatorRegistrationTestLegacyFunctionBasedKernel，测试使用整数输入和无输出的内核函数注册
TEST(OperatorRegistrationTestLegacyFunctionBasedKernel, givenKernelWithIntInput_withoutOutput_whenRegistered_thenCanBeCalled) {
  // 注册操作符 "_test::int_input(Tensor dummy, int input) -> ()"，关联到 kernelWithIntInputWithoutOutput 函数
  auto registrar = RegisterOperators()
      .op("_test::int_input(Tensor dummy, int input) -> ()", &kernelWithIntInputWithoutOutput);

  // 查找注册的操作符 "_test::int_input"
  auto op = c10::Dispatcher::singleton().findSchema({"_test::int_input", ""});
  ASSERT_TRUE(op.has_value());

  // 初始化捕获的整数输入为 0
  captured_int_input = 0;
  // 调用操作符，传入一个虚拟的 CPU 张量和整数 3，期望输出为空
  auto outputs = callOp(*op, dummyTensor(DispatchKey::CPU), 3);
  EXPECT_EQ(0, outputs.size());
  // 检查捕获的整数输入是否为 3
  EXPECT_EQ(3, captured_int_input);
}

// 定义一个接受整型输入并有整数输出的内核函数
int64_t kernelWithIntInputWithOutput(Tensor, int64_t input1) {
  // 返回输入整数值加一
  return input1 + 1;
}

// 定义测试函数 OperatorRegistrationTestLegacyFunctionBasedKernel，测试使用整数输入和整数输出的内核函数注册
TEST(OperatorRegistrationTestLegacyFunctionBasedKernel, givenKernelWithIntInput_withOutput_whenRegistered_thenCanBeCalled) {
  // 注册操作符 "_test::int_input(Tensor dummy, int input) -> int"，关联到 kernelWithIntInputWithOutput 函数
  auto registrar = RegisterOperators()
      .op("_test::int_input(Tensor dummy, int input) -> int", &kernelWithIntInputWithOutput);

  // 查找注册的操作符 "_test::int_input"
  auto op = c10::Dispatcher::singleton().findSchema({"_test::int_input", ""});
  ASSERT_TRUE(op.has_value());

  // 调用操作符，传入一个虚拟的 CPU 张量和整数 3，期望输出为一个元素的向量，值为 4
  auto outputs = callOp(*op, dummyTensor(DispatchKey::CPU), 3);
  EXPECT_EQ(1, outputs.size());
  EXPECT_EQ(4, outputs[0].toInt());
}

// 定义一个全局变量，用于捕获整型列表输入的大小
int64_t captured_input_list_size = 0;

// 定义一个接受整型列表输入的内核函数，没有输出
void kernelWithIntListInputWithoutOutput(Tensor, const std::vector<int64_t>& input1) {
  // 将输入列表的大小捕获到全局变量中
  captured_input_list_size = input1.size();
}
TEST(OperatorRegistrationTestLegacyFunctionBasedKernel, givenKernelWithIntListInput_withoutOutput_whenRegistered_thenCanBeCalled) {
  // 创建操作注册器，注册一个接受整数列表输入但没有输出的内核函数
  auto registrar = RegisterOperators()
      .op("_test::int_list_input(Tensor dummy, int[] input) -> ()", &kernelWithIntListInputWithoutOutput);

  // 查找并获取指定操作模式的操作符
  auto op = c10::Dispatcher::singleton().findSchema({"_test::int_list_input", ""});
  // 断言确保操作符已经找到
  ASSERT_TRUE(op.has_value());

  // 设置输入列表大小为0
  captured_input_list_size = 0;
  // 调用操作，并获取输出
  auto outputs = callOp(*op, dummyTensor(DispatchKey::CPU), c10::List<int64_t>({2, 4, 6}));
  // 断言输出的大小为0
  EXPECT_EQ(0, outputs.size());
  // 断言捕获的输入列表大小为3
  EXPECT_EQ(3, captured_input_list_size);
}

int64_t kernelWithIntListInputWithOutput(Tensor, const std::vector<int64_t>& input1) {
  // 返回输入列表的大小作为输出
  return input1.size();
}

TEST(OperatorRegistrationTestLegacyFunctionBasedKernel, givenKernelWithIntListInput_withOutput_whenRegistered_thenCanBeCalled) {
  // 创建操作注册器，注册一个接受整数列表输入并返回整数的内核函数
  auto registrar = RegisterOperators()
      .op("_test::int_list_input(Tensor dummy, int[] input) -> int", &kernelWithIntListInputWithOutput);

  // 查找并获取指定操作模式的操作符
  auto op = c10::Dispatcher::singleton().findSchema({"_test::int_list_input", ""});
  // 断言确保操作符已经找到
  ASSERT_TRUE(op.has_value());

  // 调用操作，并获取输出
  auto outputs = callOp(*op, dummyTensor(DispatchKey::CPU), c10::List<int64_t>({2, 4, 6}));
  // 断言输出的大小为1
  EXPECT_EQ(1, outputs.size());
  // 断言输出的第一个元素值为输入列表的大小，即3
  EXPECT_EQ(3, outputs[0].toInt());
}

void kernelWithTensorListInputWithoutOutput(const std::vector<Tensor>& input1) {
  // 将输入列表的大小捕获到全局变量中
  captured_input_list_size = input1.size();
}

TEST(OperatorRegistrationTestLegacyFunctionBasedKernel, givenKernelWithTensorListInput_withoutOutput_whenRegistered_thenCanBeCalled) {
  // 创建操作注册器，注册一个接受张量列表输入但没有输出的内核函数
  auto registrar = RegisterOperators()
      .op("_test::tensor_list_input(Tensor[] input) -> ()", &kernelWithTensorListInputWithoutOutput);

  // 查找并获取指定操作模式的操作符
  auto op = c10::Dispatcher::singleton().findSchema({"_test::tensor_list_input", ""});
  // 断言确保操作符已经找到
  ASSERT_TRUE(op.has_value());

  // 设置输入列表大小为0
  captured_input_list_size = 0;
  // 调用操作，并获取输出
  auto outputs = callOp(*op, c10::List<Tensor>({dummyTensor(DispatchKey::CPU), dummyTensor(DispatchKey::CPU)}));
  // 断言输出的大小为0
  EXPECT_EQ(0, outputs.size());
  // 断言捕获的输入列表大小为2
  EXPECT_EQ(2, captured_input_list_size);
}

int64_t kernelWithTensorListInputWithOutput(const std::vector<Tensor>& input1) {
  // 返回输入列表的大小作为输出
  return input1.size();
}

TEST(OperatorRegistrationTestLegacyFunctionBasedKernel, givenKernelWithTensorListInput_withOutput_whenRegistered_thenCanBeCalled) {
  // 创建操作注册器，注册一个接受张量列表输入并返回整数的内核函数
  auto registrar = RegisterOperators()
      .op("_test::tensor_list_input(Tensor[] input) -> int", &kernelWithTensorListInputWithOutput);

  // 查找并获取指定操作模式的操作符
  auto op = c10::Dispatcher::singleton().findSchema({"_test::tensor_list_input", ""});
  // 断言确保操作符已经找到
  ASSERT_TRUE(op.has_value());

  // 调用操作，并获取输出
  auto outputs = callOp(*op, c10::List<Tensor>({dummyTensor(DispatchKey::CPU), dummyTensor(DispatchKey::CPU)}));
  // 断言输出的大小为1
  EXPECT_EQ(1, outputs.size());
  // 断言输出的第一个元素值为输入列表的大小，即2
  EXPECT_EQ(2, outputs[0].toInt());
}

void kernelWithLegacyTensorVectorInputWithoutOutput(const std::vector<Tensor>& input1) {
  // 将输入列表的大小捕获到全局变量中
  captured_input_list_size = input1.size();
}
TEST(OperatorRegistrationTestLegacyFunctionBasedKernel, givenKernelWithLegacyTensorVectorInput_withoutOutput_whenRegistered_thenCanBeCalled) {
  // 创建操作注册器，注册接受 Tensor 向量输入且无输出的内核函数
  auto registrar = RegisterOperators()
      .op("_test::tensor_list_input(Tensor[] input) -> ()", &kernelWithLegacyTensorVectorInputWithoutOutput);

  // 查找名为 "_test::tensor_list_input" 的操作模式
  auto op = c10::Dispatcher::singleton().findSchema({"_test::tensor_list_input", ""});
  // 断言找到了对应的操作模式
  ASSERT_TRUE(op.has_value());

  // 初始化捕获的输入列表大小为 0
  captured_input_list_size = 0;
  // 调用操作，并捕获输出
  auto outputs = callOp(*op, c10::List<Tensor>({dummyTensor(DispatchKey::CPU), dummyTensor(DispatchKey::CPU)}));
  // 断言输出为空
  EXPECT_EQ(0, outputs.size());
  // 断言捕获的输入列表大小为 2
  EXPECT_EQ(2, captured_input_list_size);
}

int64_t kernelWithLegacyTensorVectorInputWithOutput(const std::vector<Tensor>& input1) {
  // 返回输入向量的大小作为输出
  return input1.size();
}

TEST(OperatorRegistrationTestLegacyFunctionBasedKernel, givenKernelWithLegacyTensorVectorInput_withOutput_whenRegistered_thenCanBeCalled) {
  // 创建操作注册器，注册接受 Tensor 向量输入且返回整数的内核函数
  auto registrar = RegisterOperators()
      .op("_test::tensor_list_input(Tensor[] input) -> int", &kernelWithLegacyTensorVectorInputWithOutput);

  // 查找名为 "_test::tensor_list_input" 的操作模式
  auto op = c10::Dispatcher::singleton().findSchema({"_test::tensor_list_input", ""});
  // 断言找到了对应的操作模式
  ASSERT_TRUE(op.has_value());

  // 调用操作，并捕获输出
  auto outputs = callOp(*op, c10::List<Tensor>({dummyTensor(DispatchKey::CPU), dummyTensor(DispatchKey::CPU)}));
  // 断言输出向量的大小为 1
  EXPECT_EQ(1, outputs.size());
  // 断言输出的第一个元素转换为整数为 2
  EXPECT_EQ(2, outputs[0].toInt());
}

void kernelWithLegacyTensorListInputWithoutOutput(std::vector<Tensor> input1) {
  // 捕获输入列表的大小
  captured_input_list_size = input1.size();
}

TEST(OperatorRegistrationTestLegacyFunctionBasedKernel, givenKernelWithLegacyTensorListInput_withoutOutput_whenRegistered_thenCanBeCalled) {
  // 创建操作注册器，注册接受 Tensor 列表输入且无输出的内核函数
  auto registrar = RegisterOperators()
      .op("_test::tensor_list_input(Tensor[] input) -> ()", &kernelWithLegacyTensorListInputWithoutOutput);

  // 查找名为 "_test::tensor_list_input" 的操作模式
  auto op = c10::Dispatcher::singleton().findSchema({"_test::tensor_list_input", ""});
  // 断言找到了对应的操作模式
  ASSERT_TRUE(op.has_value());

  // 初始化捕获的输入列表大小为 0
  captured_input_list_size = 0;
  // 调用操作，并捕获输出
  auto outputs = callOp(*op, c10::List<Tensor>({dummyTensor(DispatchKey::CPU), dummyTensor(DispatchKey::CPU)}));
  // 断言输出为空
  EXPECT_EQ(0, outputs.size());
  // 断言捕获的输入列表大小为 2
  EXPECT_EQ(2, captured_input_list_size);
}

int64_t kernelWithLegacyTensorListInputWithOutput(std::vector<Tensor> input1) {
  // 返回输入列表的大小作为输出
  return input1.size();
}

TEST(OperatorRegistrationTestLegacyFunctionBasedKernel, givenKernelWithLegacyTensorListInput_withOutput_whenRegistered_thenCanBeCalled) {
  // 创建操作注册器，注册接受 Tensor 列表输入且返回整数的内核函数
  auto registrar = RegisterOperators()
      .op("_test::tensor_list_input(Tensor[] input) -> int", &kernelWithLegacyTensorListInputWithOutput);

  // 查找名为 "_test::tensor_list_input" 的操作模式
  auto op = c10::Dispatcher::singleton().findSchema({"_test::tensor_list_input", ""});
  // 断言找到了对应的操作模式
  ASSERT_TRUE(op.has_value());

  // 调用操作，并捕获输出
  auto outputs = callOp(*op, c10::List<Tensor>({dummyTensor(DispatchKey::CPU), dummyTensor(DispatchKey::CPU)}));
  // 断言输出向量的大小为 1
  EXPECT_EQ(1, outputs.size());
  // 断言输出的第一个元素转换为整数为 2
  EXPECT_EQ(2, outputs[0].toInt());
}

std::vector<std::string> kernelWithStringListOutput(std::vector<std::string> input) {
  // 返回输入字符串向量本身作为输出
  return input;
}
TEST(OperatorRegistrationTestLegacyFunctionBasedKernel, givenKernelWithStringListOutput_whenRegistered_thenCanBeCalled) {
  // 注册操作符，将函数 kernelWithStringListOutput 注册为处理 "_test::stringlist_output" 操作的实现
  auto registrar = RegisterOperators()
      .op("_test::stringlist_output(str[] input) -> str[]", &kernelWithStringListOutput);

  // 查找并验证操作符是否成功注册
  auto op = c10::Dispatcher::singleton().findSchema({"_test::stringlist_output", ""});
  ASSERT_TRUE(op.has_value());

  // 创建字符串列表并调用操作符
  c10::List<std::string> list({"value1", "value2"});
  auto outputs = callOp(*op, list);

  // 验证操作的输出结果
  EXPECT_EQ(1, outputs.size());
  auto output = std::move(outputs[0]).toList();
  EXPECT_EQ(2, output.size());
  EXPECT_EQ("value1", output.get(0).toStringRef());
  EXPECT_EQ("value2", output.get(1).toStringRef());
}

int captured_dict_size = 0;

void kernelWithDictInputWithoutOutput(Dict<string, Tensor> input1) {
  // 将输入字典的大小捕获到 captured_dict_size 变量中
  captured_dict_size = input1.size();
}

TEST(OperatorRegistrationTestLegacyFunctionBasedKernel, givenKernelWithDictInput_withoutOutput_whenRegistered_thenCanBeCalled) {
  // 注册操作符，将函数 kernelWithDictInputWithoutOutput 注册为处理 "_test::dict_input" 操作的实现
  auto registrar = RegisterOperators()
      .op("_test::dict_input(Dict(str, Tensor) input) -> ()", &kernelWithDictInputWithoutOutput);

  // 查找并验证操作符是否成功注册
  auto op = c10::Dispatcher::singleton().findSchema({"_test::dict_input", ""});
  ASSERT_TRUE(op.has_value());

  // 准备输入数据并调用操作符
  captured_dict_size = 0;
  Dict<string, Tensor> dict;
  dict.insert("key1", dummyTensor(DispatchKey::CPU));
  dict.insert("key2", dummyTensor(DispatchKey::CUDA));
  auto outputs = callOp(*op, dict);

  // 验证操作的输出结果
  EXPECT_EQ(0, outputs.size());
  EXPECT_EQ(2, captured_dict_size);
}

string kernelWithDictInputWithOutput(Dict<string, string> input1) {
  // 返回输入字典中 "key2" 对应的值
  return input1.at("key2");
}

TEST(OperatorRegistrationTestLegacyFunctionBasedKernel, givenKernelWithDictInput_withOutput_whenRegistered_thenCanBeCalled) {
  // 注册操作符，将函数 kernelWithDictInputWithOutput 注册为处理 "_test::dict_input" 操作的实现
  auto registrar = RegisterOperators()
      .op("_test::dict_input(Dict(str, str) input) -> str", &kernelWithDictInputWithOutput);

  // 查找并验证操作符是否成功注册
  auto op = c10::Dispatcher::singleton().findSchema({"_test::dict_input", ""});
  ASSERT_TRUE(op.has_value());

  // 准备输入数据并调用操作符
  Dict<string, string> dict;
  dict.insert("key1", "value1");
  dict.insert("key2", "value2");
  auto outputs = callOp(*op, dict);

  // 验证操作的输出结果
  EXPECT_EQ(1, outputs.size());
  EXPECT_EQ("value2", outputs[0].toStringRef());
}

Dict<string, string> kernelWithDictOutput(Dict<string, string> input) {
  // 返回输入字典本身作为输出
  return input;
}

TEST(OperatorRegistrationTestLegacyFunctionBasedKernel, givenKernelWithDictOutput_whenRegistered_thenCanBeCalled) {
  // 注册操作符，将函数 kernelWithDictOutput 注册为处理 "_test::dict_output" 操作的实现
  auto registrar = RegisterOperators()
      .op("_test::dict_output(Dict(str, str) input) -> Dict(str, str)", &kernelWithDictOutput);

  // 查找并验证操作符是否成功注册
  auto op = c10::Dispatcher::singleton().findSchema({"_test::dict_output", ""});
  ASSERT_TRUE(op.has_value());

  // 准备输入数据并调用操作符
  Dict<string, string> dict;
  dict.insert("key1", "value1");
  dict.insert("key2", "value2");
  auto outputs = callOp(*op, dict);

  // 验证操作的输出结果
  EXPECT_EQ(1, outputs.size());
  auto output = c10::impl::toTypedDict<string, string>(outputs[0].toGenericDict());
  EXPECT_EQ(2, output.size());
  EXPECT_EQ("value1", output.at("key1"));
  EXPECT_EQ("value2", output.at("key2"));
}
void kernelWithUnorderedMapInputWithoutOutput(std::unordered_map<string, Tensor> input1) {
  // 记录输入的无序映射的大小
  captured_dict_size = input1.size();
}

TEST(OperatorRegistrationTestLegacyFunctionBasedKernel, givenKernelWithUnorderedMapInput_withoutOutput_whenRegistered_thenCanBeCalled) {
  // 注册操作符 "_test::dict_input(Dict(str, Tensor) input) -> ()"，关联到 kernelWithUnorderedMapInputWithoutOutput 函数
  auto registrar = RegisterOperators()
      .op("_test::dict_input(Dict(str, Tensor) input) -> ()", &kernelWithUnorderedMapInputWithoutOutput);

  // 查找已注册的操作符，确保注册成功
  auto op = c10::Dispatcher::singleton().findSchema({"_test::dict_input", ""});
  ASSERT_TRUE(op.has_value());

  // 初始化 captured_dict_size 为 0
  captured_dict_size = 0;
  // 创建一个 c10 字典，插入两个键值对
  c10::Dict<string, Tensor> dict;
  dict.insert("key1", dummyTensor(DispatchKey::CPU));
  dict.insert("key2", dummyTensor(DispatchKey::CUDA));
  // 调用操作符，并获取输出
  auto outputs = callOp(*op, dict);
  // 断言输出的大小为 0
  EXPECT_EQ(0, outputs.size());
  // 断言 captured_dict_size 的值为 2
  EXPECT_EQ(2, captured_dict_size);
}

string kernelWithUnorderedMapInputWithOutput(std::unordered_map<string, string> input1) {
  // 返回键为 "key2" 的值
  return input1.at("key2");
}

TEST(OperatorRegistrationTestLegacyFunctionBasedKernel, givenKernelWithUnorderedMapInput_withOutput_whenRegistered_thenCanBeCalled) {
  // 注册操作符 "_test::dict_input(Dict(str, str) input) -> str"，关联到 kernelWithUnorderedMapInputWithOutput 函数
  auto registrar = RegisterOperators()
      .op("_test::dict_input(Dict(str, str) input) -> str", &kernelWithUnorderedMapInputWithOutput);

  // 查找已注册的操作符，确保注册成功
  auto op = c10::Dispatcher::singleton().findSchema({"_test::dict_input", ""});
  ASSERT_TRUE(op.has_value());

  // 创建一个 c10 字典，插入两个键值对
  c10::Dict<string, string> dict;
  dict.insert("key1", "value1");
  dict.insert("key2", "value2");
  // 调用操作符，并获取输出
  auto outputs = callOp(*op, dict);
  // 断言输出的大小为 1
  EXPECT_EQ(1, outputs.size());
  // 断言输出的值为 "value2"
  EXPECT_EQ("value2", outputs[0].toStringRef());
}

std::unordered_map<string, string> kernelWithUnorderedMapOutput(std::unordered_map<string, string> input) {
  // 返回输入的无序映射
  return input;
}

TEST(OperatorRegistrationTestLegacyFunctionBasedKernel, givenKernelWithUnorderedMapOutput_whenRegistered_thenCanBeCalled) {
  // 注册操作符 "_test::dict_output(Dict(str, str) input) -> Dict(str, str)"，关联到 kernelWithUnorderedMapOutput 函数
  auto registrar = RegisterOperators()
      .op("_test::dict_output(Dict(str, str) input) -> Dict(str, str)", &kernelWithUnorderedMapOutput);

  // 查找已注册的操作符，确保注册成功
  auto op = c10::Dispatcher::singleton().findSchema({"_test::dict_output", ""});
  ASSERT_TRUE(op.has_value());

  // 创建一个 c10 字典，插入两个键值对
  c10::Dict<string, string> dict;
  dict.insert("key1", "value1");
  dict.insert("key2", "value2");
  // 调用操作符，并获取输出
  auto outputs = callOp(*op, dict);
  // 断言输出的大小为 1
  EXPECT_EQ(1, outputs.size());
  // 将输出转换为 c10::impl::GenericDict，并断言其大小为 2
  auto output = c10::impl::toTypedDict<string, string>(outputs[0].toGenericDict());
  EXPECT_EQ(2, output.size());
  // 断言输出中的键值对应
  EXPECT_EQ("value1", output.at("key1"));
  EXPECT_EQ("value2", output.at("key2"));
}

std::unordered_map<string, std::vector<int64_t>> kernelWithMapOfIntList(std::unordered_map<string, std::vector<int64_t>> input) {
  // 返回输入的无序映射
  return input;
}
// 定义一个名为 kernelWithListOfMap 的函数，接受一个 unordered_map 的 vector 输入，返回同样类型的输入
std::vector<std::unordered_map<string, int64_t>> kernelWithListOfMap(std::vector<std::unordered_map<string, int64_t>> input) {
  return input;
}
// 定义测试函数 OperatorRegistrationTestLegacyFunctionBasedKernel，测试 Legacy 格式的函数式内核注册情况，具体用例为处理包含列表映射的内核是否成功注册并可调用
TEST(OperatorRegistrationTestLegacyFunctionBasedKernel, givenKernelWithListOfMap_whenRegistered_thenCanBeCalled) {
  // 创建 RegisterOperators 对象并注册名为 _test::list_output 的操作，该操作接受 Dict(str, int)[] 类型的输入并返回同类型的输出，使用函数指针 &kernelWithListOfMap
  auto registrar = RegisterOperators()
      .op("_test::list_output(Dict(str, int)[] input) -> Dict(str, int)[]", &kernelWithListOfMap);

  // 获取名为 _test::list_output 的操作的 schema 对象
  auto op = c10::Dispatcher::singleton().findSchema({"_test::list_output", ""});
  // 断言获取的 schema 对象存在
  ASSERT_TRUE(op.has_value());

  // 创建两个 Dict 对象 dict1 和 dict2，分别插入键值对
  c10::Dict<string, int64_t> dict1;
  dict1.insert("1", 1);
  dict1.insert("2", 2);
  c10::Dict<string, int64_t> dict2;
  dict2.insert("3", 3);
  dict2.insert("4", 4);
  // 创建 List 对象 list，包含 dict1 和 dict2
  c10::List<c10::Dict<string, int64_t>> list({dict1, dict2});
  // 调用操作 op，并传入 list 作为参数，获取返回结果 outputs
  auto outputs = callOp(*op, list);
  // 断言 outputs 的大小为 1
  EXPECT_EQ(1, outputs.size());
  // 将 outputs[0] 转换为 GenericList 对象 output
  c10::impl::GenericList output = std::move(outputs[0]).toList();

  // 断言 output 的大小为 2
  EXPECT_EQ(2, output.size());
  // 断言 output 的第一个元素（类型为 GenericDict）的大小为 2
  EXPECT_EQ(2, output.get(0).toGenericDict().size());
  // 断言 output 的第一个元素中键 "1" 的值为 1
  EXPECT_EQ(1, output.get(0).toGenericDict().at("1").toInt());
  // 断言 output 的第一个元素中键 "2" 的值为 2
  EXPECT_EQ(2, output.get(0).toGenericDict().at("2").toInt());
  // 断言 output 的第二个元素（类型为 GenericDict）的大小为 2
  EXPECT_EQ(2, output.get(1).toGenericDict().size());
  // 断言 output 的第二个元素中键 "3" 的值为 3
  EXPECT_EQ(3, output.get(1).toGenericDict().at("3").toInt());
  // 断言 output 的第二个元素中键 "4" 的值为 4
  EXPECT_EQ(4, output.get(1).toGenericDict().at("4").toInt());
}

// 定义函数 kernelWithListOfMapOfIntList，接受 std::vector<std::unordered_map<string, std::vector<int64_t>>> 类型的输入并返回相同类型的输出
std::vector<std::unordered_map<string, std::vector<int64_t>>> kernelWithListOfMapOfIntList(std::vector<std::unordered_map<string, std::vector<int64_t>>> input) {
  // 直接返回输入的 input
  return input;
}
TEST(OperatorRegistrationTestLegacyFunctionBasedKernel, givenKernelWithListOfMapOfIntList_whenRegistered_thenCanBeCalled) {
  // 创建一个操作符注册器，并注册一个操作符，接受一个字典列表作为输入，返回一个相同结构的字典列表
  auto registrar = RegisterOperators()
      .op("_test::list_output(Dict(str, int[])[] input) -> Dict(str, int[])[]", &kernelWithListOfMapOfIntList);

  // 查找注册的操作符的架构信息
  auto op = c10::Dispatcher::singleton().findSchema({"_test::list_output", ""});
  ASSERT_TRUE(op.has_value());

  // 创建两个字典，并插入键值对，每个字典包含两个键，每个键关联一个整数列表
  c10::Dict<string, c10::List<int64_t>> dict1;
  dict1.insert("1", c10::List<int64_t>({1, 2}));
  dict1.insert("3", c10::List<int64_t>({3, 4}));
  c10::Dict<string, c10::List<int64_t>> dict2;
  dict2.insert("5", c10::List<int64_t>({5, 6}));
  dict2.insert("7", c10::List<int64_t>({7, 8}));

  // 创建一个列表，包含上述两个字典
  c10::List<c10::Dict<string, c10::List<int64_t>>> list({ dict1, dict2 });

  // 调用操作符，并传入列表作为参数，获取输出结果
  auto outputs = callOp(*op, list);

  // 断言输出结果的大小为1
  EXPECT_EQ(1, outputs.size());

  // 将输出结果的第一个元素转换为泛型列表
  c10::impl::GenericList output = std::move(outputs[0]).toList();

  // 断言泛型列表的大小为2
  EXPECT_EQ(2, output.size());

  // 断言第一个字典的大小为2，并检查每个键对应的整数向量的大小和值
  EXPECT_EQ(2, output.get(0).toGenericDict().size());
  EXPECT_EQ(2, output.get(0).toGenericDict().at("1").toIntVector().size());
  EXPECT_EQ(1, output.get(0).toGenericDict().at("1").toIntVector()[0]);
  EXPECT_EQ(2, output.get(0).toGenericDict().at("1").toIntVector()[1]);
  EXPECT_EQ(2, output.get(0).toGenericDict().at("3").toIntVector().size());
  EXPECT_EQ(3, output.get(0).toGenericDict().at("3").toIntVector()[0]);
  EXPECT_EQ(4, output.get(0).toGenericDict().at("3").toIntVector()[1]);

  // 断言第二个字典的大小为2，并检查每个键对应的整数向量的大小和值
  EXPECT_EQ(2, output.get(1).toGenericDict().at("5").toIntVector().size());
  EXPECT_EQ(5, output.get(1).toGenericDict().at("5").toIntVector()[0]);
  EXPECT_EQ(6, output.get(1).toGenericDict().at("5").toIntVector()[1]);
  EXPECT_EQ(2, output.get(1).toGenericDict().at("7").toIntVector().size());
  EXPECT_EQ(7, output.get(1).toGenericDict().at("7").toIntVector()[0]);
  EXPECT_EQ(8, output.get(1).toGenericDict().at("7").toIntVector()[1]);
}

bool called = false;

// 定义一个无输入参数的内核函数，设置一个标志位为true
void kernelWithoutInputs() {
  called = true;
}

TEST(OperatorRegistrationTestLegacyFunctionBasedKernel, givenFallbackKernelWithoutAnyArguments_whenRegistered_thenCanBeCalled) {
  // 注意：没有张量参数的非回退内核不起作用，因为没有办法获取调度键。
  // 对于只有回退内核的操作符，这种方式必须适用于向后兼容性。
  auto registrar = RegisterOperators()
      .op("_test::no_tensor_args() -> ()", &kernelWithoutInputs);

  // 查找注册的操作符的架构信息
  auto op = c10::Dispatcher::singleton().findSchema({"_test::no_tensor_args", ""});
  ASSERT_TRUE(op.has_value());

  // 将标志位重新设置为false，调用操作符，并断言标志位为true
  called = false;
  auto outputs = callOp(*op);
  EXPECT_TRUE(called);
}

// 定义一个带有整数参数的内核函数，返回参数加1的结果
int64_t kernelWithoutTensorInputs(int64_t arg) {
  return arg + 1;
}
// 在单元测试中定义一个测试用例，测试无需张量参数的后备内核注册和调用情况
TEST(OperatorRegistrationTestLegacyFunctionBasedKernel, givenFallbackKernelWithoutTensorArguments_whenRegistered_thenCanBeCalled) {
  // 注意：没有张量参数的非后备内核无法正常工作，因为无法获取调度键。
  // 对于只有后备内核的运算符，这必须保证向后兼容。
  auto registrar = RegisterOperators()
      .op("_test::no_tensor_args(int arg) -> int", &kernelWithoutTensorInputs);

  // 查找操作模式的架构，根据操作符名称和空字符串（无特定版本要求）
  auto op = c10::Dispatcher::singleton().findSchema({"_test::no_tensor_args", ""});
  ASSERT_TRUE(op.has_value());

  // 调用操作并获取输出结果
  auto outputs = callOp(*op, 3);
  // 断言输出的大小为1
  EXPECT_EQ(1, outputs.size());
  // 断言输出结果为4
  EXPECT_EQ(4, outputs[0].toInt());
}

// 声明可选类型变量，用于存储调用时的参数值
std::optional<Tensor> called_arg2 = c10::nullopt;
std::optional<int64_t> called_arg3 = c10::nullopt;
std::optional<std::string> called_arg4 = c10::nullopt;

// 定义一个具有可选输入参数但无输出的内核函数
void kernelWithOptInputWithoutOutput(Tensor arg1, const std::optional<Tensor>& arg2, std::optional<int64_t> arg3, std::optional<std::string> arg4) {
  // 设置调用标志为true
  called = true;
  // 存储传入的可选参数值到全局变量
  called_arg2 = arg2;
  called_arg3 = arg3;
  called_arg4 = arg4;
}

// 单元测试，测试具有可选输入但无输出的内核注册和调用情况
TEST(OperatorRegistrationTestLegacyFunctionBasedKernel, givenKernelWithOptionalInputs_withoutOutput_whenRegistered_thenCanBeCalled) {
  // 注册操作符，指定参数和对应的内核函数
  auto registrar = RegisterOperators().op("_test::opt_input(Tensor arg1, Tensor? arg2, int? arg3, str? arg4) -> ()", &kernelWithOptInputWithoutOutput);
  // 查找操作模式的架构，根据操作符名称和空字符串（无特定版本要求）
  auto op = c10::Dispatcher::singleton().findSchema({"_test::opt_input", ""});
  ASSERT_TRUE(op.has_value());

  // 重置调用标志
  called = false;
  // 调用操作并获取输出结果，传入不同的参数组合
  auto outputs = callOp(*op, dummyTensor(DispatchKey::CPU), dummyTensor(DispatchKey::CUDA), c10::IValue(), std::string("text"));
  // 断言输出的大小为0
  EXPECT_EQ(0, outputs.size());

  // 验证调用标志为true
  EXPECT_TRUE(called);
  // 验证可选参数arg2的值已设置
  EXPECT_TRUE(called_arg2.has_value());
  // 验证arg2的调度键为CUDA
  EXPECT_EQ(extractDispatchKey(*called_arg2), DispatchKey::CUDA);
  // 验证可选参数arg3的值未设置
  EXPECT_FALSE(called_arg3.has_value());
  // 验证可选参数arg4的值已设置
  EXPECT_TRUE(called_arg4.has_value());
  // 验证arg4的值为"text"
  EXPECT_EQ(*called_arg4, "text");

  // 重置调用标志
  called = false;
  // 再次调用操作，传入不同的参数组合
  outputs = callOp(*op, dummyTensor(DispatchKey::CPU), c10::IValue(), 4, c10::IValue());
  // 断言输出的大小为0
  EXPECT_EQ(0, outputs.size());

  // 验证调用标志为true
  EXPECT_TRUE(called);
  // 验证可选参数arg2的值未设置
  EXPECT_FALSE(called_arg2.has_value());
  // 验证可选参数arg3的值已设置
  EXPECT_TRUE(called_arg3.has_value());
  // 验证arg3的值为4
  EXPECT_EQ(*called_arg3, 4);
  // 验证可选参数arg4的值未设置
  EXPECT_FALSE(called_arg4.has_value());
}

// 定义一个具有可选输入和输出的内核函数
std::optional<Tensor> kernelWithOptInputWithOutput(Tensor arg1, const std::optional<Tensor>& arg2, std::optional<int64_t> arg3, std::optional<std::string> arg4) {
  // 设置调用标志为true
  called = true;
  // 存储传入的可选参数值到全局变量
  called_arg2 = arg2;
  called_arg3 = arg3;
  called_arg4 = arg4;
  // 返回arg2的值
  return arg2;
}
// 定义测试函数 OperatorRegistrationTestLegacyFunctionBasedKernel，测试带有可选输入和输出的内核注册功能
TEST(OperatorRegistrationTestLegacyFunctionBasedKernel, givenKernelWithOptionalInputs_withOutput_whenRegistered_thenCanBeCalled) {
  // 注册操作符 "_test::opt_input(Tensor arg1, Tensor? arg2, int? arg3, str? arg4) -> Tensor?"，绑定到 kernelWithOptInputWithOutput 函数
  auto registrar = RegisterOperators().op("_test::opt_input(Tensor arg1, Tensor? arg2, int? arg3, str? arg4) -> Tensor?", &kernelWithOptInputWithOutput);
  // 查找注册的操作符
  auto op = c10::Dispatcher::singleton().findSchema({"_test::opt_input", ""});
  // 断言找到操作符
  ASSERT_TRUE(op.has_value());

  // 初始化调用状态为未调用
  called = false;
  // 调用操作符，并传入一些虚拟的参数
  auto outputs = callOp(*op, dummyTensor(DispatchKey::CPU), dummyTensor(DispatchKey::CUDA), c10::IValue(), std::string("text"));
  // 断言输出结果的数量为1
  EXPECT_EQ(1, outputs.size());
  // 断言第一个输出的分发键为 CUDA
  EXPECT_EQ(DispatchKey::CUDA, extractDispatchKey(outputs[0].toTensor()));

  // 断言函数被调用过
  EXPECT_TRUE(called);
  // 断言调用时 arg2 的值存在且分发键为 CUDA
  EXPECT_TRUE(called_arg2.has_value());
  EXPECT_EQ(extractDispatchKey(*called_arg2), DispatchKey::CUDA);
  // 断言 arg3 的值不存在
  EXPECT_FALSE(called_arg3.has_value());
  // 断言调用时 arg4 的值存在且为 "text"
  EXPECT_TRUE(called_arg4.has_value());
  EXPECT_EQ(*called_arg4, "text");

  // 重置调用状态
  called = false;
  // 再次调用操作符，传入不同的参数组合
  outputs = callOp(*op, dummyTensor(DispatchKey::CPU), c10::IValue(), 4, c10::IValue());
  // 断言输出结果的数量为1
  EXPECT_EQ(1, outputs.size());
  // 断言第一个输出为 None
  EXPECT_TRUE(outputs[0].isNone());

  // 断言函数被调用过
  EXPECT_TRUE(called);
  // 断言调用时 arg2 的值不存在
  EXPECT_FALSE(called_arg2.has_value());
  // 断言调用时 arg3 的值存在且为 4
  EXPECT_TRUE(called_arg3.has_value());
  EXPECT_EQ(*called_arg3, 4);
  // 断言 arg4 的值不存在
  EXPECT_FALSE(called_arg4.has_value());
}

// 定义内核函数 kernelWithOptInputWithMultipleOutputs，接受多个可选输入并返回多个可选输出
std::tuple<std::optional<Tensor>, std::optional<int64_t>, std::optional<std::string>>
kernelWithOptInputWithMultipleOutputs(Tensor arg1, const std::optional<Tensor>& arg2, std::optional<int64_t> arg3, std::optional<std::string> arg4) {
  // 返回包含所有可选输出的元组
  return std::make_tuple(arg2, arg3, arg4);
}

// 定义测试函数 OperatorRegistrationTestLegacyFunctionBasedKernel，测试带有多个可选输入和输出的内核注册功能
TEST(OperatorRegistrationTestLegacyFunctionBasedKernel, givenKernelWithOptionalInputs_withMultipleOutputs_whenRegistered_thenCanBeCalled) {
  // 注册操作符 "_test::opt_input(Tensor arg1, Tensor? arg2, int? arg3, str? arg4) -> (Tensor?, int?, str?)"，绑定到 kernelWithOptInputWithMultipleOutputs 函数
  auto registrar = RegisterOperators().op("_test::opt_input(Tensor arg1, Tensor? arg2, int? arg3, str? arg4) -> (Tensor?, int?, str?)", &kernelWithOptInputWithMultipleOutputs);
  // 查找注册的操作符
  auto op = c10::Dispatcher::singleton().findSchema({"_test::opt_input", ""});
  // 断言找到操作符
  ASSERT_TRUE(op.has_value());

  // 调用操作符，并传入一些虚拟的参数
  auto outputs = callOp(*op, dummyTensor(DispatchKey::CPU), dummyTensor(DispatchKey::CUDA), c10::IValue(), std::string("text"));
  // 断言输出结果的数量为3
  EXPECT_EQ(3, outputs.size());
  // 断言第一个输出的分发键为 CUDA
  EXPECT_EQ(DispatchKey::CUDA, extractDispatchKey(outputs[0].toTensor()));
  // 断言第二个输出为 None
  EXPECT_TRUE(outputs[1].isNone());
  // 断言第三个输出为 "text"
  EXPECT_EQ("text", outputs[2].toStringRef());

  // 再次调用操作符，传入不同的参数组合
  outputs = callOp(*op, dummyTensor(DispatchKey::CPU), c10::IValue(), 4, c10::IValue());
  // 断言输出结果的数量为3
  EXPECT_EQ(3, outputs.size());
  // 断言第一个输出为 None
  EXPECT_TRUE(outputs[0].isNone());
  // 断言第二个输出为 4
  EXPECT_EQ(4, outputs[1].toInt());
  // 断言第三个输出为 None
  EXPECT_TRUE(outputs[2].isNone());
}

// 定义字符串拼接函数 concatKernel，接受一个 Tensor 和多个字符串参数，并返回拼接后的字符串
std::string concatKernel(const Tensor& tensor1, std::string a, const std::string& b, int64_t c) {
  // 返回拼接后的字符串
  return a + b + std::to_string(c);
}
void expectCallsConcatUnboxed(DispatchKey dispatch_key) {
  // 进入自动分派模式，设置对象mode，退出作用域时自动恢复之前的分派模式
  at::AutoDispatchBelowAutograd mode;

  // 断言查找到指定的操作模式和CPU内核
  auto op = c10::Dispatcher::singleton().findSchema({"_test::my_op", ""});
  ASSERT_TRUE(op.has_value());
  
  // 调用未装箱的操作函数，并获取返回的字符串结果
  std::string result = callOpUnboxed<std::string, const Tensor&, std::string, const std::string&, int64_t>(*op, dummyTensor(dispatch_key), "1", "2", 3);
  
  // 断言返回的结果是否为预期的字符串"123"
  EXPECT_EQ("123", result);
}

TEST(OperatorRegistrationTestLegacyFunctionBasedKernel, givenKernel_whenRegistered_thenCanBeCalledUnboxed) {
  // 注册操作"_test::my_op"并关联其CPU内核，使用注册对象registrar来管理操作
  auto registrar = RegisterOperators().op("_test::my_op(Tensor dummy, str a, str b, int c) -> str", &concatKernel);
  // 调用函数expectCallsConcatUnboxed来验证操作能否成功调用
  expectCallsConcatUnboxed(DispatchKey::CPU);
}

std::tuple<int64_t, Tensor> kernelForSchemaInference(Tensor arg1, int64_t arg2, const std::vector<Tensor>& arg3) {
  // 返回一个空元组，用于模式推断的内核函数
  return {};
}

TEST(OperatorRegistrationTestLegacyFunctionBasedKernel, givenKernel_whenRegisteredWithoutSpecifyingSchema_thenInfersSchema) {
  // 注册操作"_test::no_schema_specified"并关联其推断模式的内核函数，使用注册对象registrar来管理操作
  auto registrar = RegisterOperators()
      .op("_test::no_schema_specified", &kernelForSchemaInference);

  // 查找操作"_test::no_schema_specified"的模式，并断言找到了正确的模式
  auto op = c10::Dispatcher::singleton().findSchema({"_test::no_schema_specified", ""});
  ASSERT_TRUE(op.has_value());

  // 检查解析的模式与操作返回的模式是否相匹配，没有找到差异
  std::optional<std::string> differences = c10::findSchemaDifferences(torch::jit::parseSchema("_test::no_schema_specified(Tensor arg1, int arg2, Tensor[] arg3) -> (int, Tensor)"), op->schema());
  EXPECT_FALSE(differences.has_value());
}

template<class Return, class... Args> struct kernel_func final {
  // 泛型模板：返回类型为Return，参数类型为Args...的内核函数
  static Return func(Args...) { return {}; }
};
template<class... Args> struct kernel_func<void, Args...> final {
  // 特化泛型模板：无返回类型，参数类型为Args...的内核函数
  static void func(Args...) {}
}

TEST(OperatorRegistrationTestLegacyFunctionBasedKernel, givenMismatchedKernel_withDifferentNumArguments_whenRegistering_thenFails) {
  // 断言这个注册不会失败，因为参数匹配
  RegisterOperators()
      .op("_test::mismatch(Tensor arg) -> int", &kernel_func<int64_t, Tensor>::func);

  // 现在是一组不匹配的模式
  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg, Tensor arg2) -> int", &kernel_func<int64_t, Tensor>::func);
    }, "The number of arguments is different. 2 vs 1"
  );

  // 断言这个注册不会失败，因为参数匹配
  RegisterOperators()
      .op("_test::mismatch(Tensor arg, Tensor arg2) -> ()", &kernel_func<void, Tensor, Tensor>::func);

  // 现在是一组不匹配的模式
  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch() -> ()", &kernel_func<void, Tensor, Tensor>::func);
    }, "The number of arguments is different. 0 vs 2"
  );

  // 现在是一组不匹配的模式
  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> ()", &kernel_func<void, Tensor, Tensor>::func);
    }, "The number of arguments is different. 1 vs 2"
  );

  // 现在是一组不匹配的模式
  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg, Tensor arg2, Tensor arg3) -> ()", &kernel_func<void, Tensor, Tensor>::func);
    }, "The number of arguments is different. 3 vs 2"
  );
}
    }, "The number of arguments is different. 3 vs 2"
  );


# 这行代码似乎是 JavaScript 或类似语言中的语句，作为一个孤立的代码片段，它看起来是一个对象的闭合和一个字符串的键值对，其中包含一个解释性错误消息。
TEST(OperatorRegistrationTestLegacyFunctionBasedKernel, givenMismatchedKernel_withDifferentArgumentType_whenRegistering_thenFails) {
  // 注册操作符，匹配成功的情况
  RegisterOperators()
      .op("_test::mismatch(Tensor arg1, int arg2) -> int", &kernel_func<int64_t, Tensor, int64_t>::func);

  // 现在是一组不匹配的模式
  expectThrows<c10::Error>([] {
    // 尝试注册操作符，参数类型不匹配的情况
    RegisterOperators()
        .op("_test::mismatch(Tensor arg1, float arg2) -> int", &kernel_func<int64_t, Tensor, int64_t>::func);
    }, "Type mismatch in argument 2: float vs int"
  );

  expectThrows<c10::Error>([] {
    // 尝试注册操作符，参数类型不匹配的情况
    RegisterOperators()
        .op("_test::mismatch(int arg1, int arg2) -> int", &kernel_func<int64_t, Tensor, int64_t>::func);
    }, "Type mismatch in argument 1: int vs Tensor"
  );
}

TEST(OperatorRegistrationTestLegacyFunctionBasedKernel, givenMismatchedKernel_withDifferentNumReturns_whenRegistering_thenFails) {
  // 注册操作符，匹配成功的情况
  RegisterOperators()
      .op("_test::mismatch(Tensor arg) -> int", &kernel_func<int64_t, Tensor>::func);

  // 现在是一组不匹配的模式
  expectThrows<c10::Error>([] {
    // 尝试注册操作符，返回值数量不匹配的情况
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> ()", &kernel_func<int64_t, Tensor>::func);
    }, "The number of returns is different. 0 vs 1"
  );

  expectThrows<c10::Error>([] {
    // 尝试注册操作符，返回值数量不匹配的情况
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> (int, int)", &kernel_func<int64_t, Tensor>::func);
    }, "The number of returns is different. 2 vs 1"
  );

  // 注册操作符，匹配成功的情况
  RegisterOperators()
      .op("_test::mismatch(Tensor arg) -> ()", &kernel_func<void, Tensor>::func);

  // 现在是一组不匹配的模式
  expectThrows<c10::Error>([] {
    // 尝试注册操作符，返回值数量不匹配的情况
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> Tensor", &kernel_func<void, Tensor>::func);
    }, "The number of returns is different. 1 vs 0"
  );

  expectThrows<c10::Error>([] {
    // 尝试注册操作符，返回值数量不匹配的情况
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> (Tensor, Tensor)", &kernel_func<void, Tensor>::func);
    }, "The number of returns is different. 2 vs 0"
  );

  // 注册操作符，匹配成功的情况
  RegisterOperators()
      .op("_test::mismatch(Tensor arg) -> (Tensor, Tensor)", &kernel_func<std::tuple<Tensor, Tensor>, Tensor>::func);

  // 现在是一组不匹配的模式
  expectThrows<c10::Error>([] {
    // 尝试注册操作符，返回值数量不匹配的情况
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> ()", &kernel_func<std::tuple<Tensor, Tensor>, Tensor>::func);
    }, "The number of returns is different. 0 vs 2"
  );

  expectThrows<c10::Error>([] {
    // 尝试注册操作符，返回值数量不匹配的情况
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> Tensor", &kernel_func<std::tuple<Tensor, Tensor>, Tensor>::func);
    }, "The number of returns is different. 1 vs 2"
  );

  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> (Tensor, Tensor, Tensor)", &kernel_func<std::tuple<Tensor, Tensor>, Tensor>::func);
    }, "The number of returns is different. 3 vs 2"
  );



    // 注册自定义运算符 "_test::mismatch"，指定输入参数为 Tensor 类型，返回类型为 (Tensor, Tensor, Tensor)
    // 并将其映射到相应的 C++ 函数 kernel_func<std::tuple<Tensor, Tensor>, Tensor>::func
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> (Tensor, Tensor, Tensor)", &kernel_func<std::tuple<Tensor, Tensor>, Tensor>::func);
    // 注册操作的描述信息，指出此运算符返回值个数不同，期望为 3 个返回值，实际为 2 个返回值
    }, "The number of returns is different. 3 vs 2"
  );


这段代码是在注册一个自定义运算符到某个框架或者系统中，具体是注册了一个名为 "_test::mismatch" 的运算符，定义了它接受一个名为 `arg` 的 Tensor 参数，并且期望返回三个 Tensor 类型的结果。后面的代码片段是提供了一条描述信息，指出注册时遇到的问题，即期望的返回值个数与实际返回的个数不符（期望为3个，实际为2个）。
TEST(OperatorRegistrationTestLegacyFunctionBasedKernel, givenMismatchedKernel_withDifferentReturnTypes_whenRegistering_thenFails) {
  // 定义测试用例，验证当注册操作符时，如果内核返回类型不匹配，应该抛出异常

  // 注册一个操作符，声明返回类型为 int，与实际内核函数的返回类型不匹配，应该不会失败
  RegisterOperators()
      .op("_test::mismatch(Tensor arg) -> int", &kernel_func<int64_t, Tensor>::func);

  // 然后注册一组不匹配的模式
  expectThrows<c10::Error>([] {
    // 注册一个操作符，声明返回类型为 Tensor，与之前注册的内核函数的返回类型 int 不匹配，应该抛出异常
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> Tensor", &kernel_func<int64_t, Tensor>::func);
    }, "Type mismatch in return 1: Tensor vs int"
  );

  // 另外一个不匹配的模式
  expectThrows<c10::Error>([] {
    // 注册一个操作符，声明返回类型为 float，与之前注册的内核函数的返回类型 int 不匹配，应该抛出异常
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> float", &kernel_func<int64_t, Tensor>::func);
    }, "Type mismatch in return 1: float vs int"
  );

  // 验证这个注册操作不会失败，因为匹配成功
  RegisterOperators()
      .op("_test::mismatch(Tensor arg) -> Tensor", &kernel_func<Tensor, Tensor>::func);

  // 另外一个不匹配的模式
  expectThrows<c10::Error>([] {
    // 注册一个操作符，声明返回类型为 float，与之前注册的内核函数的返回类型 Tensor 不匹配，应该抛出异常
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> float", &kernel_func<Tensor, Tensor>::func);
    }, "Type mismatch in return 1: float vs Tensor"
  );

  // 验证这个注册操作不会失败，因为匹配成功
  RegisterOperators()
      .op("_test::mismatch(Tensor arg) -> (Tensor, int)", &kernel_func<std::tuple<Tensor, int64_t>, Tensor>::func);

  // 另外一个不匹配的模式
  expectThrows<c10::Error>([] {
    // 注册一个操作符，声明返回类型为 (Tensor, float)，与之前注册的内核函数的返回类型 (Tensor, int) 不匹配，应该抛出异常
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> (Tensor, float)", &kernel_func<std::tuple<Tensor, int64_t>, Tensor>::func);
    }, "Type mismatch in return 2: float vs int"
  );

  // 另外一个不匹配的模式
  expectThrows<c10::Error>([] {
    // 注册一个操作符，声明返回类型为 (int, int)，与之前注册的内核函数的返回类型 (Tensor, int) 不匹配，应该抛出异常
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> (int, int)", &kernel_func<std::tuple<Tensor, int64_t>, Tensor>::func);
    }, "Type mismatch in return 1: int vs Tensor"
  );
}
```