# `.\pytorch\aten\src\ATen\core\boxing\impl\kernel_lambda_legacy_test.cpp`

```py
#include <gtest/gtest.h>

// 引入测试框架的头文件

// This intentionally tests a deprecated API
// 忽略使用了废弃声明的警告
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

#include <ATen/core/boxing/impl/test_helpers.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/core/Tensor.h>
#include <torch/csrc/jit/frontend/function_schema_parser.h>

#include <ATen/core/LegacyTypeDispatch.h>

/**
 * This file tests the legacy lambda-based API for registering kernels:
 *
 * > auto registry = c10::RegisterOperators()
 * >    .op("myfunc(Tensor a) -> Tensor", [] (Tensor a) -> Tensor {...});
 */

// 使用了一些 ATen 和 Torch 的头文件，用于注册操作和定义 Tensor 等

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

// 定义了一个匿名命名空间，限定了这些函数和变量的作用域

void expectCallsIncrement(DispatchKey dispatch_key) {
  // 在自动分发模式下执行以下代码
  at::AutoDispatchBelowAutograd mode;

  // 断言操作 schema 和 CPU 内核存在
  auto op = c10::Dispatcher::singleton().findSchema({"_test::my_op", ""});
  ASSERT_TRUE(op.has_value());

  // 调用注册的操作，期望返回值增加了1
  auto result = callOp(*op, dummyTensor(dispatch_key), 5);
  EXPECT_EQ(1, result.size());
  EXPECT_EQ(6, result[0].toInt());
}

// 定义了测试函数，测试注册基于 lambda 表达式的内核

TEST(OperatorRegistrationTestLegacyLambdaBasedKernel, givenKernel_whenRegistered_thenCanBeCalled) {
  // 注册一个操作，对输入加1的 lambda 内核
  auto registrar = RegisterOperators().op("_test::my_op(Tensor dummy, int input) -> int", [] (const Tensor& tensor, int64_t input) -> int64_t {
      return input + 1;
    });
  // 期望调用后增加了1
  expectCallsIncrement(DispatchKey::CPU);
}

// 另一个测试函数，测试在构造函数中注册内核

TEST(OperatorRegistrationTestLegacyLambdaBasedKernel, givenKernel_whenRegisteredInConstructor_thenCanBeCalled) {
  // 在构造函数中注册一个操作，对输入加1的 lambda 内核
  auto registrar = RegisterOperators("_test::my_op(Tensor dummy, int input) -> int", [] (const Tensor& tensor, int64_t input) -> int64_t {
      return input + 1;
    });
  // 期望调用后增加了1
  expectCallsIncrement(DispatchKey::CPU);
}

// 测试多个操作和内核同时注册的情况

TEST(OperatorRegistrationTestLegacyLambdaBasedKernel, givenMultipleOperatorsAndKernels_whenRegisteredInOneRegistrar_thenCallsRightKernel) {
  // 在一个注册器中注册多个操作和内核
  auto registrar = RegisterOperators()
      .op("_test::my_op(Tensor dummy, int input) -> int", [] (const Tensor& tensor, int64_t input) -> int64_t {
          return input + 1;
        })
      .op("_test::error(Tensor dummy, int input) -> int", [] (const Tensor& tensor, int64_t input) -> int64_t {
          EXPECT_TRUE(false); // 这个内核不应该被调用
          return 0;
        });
  // 期望调用后增加了1
  expectCallsIncrement(DispatchKey::CPU);
}

// 测试在多个注册器中注册的情况

TEST(OperatorRegistrationTestLegacyLambdaBasedKernel, givenMultipleOperatorsAndKernels_whenRegisteredInMultipleRegistrars_thenCallsRightKernel) {
  // 在第一个注册器中注册一个操作和内核
  auto registrar1 = RegisterOperators().op("_test::my_op(Tensor dummy, int input) -> int", [] (const Tensor& tensor, int64_t input) -> int64_t {
      return input + 1;
    });
  // 在第二个注册器中注册一个操作和内核
  auto registrar2 = RegisterOperators().op("_test::error(Tensor dummy, int input) -> int", [] (const Tensor& tensor, int64_t input) -> int64_t {
      EXPECT_TRUE(false); // 这个内核不应该被调用
      return 0;
    });
    });
    // End of an anonymous function or a block in the code.
    // This semicolon likely terminates a statement or function call.
    // It marks the completion of some code logic or operation.

  expectCallsIncrement(DispatchKey::CPU);
    // Call to the function `expectCallsIncrement` with an argument `DispatchKey::CPU`.
    // This function call presumably increments some counter related to CPU dispatch calls.
    // It triggers an expectation or validation related to CPU dispatch behavior.
}

// 定义一个测试用例 OperatorRegistrationTestLegacyLambdaBasedKernel，验证在注册对象超出作用域后无法再调用
TEST(OperatorRegistrationTestLegacyLambdaBasedKernel, givenKernel_whenRegistrationRunsOutOfScope_thenCannotBeCalledAnymore) {
  {
    // 创建操作符注册器，注册一个名为 "_test::my_op" 的操作符，其实现为 lambda 函数
    auto registrar = RegisterOperators().op("_test::my_op(Tensor dummy, int input) -> int", [] (const Tensor& tensor, int64_t input) -> int64_t {
        return input + 1;
      });

    // 预期在 CPU 上调用 expectCallsIncrement 函数
    expectCallsIncrement(DispatchKey::CPU);
  }

  // 现在注册器已经析构。断言 "_test::my_op" 操作符已经不可用。
  expectDoesntFindOperator("_test::my_op");
}

// 定义一个布尔型变量 was_called，用于记录函数是否被调用过
bool was_called = false;

// 定义测试用例 OperatorRegistrationTestLegacyLambdaBasedKernel，验证无返回值的操作注册
TEST(OperatorRegistrationTestLegacyLambdaBasedKernel, givenKernelWithoutOutput_whenRegistered_thenCanBeCalled) {
  // 创建操作符注册器，注册一个名为 "_test::no_return" 的操作符，其实现为 lambda 函数
  auto registrar = RegisterOperators().op("_test::no_return(Tensor dummy) -> ()", [] (const Tensor&) -> void {
    // 将 was_called 标记为 true
    was_called = true;
  });

  // 查找名为 "_test::no_return" 的操作符架构
  auto op = c10::Dispatcher::singleton().findSchema({"_test::no_return", ""});
  ASSERT_TRUE(op.has_value());

  // 重置 was_called 为 false，并调用操作符，期望 was_called 变为 true
  was_called = false;
  auto result = callOp(*op, dummyTensor(DispatchKey::CPU));
  EXPECT_TRUE(was_called);
  // 期望结果返回空的 std::tuple
  EXPECT_EQ(0, result.size());
}

// 定义测试用例 OperatorRegistrationTestLegacyLambdaBasedKernel，验证返回空的元组的操作注册
TEST(OperatorRegistrationTestLegacyLambdaBasedKernel, givenKernelWithZeroOutputs_whenRegistered_thenCanBeCalled) {
  // 创建操作符注册器，注册一个名为 "_test::zero_outputs" 的操作符，其实现为 lambda 函数
  auto registrar = RegisterOperators().op("_test::zero_outputs(Tensor dummy) -> ()", [] (const Tensor&) -> std::tuple<> {
    // 将 was_called 标记为 true，并返回空的 std::tuple
    was_called = true;
    return std::make_tuple();
  });

  // 查找名为 "_test::zero_outputs" 的操作符架构
  auto op = c10::Dispatcher::singleton().findSchema({"_test::zero_outputs", ""});
  ASSERT_TRUE(op.has_value());

  // 重置 was_called 为 false，并调用操作符，期望 was_called 变为 true
  was_called = false;
  auto result = callOp(*op, dummyTensor(DispatchKey::CPU));
  EXPECT_TRUE(was_called);
  // 期望结果返回空的 std::tuple
  EXPECT_EQ(0, result.size());
}

// 定义测试用例 OperatorRegistrationTestLegacyLambdaBasedKernel，验证返回整数的操作注册
TEST(OperatorRegistrationTestLegacyLambdaBasedKernel, givenKernelWithIntOutput_whenRegistered_thenCanBeCalled) {
  // 创建操作符注册器，注册一个名为 "_test::int_output" 的操作符，其实现为 lambda 函数
  auto registrar = RegisterOperators()
      .op("_test::int_output(Tensor dummy, int a, int b) -> int", [] (Tensor, int64_t a, int64_t b) -> int64_t {
        // 返回 a + b 的结果
        return a + b;
      });

  // 查找名为 "_test::int_output" 的操作符架构
  auto op = c10::Dispatcher::singleton().findSchema({"_test::int_output", ""});
  ASSERT_TRUE(op.has_value());

  // 调用操作符，并期望结果返回一个大小为 1 的向量，其值为 9
  auto result = callOp(*op, dummyTensor(DispatchKey::CPU), 3, 6);
  EXPECT_EQ(1, result.size());
  EXPECT_EQ(9, result[0].toInt());
}

// 定义测试用例 OperatorRegistrationTestLegacyLambdaBasedKernel，验证返回张量的操作注册
TEST(OperatorRegistrationTestLegacyLambdaBasedKernel, givenKernelWithTensorOutput_whenRegistered_thenCanBeCalled) {
  // 创建操作符注册器，注册一个名为 "_test::returning_tensor" 的操作符，其实现为 lambda 函数
  auto registrar = RegisterOperators()
      .op("_test::returning_tensor(Tensor input) -> Tensor", [] (const Tensor& input) -> Tensor {
        // 返回输入张量
        return input;
      });

  // 查找名为 "_test::returning_tensor" 的操作符架构
  auto op = c10::Dispatcher::singleton().findSchema({"_test::returning_tensor", ""});
  ASSERT_TRUE(op.has_value());

  // 调用操作符，并期望结果返回一个大小为 1 的向量，其分发键为 CPU
  auto result = callOp(*op, dummyTensor(DispatchKey::CPU));
  EXPECT_EQ(1, result.size());
  EXPECT_EQ(DispatchKey::CPU, extractDispatchKey(result[0].toTensor()));

  // 再次调用操作符，并期望结果返回一个大小为 1 的向量，其分发键为 CUDA
  result = callOp(*op, dummyTensor(DispatchKey::CUDA));
  EXPECT_EQ(1, result.size());
  EXPECT_EQ(DispatchKey::CUDA, extractDispatchKey(result[0].toTensor()));
}
# 定义测试函数 OperatorRegistrationTestLegacyLambdaBasedKernel，测试注册基于 lambda 的遗留内核操作
TEST(OperatorRegistrationTestLegacyLambdaBasedKernel, givenKernelWithTensorListOutput_whenRegistered_thenCanBeCalled) {
  # 注册运算符，定义名称为 _test::list_output 的操作，接受三个 Tensor 输入并返回 Tensor 数组
  auto registrar = RegisterOperators()
      .op("_test::list_output(Tensor input1, Tensor input2, Tensor input3) -> Tensor[]", [] (const Tensor& input1, const Tensor& input2, const Tensor& input3) -> std::vector<Tensor> {
        # 返回包含输入 Tensor 的向量
        return {input1, input2, input3};
      });

  # 查找名称为 _test::list_output 的操作模式
  auto op = c10::Dispatcher::singleton().findSchema({"_test::list_output", ""});
  # 断言操作模式存在
  ASSERT_TRUE(op.has_value());

  # 调用操作 op，并传入虚拟的 CPU 张量和 CUDA 张量作为输入
  auto result = callOp(*op, dummyTensor(DispatchKey::CPU), dummyTensor(DispatchKey::CUDA), dummyTensor(DispatchKey::CPU));
  # 断言结果向量大小为 1
  EXPECT_EQ(1, result.size());
  # 断言结果向量第一个元素转换为 Tensor 向量后大小为 3
  EXPECT_EQ(3, result[0].toTensorVector().size());
  # 断言结果向量第一个元素的第一个 Tensor 的分发键为 CPU
  EXPECT_EQ(DispatchKey::CPU, extractDispatchKey(result[0].toTensorVector()[0]));
  # 断言结果向量第一个元素的第二个 Tensor 的分发键为 CUDA
  EXPECT_EQ(DispatchKey::CUDA, extractDispatchKey(result[0].toTensorVector()[1]));
  # 断言结果向量第一个元素的第三个 Tensor 的分发键为 CPU
  EXPECT_EQ(DispatchKey::CPU, extractDispatchKey(result[0].toTensorVector()[2]));
}

# 定义测试函数 OperatorRegistrationTestLegacyLambdaBasedKernel，测试注册基于 lambda 的遗留内核操作
TEST(OperatorRegistrationTestLegacyLambdaBasedKernel, givenKernelWithIntListOutput_whenRegistered_thenCanBeCalled) {
  # 注册运算符，定义名称为 _test::list_output 的操作，接受一个 Tensor 和三个整数输入，并返回整数数组
  auto registrar = RegisterOperators()
      .op("_test::list_output(Tensor dummy, int input1, int input2, int input3) -> int[]", [](const Tensor&, int64_t input1, int64_t input2, int64_t input3) -> std::vector<int64_t> {
        # 返回包含输入整数的向量
        return {input1, input2, input3};
      });

  # 查找名称为 _test::list_output 的操作模式
  auto op = c10::Dispatcher::singleton().findSchema({"_test::list_output", ""});
  # 断言操作模式存在
  ASSERT_TRUE(op.has_value());

  # 调用操作 op，并传入虚拟的 CPU 张量以及整数 2、4、6 作为输入
  auto result = callOp(*op, dummyTensor(DispatchKey::CPU), 2, 4, 6);
  # 断言结果向量大小为 1
  EXPECT_EQ(1, result.size());
  # 断言结果向量第一个元素转换为整数向量后大小为 3
  EXPECT_EQ(3, result[0].toIntVector().size());
  # 断言结果向量第一个元素的第一个整数为 2
  EXPECT_EQ(2, result[0].toIntVector()[0]);
  # 断言结果向量第一个元素的第二个整数为 4
  EXPECT_EQ(4, result[0].toIntVector()[1]);
  # 断言结果向量第一个元素的第三个整数为 6
  EXPECT_EQ(6, result[0].toIntVector()[2]);
}
TEST(OperatorRegistrationTestLegacyLambdaBasedKernel, givenKernelWithMultipleOutputs_whenRegistered_thenCanBeCalled) {
  // 创建操作注册器，用于注册具有多个输出的自定义运算符
  auto registrar = RegisterOperators()
     // 注册自定义运算符 "_test::multiple_outputs"，接受一个Tensor参数，并返回包含多种类型的数据的元组
     .op("_test::multiple_outputs(Tensor dummy) -> (Tensor, int, Tensor[], int?, Dict(str, Tensor))", [] (Tensor) -> std::tuple<Tensor, int64_t, std::vector<Tensor>, std::optional<int64_t>, Dict<string, Tensor>> {
       // 创建一个字典对象
       Dict<string, Tensor> dict;
       // 在字典中插入名为 "first" 的Tensor，使用CPU分发键
       dict.insert("first", dummyTensor(DispatchKey::CPU));
       // 在字典中插入名为 "second" 的Tensor，使用CUDA分发键
       dict.insert("second", dummyTensor(DispatchKey::CUDA));
       // 返回包含多种数据类型的元组，包括Tensor、int64_t、Tensor向量、可选的int64_t和字典
       return std::tuple<Tensor, int64_t, std::vector<Tensor>, std::optional<int64_t>, Dict<string, Tensor>>(
         dummyTensor(DispatchKey::CUDA),                       // 返回一个CUDA Tensor
         5,                                                    // 返回整数值 5
         {dummyTensor(DispatchKey::CPU), dummyTensor(DispatchKey::CUDA)},  // 返回包含两个Tensor的向量
         std::optional<int64_t>(std::in_place, 0),             // 返回一个包含值 0 的可选int64_t
         dict                                                  // 返回前面创建的字典对象
       );
     });

  // 查找已注册的操作模式，检查是否成功注册
  auto op = c10::Dispatcher::singleton().findSchema({"_test::multiple_outputs", ""});
  ASSERT_TRUE(op.has_value());

  // 调用注册的操作，并检查返回结果
  auto result = callOp(*op, dummyTensor(DispatchKey::CPU));
  EXPECT_EQ(5, result.size());                    // 检查返回结果的大小是否为 5
  EXPECT_EQ(DispatchKey::CUDA, extractDispatchKey(result[0].toTensor()));  // 检查第一个Tensor的分发键是否为CUDA
  EXPECT_EQ(5, result[1].toInt());                // 检查第二个返回值是否为整数 5
  EXPECT_EQ(2, result[2].toTensorVector().size()); // 检查第三个返回值（Tensor向量）的大小是否为 2
  EXPECT_EQ(DispatchKey::CPU, extractDispatchKey(result[2].toTensorVector()[0]));  // 检查第一个Tensor向量元素的分发键是否为CPU
  EXPECT_EQ(DispatchKey::CUDA, extractDispatchKey(result[2].toTensorVector()[1])); // 检查第二个Tensor向量元素的分发键是否为CUDA
  EXPECT_EQ(0, result[3].toInt());                // 检查第四个返回值（可选int64_t）是否为 0
  // 将通用字典转换为特定类型的字典，并检查其大小和内容
  auto result_dict = c10::impl::toTypedDict<string, Tensor>(result[4].toGenericDict());
  EXPECT_EQ(2, result_dict.size());               // 检查字典的大小是否为 2
  EXPECT_EQ(DispatchKey::CPU, extractDispatchKey(result_dict.at("first")));     // 检查字典中名为 "first" 的Tensor的分发键是否为CPU
  EXPECT_EQ(DispatchKey::CUDA, extractDispatchKey(result_dict.at("second")));   // 检查字典中名为 "second" 的Tensor的分发键是否为CUDA
}

TEST(OperatorRegistrationTestLegacyLambdaBasedKernel, givenKernelWithTensorInputByReference_withOutput_whenRegistered_thenCanBeCalled) {
  // 创建操作注册器，用于注册具有引用输入和输出的自定义运算符
  auto registrar = RegisterOperators()
      // 注册自定义运算符 "_test::tensor_input"，接受一个常量引用的Tensor参数，并返回一个Tensor
      .op("_test::tensor_input(Tensor input) -> Tensor", [] (const Tensor& input1) -> Tensor {
        // 直接返回输入的Tensor
        return input1;
      });

  // 查找已注册的操作模式，检查是否成功注册
  auto op = c10::Dispatcher::singleton().findSchema({"_test::tensor_input", ""});
  ASSERT_TRUE(op.has_value());

  // 调用注册的操作，并检查返回结果
  auto result = callOp(*op, dummyTensor(DispatchKey::CPU));
  EXPECT_EQ(1, result.size());                    // 检查返回结果的大小是否为 1
  EXPECT_EQ(DispatchKey::CPU, extractDispatchKey(result[0].toTensor()));  // 检查返回的Tensor的分发键是否为CPU

  // 再次调用注册的操作，以不同的输入，并检查返回结果
  result = callOp(*op, dummyTensor(DispatchKey::CUDA));
  EXPECT_EQ(1, result.size());                    // 检查返回结果的大小是否为 1
  EXPECT_EQ(DispatchKey::CUDA, extractDispatchKey(result[0].toTensor())); // 检查返回的Tensor的分发键是否为CUDA
}
// 测试宏定义，创建一个测试用例 OperatorRegistrationTestLegacyLambdaBasedKernel，测试带有按值输入张量和输出的内核是否注册并可调用
TEST(OperatorRegistrationTestLegacyLambdaBasedKernel, givenKernelWithTensorInputByValue_withOutput_whenRegistered_thenCanBeCalled) {
  // 创建注册器对象 registrar，注册 _test::tensor_input 操作符
  auto registrar = RegisterOperators()
      .op("_test::tensor_input(Tensor input) -> Tensor", [](Tensor input1) -> Tensor {
        // Lambda 表达式，接收 Tensor 输入并直接返回
        return input1;
      });

  // 查找注册的操作符 _test::tensor_input，并断言其存在
  auto op = c10::Dispatcher::singleton().findSchema({"_test::tensor_input", ""});
  ASSERT_TRUE(op.has_value());

  // 调用注册的操作符，并传入一个虚拟的 CPU 张量，期望返回一个结果张量
  auto result = callOp(*op, dummyTensor(DispatchKey::CPU));
  EXPECT_EQ(1, result.size());  // 检查结果中张量的数量为 1
  EXPECT_EQ(DispatchKey::CPU, extractDispatchKey(result[0].toTensor()));  // 检查结果张量的分发键为 CPU

  // 再次调用注册的操作符，传入一个虚拟的 CUDA 张量，期望返回一个结果张量
  result = callOp(*op, dummyTensor(DispatchKey::CUDA));
  EXPECT_EQ(1, result.size());  // 检查结果中张量的数量为 1
  EXPECT_EQ(DispatchKey::CUDA, extractDispatchKey(result[0].toTensor()));  // 检查结果张量的分发键为 CUDA
}

// 全局变量，用于捕获输入的张量
Tensor captured_input;

// 测试宏定义，创建一个测试用例 OperatorRegistrationTestLegacyLambdaBasedKernel，测试带有按引用输入张量但无输出的内核是否注册并可调用
TEST(OperatorRegistrationTestLegacyLambdaBasedKernel, givenKernelWithTensorInputByReference_withoutOutput_whenRegistered_thenCanBeCalled) {
  // 创建注册器对象 registrar，注册 _test::tensor_input 操作符
  auto registrar = RegisterOperators()
      .op("_test::tensor_input(Tensor input) -> ()", [] (const Tensor& input1) -> void {
        // Lambda 表达式，捕获输入张量并存储到全局变量 captured_input
        captured_input = input1;
      });

  // 查找注册的操作符 _test::tensor_input，并断言其存在
  auto op = c10::Dispatcher::singleton().findSchema({"_test::tensor_input", ""});
  ASSERT_TRUE(op.has_value());

  // 调用注册的操作符，并传入一个虚拟的 CPU 张量，期望返回空输出
  auto outputs = callOp(*op, dummyTensor(DispatchKey::CPU));
  EXPECT_EQ(0, outputs.size());  // 检查输出的张量数量为 0
  EXPECT_EQ(DispatchKey::CPU, extractDispatchKey(captured_input));  // 检查捕获的输入张量的分发键为 CPU

  // 再次调用注册的操作符，并传入一个虚拟的 CUDA 张量，期望返回空输出
  outputs = callOp(*op, dummyTensor(DispatchKey::CUDA));
  EXPECT_EQ(0, outputs.size());  // 检查输出的张量数量为 0
  EXPECT_EQ(DispatchKey::CUDA, extractDispatchKey(captured_input));  // 检查捕获的输入张量的分发键为 CUDA
}

// 测试宏定义，创建一个测试用例 OperatorRegistrationTestLegacyLambdaBasedKernel，测试带有按值输入张量但无输出的内核是否注册并可调用
TEST(OperatorRegistrationTestLegacyLambdaBasedKernel, givenKernelWithTensorInputByValue_withoutOutput_whenRegistered_thenCanBeCalled) {
  // 创建注册器对象 registrar，注册 _test::tensor_input 操作符
  auto registrar = RegisterOperators()
      .op("_test::tensor_input(Tensor input) -> ()", [] (Tensor input1) -> void {
        // Lambda 表达式，捕获输入张量并存储到全局变量 captured_input
        captured_input = input1;
      });

  // 查找注册的操作符 _test::tensor_input，并断言其存在
  auto op = c10::Dispatcher::singleton().findSchema({"_test::tensor_input", ""});
  ASSERT_TRUE(op.has_value());

  // 调用注册的操作符，并传入一个虚拟的 CPU 张量，期望返回空输出
  auto outputs = callOp(*op, dummyTensor(DispatchKey::CPU));
  EXPECT_EQ(0, outputs.size());  // 检查输出的张量数量为 0
  EXPECT_EQ(DispatchKey::CPU, extractDispatchKey(captured_input));  // 检查捕获的输入张量的分发键为 CPU

  // 再次调用注册的操作符，并传入一个虚拟的 CUDA 张量，期望返回空输出
  outputs = callOp(*op, dummyTensor(DispatchKey::CUDA));
  EXPECT_EQ(0, outputs.size());  // 检查输出的张量数量为 0
  EXPECT_EQ(DispatchKey::CUDA, extractDispatchKey(captured_input));  // 检查捕获的输入张量的分发键为 CUDA
}

// 全局变量，用于捕获输入的整数
int64_t captured_int_input = 0;

// 测试宏定义，创建一个测试用例 OperatorRegistrationTestLegacyLambdaBasedKernel，测试带有整数输入但无输出的内核是否注册并可调用
TEST(OperatorRegistrationTestLegacyLambdaBasedKernel, givenKernelWithIntInput_withoutOutput_whenRegistered_thenCanBeCalled) {
  // 创建注册器对象 registrar，注册 _test::int_input 操作符
  auto registrar = RegisterOperators()
      .op("_test::int_input(Tensor dummy, int input) -> ()", [](Tensor, int64_t input1) -> void {
        // Lambda 表达式，捕获整数输入并存储到全局变量 captured_int_input
        captured_int_input = input1;
      });

  // 查找注册的操作符 _test::int_input，并断言其存在
  auto op = c10::Dispatcher::singleton().findSchema({"_test::int_input", ""});
  ASSERT_TRUE(op.has_value());

  // 初始化捕获的整数输入为 0
  captured_int_input = 0;
  // 调用注册的操作符，并传入一个虚拟的 CPU 张量和整数输入 3，期望返回空输出
  auto outputs = callOp(*op, dummyTensor(DispatchKey::CPU), 3);
  EXPECT_EQ(0, outputs.size());  // 检查输出的张量数量为 0
  EXPECT_EQ(3, captured_int_input);  // 检查捕获的整数输入为 3
}
TEST(OperatorRegistrationTestLegacyLambdaBasedKernel, givenKernelWithIntInput_withOutput_whenRegistered_thenCanBeCalled) {
  // 创建操作注册器，注册一个接受整数输入并返回整数输出的操作
  auto registrar = RegisterOperators()
      .op("_test::int_input(Tensor dummy, int input) -> int", [] (Tensor, int64_t input1) -> int64_t {
        return input1 + 1;
      });

  // 查找并验证已注册的操作模式
  auto op = c10::Dispatcher::singleton().findSchema({"_test::int_input", ""});
  ASSERT_TRUE(op.has_value());

  // 调用操作，期望返回一个整数结果
  auto outputs = callOp(*op, dummyTensor(DispatchKey::CPU), 3);
  EXPECT_EQ(1, outputs.size());
  EXPECT_EQ(4, outputs[0].toInt());
}

int64_t captured_input_list_size = 0;

TEST(OperatorRegistrationTestLegacyLambdaBasedKernel, givenKernelWithIntListInput_withoutOutput_whenRegistered_thenCanBeCalled) {
  // 创建操作注册器，注册一个接受整数列表输入但没有输出的操作
  auto registrar = RegisterOperators()
      .op("_test::int_list_input(Tensor dummy, int[] input) -> ()", [] (Tensor, const std::vector<int64_t>& input1) -> void {
        captured_input_list_size = input1.size();
      });

  // 查找并验证已注册的操作模式
  auto op = c10::Dispatcher::singleton().findSchema({"_test::int_list_input", ""});
  ASSERT_TRUE(op.has_value());

  // 初始化捕获输入列表大小的变量为0，并调用操作
  captured_input_list_size = 0;
  auto outputs = callOp(*op, dummyTensor(DispatchKey::CPU), c10::List<int64_t>({2, 4, 6}));
  EXPECT_EQ(0, outputs.size());
  EXPECT_EQ(3, captured_input_list_size);
}

TEST(OperatorRegistrationTestLegacyLambdaBasedKernel, givenKernelWithIntListInput_withOutput_whenRegistered_thenCanBeCalled) {
  // 创建操作注册器，注册一个接受整数列表输入并返回整数输出的操作
  auto registrar = RegisterOperators()
      .op("_test::int_list_input(Tensor dummy, int[] input) -> int", [](Tensor, const std::vector<int64_t>& input1) -> int64_t {
        return input1.size();
      });

  // 查找并验证已注册的操作模式
  auto op = c10::Dispatcher::singleton().findSchema({"_test::int_list_input", ""});
  ASSERT_TRUE(op.has_value());

  // 调用操作，期望返回一个整数结果
  auto outputs = callOp(*op, dummyTensor(DispatchKey::CPU), c10::List<int64_t>({2, 4, 6}));
  EXPECT_EQ(1, outputs.size());
  EXPECT_EQ(3, outputs[0].toInt());
}

TEST(OperatorRegistrationTestLegacyLambdaBasedKernel, givenKernelWithTensorListInput_withoutOutput_whenRegistered_thenCanBeCalled) {
  // 创建操作注册器，注册一个接受张量列表输入但没有输出的操作
  auto registrar = RegisterOperators()
      .op("_test::tensor_list_input(Tensor[] input) -> ()", [] (const std::vector<Tensor>& input1) -> void {
        captured_input_list_size = input1.size();
      });

  // 查找并验证已注册的操作模式
  auto op = c10::Dispatcher::singleton().findSchema({"_test::tensor_list_input", ""});
  ASSERT_TRUE(op.has_value());

  // 初始化捕获输入列表大小的变量为0，并调用操作
  captured_input_list_size = 0;
  auto outputs = callOp(*op, c10::List<Tensor>({dummyTensor(DispatchKey::CPU), dummyTensor(DispatchKey::CPU)}));
  EXPECT_EQ(0, outputs.size());
  EXPECT_EQ(2, captured_input_list_size);
}
TEST(OperatorRegistrationTestLegacyLambdaBasedKernel, givenKernelWithTensorVectorInput_withOutput_whenRegistered_thenCanBeCalled) {
  // 创建操作注册器并注册具有张量向量输入和整数输出的操作
  auto registrar = RegisterOperators()
      .op("_test::tensor_list_input(Tensor[] input) -> int", [] (const std::vector<Tensor>& input1) -> int64_t {
        return input1.size();
      });

  // 查找已注册操作的架构
  auto op = c10::Dispatcher::singleton().findSchema({"_test::tensor_list_input", ""});
  // 断言确保操作架构存在
  ASSERT_TRUE(op.has_value());

  // 调用操作并获取输出
  auto outputs = callOp(*op, c10::List<Tensor>({dummyTensor(DispatchKey::CPU), dummyTensor(DispatchKey::CPU)}));
  // 断言确保输出的大小为1
  EXPECT_EQ(1, outputs.size());
  // 断言确保输出的第一个元素为2
  EXPECT_EQ(2, outputs[0].toInt());
}

TEST(OperatorRegistrationTestLegacyLambdaBasedKernel, givenKernelWithLegacyTensorVectorInput_withoutOutput_whenRegistered_thenCanBeCalled) {
  // 创建操作注册器并注册具有遗留张量向量输入但无输出的操作
  auto registrar = RegisterOperators()
      .op("_test::tensor_list_input(Tensor[] input) -> ()", [] (const std::vector<Tensor>& input1) -> void {
        captured_input_list_size = input1.size();
      });

  // 查找已注册操作的架构
  auto op = c10::Dispatcher::singleton().findSchema({"_test::tensor_list_input", ""});
  // 断言确保操作架构存在
  ASSERT_TRUE(op.has_value());

  // 初始化捕获的输入列表大小为0
  captured_input_list_size = 0;
  // 调用操作并获取输出
  auto outputs = callOp(*op, c10::List<Tensor>({dummyTensor(DispatchKey::CPU), dummyTensor(DispatchKey::CPU)}));
  // 断言确保没有输出
  EXPECT_EQ(0, outputs.size());
  // 断言确保捕获的输入列表大小为2
  EXPECT_EQ(2, captured_input_list_size);
}

TEST(OperatorRegistrationTestLegacyLambdaBasedKernel, givenKernelWithLegacyTensorVectorInput_withOutput_whenRegistered_thenCanBeCalled) {
  // 创建操作注册器并注册具有遗留张量向量输入和整数输出的操作
  auto registrar = RegisterOperators()
      .op("_test::tensor_list_input(Tensor[] input) -> int", [] (const std::vector<Tensor>& input1) -> int64_t {
        return input1.size();
      });

  // 查找已注册操作的架构
  auto op = c10::Dispatcher::singleton().findSchema({"_test::tensor_list_input", ""});
  // 断言确保操作架构存在
  ASSERT_TRUE(op.has_value());

  // 调用操作并获取输出
  auto outputs = callOp(*op, c10::List<Tensor>({dummyTensor(DispatchKey::CPU), dummyTensor(DispatchKey::CPU)}));
  // 断言确保输出的大小为1
  EXPECT_EQ(1, outputs.size());
  // 断言确保输出的第一个元素为2
  EXPECT_EQ(2, outputs[0].toInt());
}

TEST(OperatorRegistrationTestLegacyLambdaBasedKernel, givenKernelWithLegacyTensorListInput_withoutOutput_whenRegistered_thenCanBeCalled) {
  // 创建操作注册器并注册具有遗留张量列表输入但无输出的操作
  auto registrar = RegisterOperators()
      .op("_test::tensor_list_input(Tensor[] input) -> ()", [] (std::vector<Tensor> input1) -> void {
        captured_input_list_size = input1.size();
      });

  // 查找已注册操作的架构
  auto op = c10::Dispatcher::singleton().findSchema({"_test::tensor_list_input", ""});
  // 断言确保操作架构存在
  ASSERT_TRUE(op.has_value());

  // 初始化捕获的输入列表大小为0
  captured_input_list_size = 0;
  // 调用操作并获取输出
  auto outputs = callOp(*op, c10::List<Tensor>({dummyTensor(DispatchKey::CPU), dummyTensor(DispatchKey::CPU)}));
  // 断言确保没有输出
  EXPECT_EQ(0, outputs.size());
  // 断言确保捕获的输入列表大小为2
  EXPECT_EQ(2, captured_input_list_size);
}
TEST(OperatorRegistrationTestLegacyLambdaBasedKernel, givenKernelWithLegacyTensorListInput_withOutput_whenRegistered_thenCanBeCalled) {
  // 创建操作注册对象，并注册带有张量列表输入和整数输出的操作
  auto registrar = RegisterOperators()
      .op("_test::tensor_list_input(Tensor[] input) -> int", [] (std::vector<Tensor> input1) -> int64_t {
        return input1.size();  // 返回输入张量列表的大小作为输出
      });

  // 查找已注册的操作模式
  auto op = c10::Dispatcher::singleton().findSchema({"_test::tensor_list_input", ""});
  ASSERT_TRUE(op.has_value());  // 断言确保找到对应的操作模式

  // 调用已注册的操作，传入两个CPU张量，并期望得到一个输出
  auto outputs = callOp(*op, c10::List<Tensor>({dummyTensor(DispatchKey::CPU), dummyTensor(DispatchKey::CPU)}));
  EXPECT_EQ(1, outputs.size());  // 断言确保输出的大小为1
  EXPECT_EQ(2, outputs[0].toInt());  // 断言确保输出的第一个元素为2
}

TEST(OperatorRegistrationTestLegacyLambdaBasedKernel, givenKernelWithStringListOutput_whenRegistered_thenCanBeCalled) {
  // 创建操作注册对象，并注册带有字符串列表输入和相同字符串列表输出的操作
  auto registrar = RegisterOperators()
      .op("_test::stringlist_output(str[] input) -> str[]", [](std::vector<std::string> input) {
        return input;  // 直接返回输入的字符串列表作为输出
      });

  // 查找已注册的操作模式
  auto op = c10::Dispatcher::singleton().findSchema({"_test::stringlist_output", ""});
  ASSERT_TRUE(op.has_value());  // 断言确保找到对应的操作模式

  // 创建字符串列表，调用已注册的操作，并期望得到一个输出
  c10::List<std::string> list({"value1", "value2"});
  auto outputs = callOp(*op, list);
  EXPECT_EQ(1, outputs.size());  // 断言确保输出的大小为1
  auto output = std::move(outputs[0]).toList();

  EXPECT_EQ(2, output.size());  // 断言确保输出列表的大小为2
  EXPECT_EQ("value1", output.get(0).toStringRef());  // 断言确保第一个输出为"value1"
  EXPECT_EQ("value2", output.get(1).toStringRef());  // 断言确保第二个输出为"value2"
}

TEST(OperatorRegistrationTestLegacyLambdaBasedKernel, givenKernelWithDictInput_withoutOutput_whenRegistered_thenCanBeCalled) {
  int captured_dict_size = 0;

  // 创建操作注册对象，并注册带有字典输入但无输出的操作
  auto registrar = RegisterOperators()
      .op("_test::dict_input(Dict(str, Tensor) input) -> ()", [&] (Dict<string, Tensor> input1) {
        captured_dict_size = input1.size();  // 记录输入字典的大小
      });

  // 查找已注册的操作模式
  auto op = c10::Dispatcher::singleton().findSchema({"_test::dict_input", ""});
  ASSERT_TRUE(op.has_value());  // 断言确保找到对应的操作模式

  captured_dict_size = 0;
  // 创建包含两个张量的字典，调用已注册的操作，并期望无输出
  Dict<string, Tensor> dict;
  dict.insert("key1", dummyTensor(DispatchKey::CPU));
  dict.insert("key2", dummyTensor(DispatchKey::CUDA));
  auto outputs = callOp(*op, dict);
  EXPECT_EQ(0, outputs.size());  // 断言确保输出的大小为0
  EXPECT_EQ(2, captured_dict_size);  // 断言确保记录的字典大小为2
}

TEST(OperatorRegistrationTestLegacyLambdaBasedKernel, givenKernelWithDictInput_withOutput_whenRegistered_thenCanBeCalled) {
  // 创建操作注册对象，并注册带有字典输入和字符串输出的操作
  auto registrar = RegisterOperators()
      .op("_test::dict_input(Dict(str, str) input) -> str", [&] (Dict<string, string> input1) {
        return input1.at("key2");  // 返回字典中"key2"对应的值作为输出
      });

  // 查找已注册的操作模式
  auto op = c10::Dispatcher::singleton().findSchema({"_test::dict_input", ""});
  ASSERT_TRUE(op.has_value());  // 断言确保找到对应的操作模式

  // 创建包含两个键值对的字典，调用已注册的操作，并期望得到一个输出
  Dict<string, string> dict;
  dict.insert("key1", "value1");
  dict.insert("key2", "value2");
  auto outputs = callOp(*op, dict);
  EXPECT_EQ(1, outputs.size());  // 断言确保输出的大小为1
  EXPECT_EQ("value2", outputs[0].toStringRef());  // 断言确保输出为"value2"
}

TEST(OperatorRegistrationTestLegacyLambdaBasedKernel, givenKernelWithDictOutput_whenRegistered_thenCanBeCalled) {
  // 创建操作注册对象，并注册带有字典输入和相同字典输出的操作
  auto registrar = RegisterOperators()
    .op("_test::dict_output(Dict(str, str) input) -> Dict(str, str)", [] (Dict<string, string> input) {
      return input;  // 直接返回输入的字典作为输出
    });
    });

  // 获取名为 "_test::dict_output" 的操作的调度器单例，并查找其对应的模式
  auto op = c10::Dispatcher::singleton().findSchema({"_test::dict_output", ""});
  // 断言操作存在
  ASSERT_TRUE(op.has_value());

  // 创建一个键值对类型为 <string, string> 的字典
  Dict<string, string> dict;
  // 向字典中插入两组键值对
  dict.insert("key1", "value1");
  dict.insert("key2", "value2");
  // 调用 op 所表示的操作，并传入字典作为参数，获取操作的输出
  auto outputs = callOp(*op, dict);
  // 断言输出的大小为 1
  EXPECT_EQ(1, outputs.size());
  // 将输出转换为 <string, string> 类型的字典
  auto output = c10::impl::toTypedDict<string, string>(outputs[0].toGenericDict());

  // 断言输出字典的大小为 2
  EXPECT_EQ(2, output.size());
  // 断言输出字典中键 "key1" 的值为 "value1"
  EXPECT_EQ("value1", output.at("key1"));
  // 断言输出字典中键 "key2" 的值为 "value2"
  EXPECT_EQ("value2", output.at("key2"));
TEST(OperatorRegistrationTestLegacyLambdaBasedKernel, givenKernelWithUnorderedMapInput_withoutOutput_whenRegistered_thenCanBeCalled) {
  // 初始化捕获的字典大小为0
  int captured_dict_size = 0;

  // 注册操作符，接受输入为字典类型的Tensor，没有输出，使用Lambda表达式定义操作行为
  auto registrar = RegisterOperators()
      .op("_test::dict_input(Dict(str, Tensor) input) -> ()", [&] (std::unordered_map<string, Tensor> input1) {
        // Lambda函数体内部，捕获传入的字典input1，将其大小赋值给captured_dict_size
        captured_dict_size = input1.size();
      });

  // 查找已注册的操作符的模式
  auto op = c10::Dispatcher::singleton().findSchema({"_test::dict_input", ""});
  // 断言确保找到相应的操作符模式
  ASSERT_TRUE(op.has_value());

  // 重置捕获的字典大小
  captured_dict_size = 0;
  // 创建一个C++标准库的Dict<string, Tensor>对象，并插入两个假的Tensor对象
  c10::Dict<string, Tensor> dict;
  dict.insert("key1", dummyTensor(DispatchKey::CPU));
  dict.insert("key2", dummyTensor(DispatchKey::CUDA));
  // 调用操作符，并传入字典dict作为参数，接收返回的输出
  auto outputs = callOp(*op, dict);
  // 断言检查输出的大小为0
  EXPECT_EQ(0, outputs.size());
  // 断言检查捕获的字典大小为2
  EXPECT_EQ(2, captured_dict_size);
}

TEST(OperatorRegistrationTestLegacyLambdaBasedKernel, givenKernelWithUnorderedMapInput_withOutput_whenRegistered_thenCanBeCalled) {
  // 注册操作符，接受输入为字典类型的字符串对，输出为字符串类型，使用Lambda表达式定义操作行为
  auto registrar = RegisterOperators()
      .op("_test::dict_input(Dict(str, str) input) -> str", [&] (std::unordered_map<string, string> input1) {
        // Lambda函数体内部，捕获传入的字典input1，返回key2对应的值
        return input1.at("key2");
      });

  // 查找已注册的操作符的模式
  auto op = c10::Dispatcher::singleton().findSchema({"_test::dict_input", ""});
  // 断言确保找到相应的操作符模式
  ASSERT_TRUE(op.has_value());

  // 创建一个C++标准库的Dict<string, string>对象，并插入两个键值对
  c10::Dict<string, string> dict;
  dict.insert("key1", "value1");
  dict.insert("key2", "value2");
  // 调用操作符，并传入字典dict作为参数，接收返回的输出
  auto outputs = callOp(*op, dict);
  // 断言检查输出的大小为1
  EXPECT_EQ(1, outputs.size());
  // 断言检查输出的值为"value2"
  EXPECT_EQ("value2", outputs[0].toStringRef());
}

TEST(OperatorRegistrationTestLegacyLambdaBasedKernel, givenKernelWithUnorderedMapOutput_whenRegistered_thenCanBeCalled) {
  // 注册操作符，接受输入为字典类型的字符串对，输出为相同类型的字典，使用Lambda表达式定义操作行为
  auto registrar = RegisterOperators()
    .op("_test::dict_output(Dict(str, str) input) -> Dict(str, str)", [] (std::unordered_map<string, string> input) {
      // Lambda函数体内部，捕获传入的字典input，直接返回它
      return input;
    });

  // 查找已注册的操作符的模式
  auto op = c10::Dispatcher::singleton().findSchema({"_test::dict_output", ""});
  // 断言确保找到相应的操作符模式
  ASSERT_TRUE(op.has_value());

  // 创建一个C++标准库的Dict<string, string>对象，并插入两个键值对
  c10::Dict<string, string> dict;
  dict.insert("key1", "value1");
  dict.insert("key2", "value2");
  // 调用操作符，并传入字典dict作为参数，接收返回的输出
  auto outputs = callOp(*op, dict);
  // 断言检查输出的大小为1
  EXPECT_EQ(1, outputs.size());
  // 将输出转换为具体类型为<string, string>的字典
  auto output = c10::impl::toTypedDict<string, string>(outputs[0].toGenericDict());
  // 断言检查输出的字典大小为2
  EXPECT_EQ(2, output.size());
  // 断言检查输出字典中的值与预期相符
  EXPECT_EQ("value1", output.at("key1"));
  EXPECT_EQ("value2", output.at("key2"));
}
TEST(OperatorRegistrationTestLegacyLambdaBasedKernel, givenKernelWithMapOfList_whenRegistered_thenCanBeCalled) {
  // 创建注册器，注册操作符 "_test::dict_output(Dict(str, int[]) input) -> Dict(str, int[])"，使用 lambda 函数定义操作逻辑
  auto registrar = RegisterOperators()
      .op("_test::dict_output(Dict(str, int[]) input) -> Dict(str, int[])", [](std::unordered_map<string, std::vector<int64_t>> input) {
        return input;
      });

  // 查找注册的操作符，并断言其存在
  auto op = c10::Dispatcher::singleton().findSchema({"_test::dict_output", ""});
  ASSERT_TRUE(op.has_value());

  // 创建 c10::Dict 对象，插入两个键值对，每个值是 c10::List<int64_t> 类型的列表
  c10::Dict<string, c10::List<int64_t>> dict;
  dict.insert("key1", c10::List<int64_t>({10, 20}));
  dict.insert("key2", c10::List<int64_t>({30, 40}));

  // 调用操作符，并获取输出结果
  auto outputs = callOp(*op, dict);

  // 断言输出结果的大小为 1
  EXPECT_EQ(1, outputs.size());

  // 将输出结果转换为具体类型为 string, c10::List<int64_t> 的字典
  auto output = c10::impl::toTypedDict<string, c10::List<int64_t>>(outputs[0].toGenericDict());

  // 断言输出字典的大小为 2
  EXPECT_EQ(2, output.size());

  // 断言第一个键 "key1" 的列表长度为 2
  EXPECT_EQ(2, output.at("key1").size());
  // 断言第一个键 "key1" 的第一个元素为 10
  EXPECT_EQ(10, output.at("key1").get(0));
  // 断言第一个键 "key1" 的第二个元素为 20
  EXPECT_EQ(20, output.at("key1").get(1));

  // 断言第二个键 "key2" 的列表长度为 2
  EXPECT_EQ(2, output.at("key2").size());
  // 断言第二个键 "key2" 的第一个元素为 30
  EXPECT_EQ(30, output.at("key2").get(0));
  // 断言第二个键 "key2" 的第二个元素为 40
  EXPECT_EQ(40, output.at("key2").get(1));
}


TEST(OperatorRegistrationTestLegacyLambdaBasedKernel, givenKernelWithMapOfListOfMap_whenRegistered_thenCanBeCalled) {
  // 创建注册器，注册操作符 "_test::dict_output(Dict(str, Dict(int,str)[]) input) -> Dict(str, Dict(int,str)[])"，使用 lambda 函数定义操作逻辑
  auto registrar = RegisterOperators()
      .op("_test::dict_output(Dict(str, Dict(int,str)[]) input) -> Dict(str, Dict(int,str)[])", [](std::unordered_map<string, std::vector<std::unordered_map<int64_t, string>>> input) {
        return input;
      });

  // 查找注册的操作符，并断言其存在
  auto op = c10::Dispatcher::singleton().findSchema({"_test::dict_output", ""});
  ASSERT_TRUE(op.has_value());

  // 创建 c10::Dict 对象，插入两个键值对，每个值是 c10::List<c10::Dict<int64_t, string>> 类型的列表
  c10::Dict<string, c10::List<c10::Dict<int64_t, string>>> dict;
  
  // 创建第一个字典，插入两个键值对
  c10::Dict<int64_t, string> dict1;
  dict1.insert(10, "10");
  dict1.insert(20, "20");
  // 将第一个字典插入到第一个键 "key1" 的列表中
  dict.insert("key1", c10::List<c10::Dict<int64_t, string>>({dict1}));

  // 创建第二个字典，插入两个键值对
  c10::Dict<int64_t, string> dict2;
  dict2.insert(30, "30");
  dict2.insert(40, "40");
  // 将第二个字典插入到第二个键 "key2" 的列表中
  dict.insert("key2", c10::List<c10::Dict<int64_t, string>>({dict2}));

  // 调用操作符，并获取输出结果
  auto outputs = callOp(*op, dict);

  // 断言输出结果的大小为 1
  EXPECT_EQ(1, outputs.size());

  // 将输出结果转换为具体类型为 string, c10::List<std::unordered_map<int64_t, string>> 的字典
  auto output = c10::impl::toTypedDict<string, c10::List<std::unordered_map<int64_t, string>>>(outputs[0].toGenericDict());

  // 断言输出字典的大小为 2
  EXPECT_EQ(2, output.size());

  // 断言第一个键 "key1" 的列表长度为 1
  EXPECT_EQ(1, output.at("key1").size());
  // 断言第一个键 "key1" 的第一个元素是一个字典，其大小为 2
  EXPECT_EQ(2, output.at("key1").get(0).size());
  // 断言第一个键 "key1" 的第一个元素中键 10 对应的值为 "10"
  EXPECT_EQ("10", output.at("key1").get(0).at(10));
  // 断言第一个键 "key1" 的第一个元素中键 20 对应的值为 "20"
  EXPECT_EQ("20", output.at("key1").get(0).at(20));

  // 断言第二个键 "key2" 的列表长度为 1
  EXPECT_EQ(1, output.at("key2").size());
  // 断言第二个键 "key2" 的第一个元素是一个字典，其大小为 2
  EXPECT_EQ(2, output.at("key2").get(0).size());
  // 断言第二个键 "key2" 的第一个元素中键 30 对应的值为 "30"
  EXPECT_EQ("30", output.at("key2").get(0).at(30));
  // 断言第二个键 "key2" 的第一个元素中键 40 对应的值为 "40"
  EXPECT_EQ("40", output.at("key2").get(0).at(40));
}
// 定义一个测试函数，测试使用旧版 Lambda 的内核注册机制
TEST(OperatorRegistrationTestLegacyLambdaBasedKernel, givenKernelWithListOfMap_whenRegistered_thenCanBeCalled) {
  // 创建操作符注册器对象，并注册一个操作符，该操作符接受一个包含多个无序映射的向量作为输入，并直接返回该输入
  auto registrar = RegisterOperators()
      .op("_test::list_output(Dict(str, int)[] input) -> Dict(str, int)[]", [](std::vector<std::unordered_map<string, int64_t>> input) {
        return input;
      });

  // 查找并获取名为 "_test::list_output" 的操作的模式
  auto op = c10::Dispatcher::singleton().findSchema({"_test::list_output", ""});
  // 断言找到的操作模式有效
  ASSERT_TRUE(op.has_value());

  // 创建两个 c10::Dict 对象，每个对象包含一些键值对
  c10::Dict<string, int64_t> dict1;
  dict1.insert("1", 1);
  dict1.insert("2", 2);
  c10::Dict<string, int64_t> dict2;
  dict2.insert("3", 3);
  dict2.insert("4", 4);
  
  // 创建一个 c10::List 对象，其中包含两个前面创建的字典对象
  c10::List<c10::Dict<string, int64_t>> list({dict1, dict2});
  
  // 调用先前注册的操作符，并传入 c10::List 对象作为参数，获取输出
  auto outputs = callOp(*op, list);
  
  // 断言输出结果的大小为 1
  EXPECT_EQ(1, outputs.size());
  
  // 将输出中的第一个元素转换为 c10::impl::GenericList 对象
  c10::impl::GenericList output = std::move(outputs[0]).toList();

  // 断言 GenericList 的大小为 2
  EXPECT_EQ(2, output.size());
  
  // 断言第一个 GenericDict 对象的大小为 2，并检查具体的键值对
  EXPECT_EQ(2, output.get(0).toGenericDict().size());
  EXPECT_EQ(1, output.get(0).toGenericDict().at("1").toInt());
  EXPECT_EQ(2, output.get(0).toGenericDict().at("2").toInt());
  
  // 断言第二个 GenericDict 对象的大小为 2，并检查具体的键值对
  EXPECT_EQ(2, output.get(1).toGenericDict().size());
  EXPECT_EQ(3, output.get(1).toGenericDict().at("3").toInt());
  EXPECT_EQ(4, output.get(1).toGenericDict().at("4").toInt());
}
TEST(OperatorRegistrationTestLegacyLambdaBasedKernel, givenKernelWithListOfMapOfIntList_whenRegistered_thenCanBeCalled) {
  // 创建一个操作符注册器，用于注册具有列表映射整数列表输入的操作符
  auto registrar = RegisterOperators()
      .op("_test::list_output(Dict(str, int[])[] input) -> Dict(str, int[])[]", [](std::vector<std::unordered_map<string, std::vector<int64_t>>> input) {
        // Lambda函数，返回输入参数本身
        return input;
      });

  // 查找指定名称的操作符模式
  auto op = c10::Dispatcher::singleton().findSchema({"_test::list_output", ""});
  ASSERT_TRUE(op.has_value());

  // 创建第一个字典对象 dict1
  c10::Dict<string, c10::List<int64_t>> dict1;
  dict1.insert("1", c10::List<int64_t>({1, 2}));
  dict1.insert("3", c10::List<int64_t>({3, 4}));

  // 创建第二个字典对象 dict2
  c10::Dict<string, c10::List<int64_t>> dict2;
  dict2.insert("5", c10::List<int64_t>({5, 6}));
  dict2.insert("7", c10::List<int64_t>({7, 8}));

  // 创建包含两个字典对象的列表
  c10::List<c10::Dict<string, c10::List<int64_t>>> list({ dict1, dict2 });

  // 调用操作符并获取输出结果
  auto outputs = callOp(*op, list);

  // 断言输出列表的大小为1
  EXPECT_EQ(1, outputs.size());

  // 将输出列表中的元素转换为通用列表对象
  c10::impl::GenericList output = std::move(outputs[0]).toList();

  // 断言通用列表的大小为2
  EXPECT_EQ(2, output.size());

  // 对第一个字典对象的验证
  EXPECT_EQ(2, output.get(0).toGenericDict().size());
  EXPECT_EQ(2, output.get(0).toGenericDict().at("1").toIntVector().size());
  EXPECT_EQ(1, output.get(0).toGenericDict().at("1").toIntVector()[0]);
  EXPECT_EQ(2, output.get(0).toGenericDict().at("1").toIntVector()[1]);
  EXPECT_EQ(2, output.get(0).toGenericDict().at("3").toIntVector().size());
  EXPECT_EQ(3, output.get(0).toGenericDict().at("3").toIntVector()[0]);
  EXPECT_EQ(4, output.get(0).toGenericDict().at("3").toIntVector()[1]);

  // 对第二个字典对象的验证
  EXPECT_EQ(2, output.get(1).toGenericDict().at("5").toIntVector().size());
  EXPECT_EQ(5, output.get(1).toGenericDict().at("5").toIntVector()[0]);
  EXPECT_EQ(6, output.get(1).toGenericDict().at("5").toIntVector()[1]);
  EXPECT_EQ(2, output.get(1).toGenericDict().at("7").toIntVector().size());
  EXPECT_EQ(7, output.get(1).toGenericDict().at("7").toIntVector()[0]);
  EXPECT_EQ(8, output.get(1).toGenericDict().at("7").toIntVector()[1]);
}

TEST(OperatorRegistrationTestLegacyLambdaBasedKernel, givenFallbackKernelWithoutAnyArguments_whenRegistered_thenCanBeCalled) {
  // 注意: 无张量参数的非回退内核不起作用，因为无法获取调度键。
  // 对于仅具有回退内核的操作符，这必须适用于向后兼容性。
  bool called = false;
  // 创建一个操作符注册器，注册没有张量参数的操作符
  auto registrar = RegisterOperators()
      .op("_test::no_tensor_args() -> ()", [&] () {called = true;});

  // 查找指定名称的操作符模式
  auto op = c10::Dispatcher::singleton().findSchema({"_test::no_tensor_args", ""});
  ASSERT_TRUE(op.has_value());

  called = false;
  // 调用操作符
  auto outputs = callOp(*op);
  // 断言调用成功
  EXPECT_TRUE(called);
}
TEST(OperatorRegistrationTestLegacyLambdaBasedKernel, givenFallbackKernelWithoutTensorArguments_whenRegistered_thenCanBeCalled) {
  // 注：没有张量参数的非回退内核无法工作，因为无法获取调度键。对于只有回退内核的运算符，这必须为向后兼容性而工作。
  auto registrar = RegisterOperators()
      .op("_test::no_tensor_args(int arg) -> int", [] (int64_t arg) {return arg + 1;});

  // 查找指定模式的操作符模式
  auto op = c10::Dispatcher::singleton().findSchema({"_test::no_tensor_args", ""});
  ASSERT_TRUE(op.has_value());

  // 调用操作符，并验证输出的正确性
  auto outputs = callOp(*op, 3);
  EXPECT_EQ(1, outputs.size());
  EXPECT_EQ(4, outputs[0].toInt());
}

TEST(OperatorRegistrationTestLegacyLambdaBasedKernel, givenKernelWithOptionalInputs_withoutOutput_whenRegistered_thenCanBeCalled) {
  bool called = false;
  std::optional<Tensor> called_arg2 = c10::nullopt;
  std::optional<int64_t> called_arg3 = c10::nullopt;
  std::optional<std::string> called_arg4 = c10::nullopt;

  // 注册具有可选输入但无输出的操作符
  auto registrar = RegisterOperators().op(
    "_test::opt_input(Tensor arg1, Tensor? arg2, int? arg3, str? arg4) -> ()",
    [&] (Tensor arg1, const std::optional<Tensor>& arg2, std::optional<int64_t> arg3, std::optional<std::string> arg4) {
      called = true;
      called_arg2 = arg2;
      called_arg3 = arg3;
      called_arg4 = arg4;
    });
  auto op = c10::Dispatcher::singleton().findSchema({"_test::opt_input", ""});
  ASSERT_TRUE(op.has_value());

  called = false;
  // 调用操作符，并验证输出的正确性
  auto outputs = callOp(*op, dummyTensor(DispatchKey::CPU), dummyTensor(DispatchKey::CUDA), c10::IValue(), std::string("text"));
  EXPECT_EQ(0, outputs.size());

  EXPECT_TRUE(called);
  EXPECT_TRUE(called_arg2.has_value());
  EXPECT_EQ(extractDispatchKey(*called_arg2), DispatchKey::CUDA);
  EXPECT_FALSE(called_arg3.has_value());
  EXPECT_TRUE(called_arg4.has_value());
  EXPECT_EQ(*called_arg4, "text");

  called = false;
  // 再次调用操作符，并验证输出的正确性
  outputs = callOp(*op, dummyTensor(DispatchKey::CPU), c10::IValue(), 4, c10::IValue());
  EXPECT_EQ(0, outputs.size());

  EXPECT_TRUE(called);
  EXPECT_FALSE(called_arg2.has_value());
  EXPECT_TRUE(called_arg3.has_value());
  EXPECT_EQ(*called_arg3, 4);
  EXPECT_FALSE(called_arg4.has_value());
}

TEST(OperatorRegistrationTestLegacyLambdaBasedKernel, givenKernelWithOptionalInputs_withOutput_whenRegistered_thenCanBeCalled) {
  bool called = false;
  std::optional<Tensor> called_arg2 = c10::nullopt;
  std::optional<int64_t> called_arg3 = c10::nullopt;
  std::optional<std::string> called_arg4 = c10::nullopt;

  // 注册具有可选输入和输出的操作符
  auto registrar = RegisterOperators().op(
    "_test::opt_input(Tensor arg1, Tensor? arg2, int? arg3, str? arg4) -> Tensor?",
    [&] (Tensor arg1, const std::optional<Tensor>& arg2, std::optional<int64_t> arg3, std::optional<std::string> arg4) {
      called = true;
      called_arg2 = arg2;
      called_arg3 = arg3;
      called_arg4 = arg4;
      return arg2;
    });
  });
  // 使用 c10::Dispatcher 的单例获取名称为 "_test::opt_input" 的操作模式
  auto op = c10::Dispatcher::singleton().findSchema({"_test::opt_input", ""});
  // 断言操作模式存在
  ASSERT_TRUE(op.has_value());

  called = false;
  // 调用找到的操作模式，传入多个参数：包括两个 dummyTensor，一个空的 c10::IValue，和一个字符串 "text"
  auto outputs = callOp(*op, dummyTensor(DispatchKey::CPU), dummyTensor(DispatchKey::CUDA), c10::IValue(), std::string("text"));
  // 断言输出的大小为 1
  EXPECT_EQ(1, outputs.size());
  // 断言第一个输出的分发键为 DispatchKey::CUDA
  EXPECT_EQ(DispatchKey::CUDA, extractDispatchKey(outputs[0].toTensor()));

  // 断言 called 变量为 true
  EXPECT_TRUE(called);
  // 断言 called_arg2 有值
  EXPECT_TRUE(called_arg2.has_value());
  // 断言 called_arg2 的值的分发键为 DispatchKey::CUDA
  EXPECT_EQ(extractDispatchKey(*called_arg2), DispatchKey::CUDA);
  // 断言 called_arg3 没有值
  EXPECT_FALSE(called_arg3.has_value());
  // 断言 called_arg4 有值
  EXPECT_TRUE(called_arg4.has_value());
  // 断言 called_arg4 的值为字符串 "text"
  EXPECT_EQ(*called_arg4, "text");

  called = false;
  // 再次调用相同的操作模式，传入一个 dummyTensor，一个 c10::IValue 值 4，和一个空的 c10::IValue
  outputs = callOp(*op, dummyTensor(DispatchKey::CPU), c10::IValue(), 4, c10::IValue());
  // 断言输出的大小为 1
  EXPECT_EQ(1, outputs.size());
  // 断言第一个输出为 None
  EXPECT_TRUE(outputs[0].isNone());

  // 断言 called 变量为 true
  EXPECT_TRUE(called);
  // 断言 called_arg2 没有值
  EXPECT_FALSE(called_arg2.has_value());
  // 断言 called_arg3 有值
  EXPECT_TRUE(called_arg3.has_value());
  // 断言 called_arg3 的值为 4
  EXPECT_EQ(*called_arg3, 4);
  // 断言 called_arg4 没有值
  EXPECT_FALSE(called_arg4.has_value());
TEST(OperatorRegistrationTestLegacyLambdaBasedKernel, givenKernelWithOptionalInputs_withMultipleOutputs_whenRegistered_thenCanBeCalled) {
  // 创建注册器并注册运算符，指定运算符的签名和Lambda函数
  auto registrar = RegisterOperators().op(
    "_test::opt_input(Tensor arg1, Tensor? arg2, int? arg3, str? arg4) -> (Tensor?, int?, str?)",
    [] (Tensor arg1, const std::optional<Tensor>& arg2, std::optional<int64_t> arg3, std::optional<std::string> arg4) {
      // Lambda函数返回一个tuple，包含可选输入参数的值
      return std::make_tuple(arg2, arg3, arg4);
    });
  // 查找已注册的运算符的模式(schema)
  auto op = c10::Dispatcher::singleton().findSchema({"_test::opt_input", ""});
  // 断言找到对应的运算符模式
  ASSERT_TRUE(op.has_value());

  // 调用注册的运算符，传入参数，并获取返回的输出
  auto outputs = callOp(*op, dummyTensor(DispatchKey::CPU), dummyTensor(DispatchKey::CUDA), c10::IValue(), std::string("text"));
  // 断言输出的大小为3
  EXPECT_EQ(3, outputs.size());
  // 断言第一个输出的DispatchKey为CUDA
  EXPECT_EQ(DispatchKey::CUDA, extractDispatchKey(outputs[0].toTensor()));
  // 断言第二个输出为None类型
  EXPECT_TRUE(outputs[1].isNone());
  // 断言第三个输出为字符串"text"
  EXPECT_EQ("text", outputs[2].toStringRef());

  // 再次调用注册的运算符，传入另一组参数，并获取返回的输出
  outputs = callOp(*op, dummyTensor(DispatchKey::CPU), c10::IValue(), 4, c10::IValue());
  // 断言输出的大小为3
  EXPECT_EQ(3, outputs.size());
  // 断言第一个输出为None类型
  EXPECT_TRUE(outputs[0].isNone());
  // 断言第二个输出为整数4
  EXPECT_EQ(4, outputs[1].toInt());
  // 断言第三个输出为None类型
  EXPECT_TRUE(outputs[2].isNone());
}

void expectCallsConcatUnboxed(DispatchKey dispatch_key) {
  // 进入自动禁用Autograd模式
  at::AutoDispatchBelowAutograd mode;

  // 断言找到已注册的运算符模式和CPU内核
  auto op = c10::Dispatcher::singleton().findSchema({"_test::my_op", ""});
  ASSERT_TRUE(op.has_value());
  // 调用未封装的callOp函数，传入参数并获取字符串类型的结果
  std::string result = callOpUnboxed<std::string, const Tensor&, std::string, const std::string&, int64_t>(*op, dummyTensor(dispatch_key), "1", "2", 3);
  // 断言结果等于"prefix123"
  EXPECT_EQ("prefix123", result);
}

TEST(OperatorRegistrationTestLegacyLambdaBasedKernel, givenKernel_whenRegistered_thenCanBeCalledUnboxed) {
  // 定义前缀字符串
  std::string prefix = "prefix";
  // 创建注册器并注册运算符，指定运算符的签名和Lambda函数
  auto registrar = RegisterOperators().op("_test::my_op(Tensor dummy, str a, str b, int c) -> str", [&] (const Tensor& tensor1, std::string a, const std::string& b, int64_t c) {
    // Lambda函数返回连接前缀、字符串a、b和整数c的结果字符串
    return prefix + a + b + std::to_string(c);
  });
  // 调用函数，传入DispatchKey::CPU参数
  expectCallsConcatUnboxed(DispatchKey::CPU);
}

TEST(OperatorRegistrationTestLegacyLambdaBasedKernel, givenKernel_whenRegisteredWithoutSpecifyingSchema_thenInfersSchema) {
  // 创建注册器并注册运算符，指定运算符的签名和Lambda函数
  auto registrar = RegisterOperators()
      .op("_test::no_schema_specified", [] (Tensor arg1, int64_t arg2, const std::vector<Tensor>& arg3) -> std::tuple<int64_t, Tensor> {return {};});

  // 查找已注册的运算符的模式(schema)
  auto op = c10::Dispatcher::singleton().findSchema({"_test::no_schema_specified", ""});
  // 断言找到对应的运算符模式
  ASSERT_TRUE(op.has_value());

  // 检查预期和实际的运算符模式之间的差异
  std::optional<std::string> differences = c10::findSchemaDifferences(torch::jit::parseSchema("_test::no_schema_specified(Tensor arg1, int arg2, Tensor[] arg3) -> (int, Tensor)"), op->schema());
  // 断言没有差异存在
  EXPECT_FALSE(differences.has_value());
}
TEST(OperatorRegistrationTestLegacyLambdaBasedKernel, givenMismatchedKernel_withDifferentNumArguments_whenRegistering_thenFails) {
  // 定义测试用例: 注册操作符，期望注册失败因为参数数量不匹配

  // 注册操作符，声明一个参数的函数
  RegisterOperators()
      .op("_test::mismatch(Tensor arg) -> int", [] (Tensor) -> int64_t {return 0;});

  // 期望抛出异常，因为参数数量不同
  expectThrows<c10::Error>([] {
    // 注册操作符，声明两个参数的函数
    RegisterOperators()
        .op("_test::mismatch(Tensor arg, Tensor arg2) -> int", [] (Tensor) -> int64_t {return 0;});
    }, "The number of arguments is different. 2 vs 1"
  );

  // 注册操作符，声明两个参数的函数
  RegisterOperators()
      .op("_test::mismatch(Tensor arg, Tensor arg2) -> ()", [] (Tensor, Tensor) -> void {});

  // 期望抛出异常，因为参数数量不同
  expectThrows<c10::Error>([] {
    // 注册操作符，声明无参数的函数
    RegisterOperators()
        .op("_test::mismatch() -> ()", [] (Tensor, Tensor) -> void {});
    }, "The number of arguments is different. 0 vs 2"
  );

  // 期望抛出异常，因为参数数量不同
  expectThrows<c10::Error>([] {
    // 注册操作符，声明一个参数的函数
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> ()", [] (Tensor, Tensor) -> void {});
    }, "The number of arguments is different. 1 vs 2"
  );

  // 期望抛出异常，因为参数数量不同
  expectThrows<c10::Error>([] {
    // 注册操作符，声明三个参数的函数
    RegisterOperators()
        .op("_test::mismatch(Tensor arg, Tensor arg2, Tensor arg3) -> ()", [] (Tensor, Tensor) -> void {});
    }, "The number of arguments is different. 3 vs 2"
  );
}

TEST(OperatorRegistrationTestLegacyLambdaBasedKernel, givenMismatchedKernel_withDifferentArgumentType_whenRegistering_thenFails) {
  // 定义测试用例: 注册操作符，期望注册失败因为参数类型不匹配

  // 注册操作符，声明两个参数的函数，参数类型匹配
  RegisterOperators()
      .op("_test::mismatch(Tensor arg1, int arg2) -> int", [] (Tensor, int64_t) -> int64_t {return 0;});

  // 期望抛出异常，因为第二个参数类型不匹配
  expectThrows<c10::Error>([] {
    // 注册操作符，声明两个参数的函数，第二个参数类型不匹配
    RegisterOperators()
        .op("_test::mismatch(Tensor arg1, float arg2) -> int", [] (Tensor, int64_t) -> int64_t {return 0;});
    }, "Type mismatch in argument 2: float vs int"
  );

  // 期望抛出异常，因为第一个参数类型不匹配
  expectThrows<c10::Error>([] {
    // 注册操作符，声明两个参数的函数，第一个参数类型不匹配
    RegisterOperators()
        .op("_test::mismatch(int arg1, int arg2) -> int", [] (Tensor, int64_t) -> int64_t {return 0;});
    }, "Type mismatch in argument 1: int vs Tensor"
  );
}

TEST(OperatorRegistrationTestLegacyLambdaBasedKernel, givenMismatchedKernel_withDifferentNumReturns_whenRegistering_thenFails) {
  // 定义测试用例: 注册操作符，期望注册失败因为返回值数量不匹配

  // 注册操作符，声明一个参数的函数，返回值数量匹配
  RegisterOperators()
      .op("_test::mismatch(Tensor arg) -> int", [] (Tensor) -> int64_t {return 0;});

  // 期望抛出异常，因为返回值数量不匹配
  expectThrows<c10::Error>([] {
    // 注册操作符，声明一个参数的函数，返回值数量不匹配
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> ()", [] (Tensor) -> int64_t {return 0;});
    }, "The number of returns is different. 0 vs 1"
  );

  // 期望抛出异常，因为返回值数量不匹配
  expectThrows<c10::Error>([] {
    // 注册操作符，声明一个参数的函数，返回值数量不匹配
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> (int, int)", [] (Tensor) -> int64_t {return 0;});
  // 断言这不会失败，因为它匹配
  RegisterOperators()
      .op("_test::mismatch(Tensor arg) -> ()", [] (Tensor) -> void {});

  // 然后是一组不匹配的模式
  expectThrows<c10::Error>([] {
    // 注册操作符，声明一个不匹配的函数签名，期望抛出异常
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> Tensor", [] (Tensor) -> void {});
    }, "The number of returns is different. 1 vs 0"
  );

  // 期望抛出异常，因为返回值数量不同
  expectThrows<c10::Error>([] {
    // 注册操作符，声明一个不匹配的函数签名，期望抛出异常
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> (Tensor, Tensor)", [] (Tensor) -> void {});
    }, "The number of returns is different. 2 vs 0"
  );

  // 断言这不会失败，因为它匹配
  RegisterOperators()
      .op("_test::mismatch(Tensor arg) -> (Tensor, Tensor)", [] (Tensor) -> std::tuple<Tensor, Tensor> {return {};});

  // 然后是一组不匹配的模式
  expectThrows<c10::Error>([] {
    // 注册操作符，声明一个不匹配的函数签名，期望抛出异常
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> ()", [] (Tensor) -> std::tuple<Tensor, Tensor> {return {};});
    }, "The number of returns is different. 0 vs 2"
  );

  // 期望抛出异常，因为返回值数量不同
  expectThrows<c10::Error>([] {
    // 注册操作符，声明一个不匹配的函数签名，期望抛出异常
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> Tensor", [] (Tensor) -> std::tuple<Tensor, Tensor> {return {};});
    }, "The number of returns is different. 1 vs 2"
  );

  // 期望抛出异常，因为返回值数量不同
  expectThrows<c10::Error>([] {
    // 注册操作符，声明一个不匹配的函数签名，期望抛出异常
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> (Tensor, Tensor, Tensor)", [] (Tensor) -> std::tuple<Tensor, Tensor> {return {};});
    }, "The number of returns is different. 3 vs 2"
  );
TEST(OperatorRegistrationTestLegacyLambdaBasedKernel, givenMismatchedKernel_withDifferentReturnTypes_whenRegistering_thenFails) {
  // 测试用例：对于不同返回类型的不匹配内核注册，应该失败

  // 断言这里不会失败，因为它匹配
  RegisterOperators()
      .op("_test::mismatch(Tensor arg) -> int", [] (Tensor) -> int64_t {return 0;});

  // 然后是一组不匹配的模式
  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> Tensor", [] (Tensor) -> int64_t {return 0;});
    }, "Type mismatch in return 1: Tensor vs int"
  );

  // 然后是一组不匹配的模式
  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> float", [] (Tensor) -> int64_t {return 0;});
    }, "Type mismatch in return 1: float vs int"
  );

  // 断言这里不会失败，因为它匹配
  RegisterOperators()
      .op("_test::mismatch(Tensor arg) -> Tensor", [] (Tensor) -> Tensor {return {};});

  // 然后是一组不匹配的模式
  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> float", [] (Tensor) -> Tensor {return {};});
    }, "Type mismatch in return 1: float vs Tensor"
  );

  // 断言这里不会失败，因为它匹配
  RegisterOperators()
      .op("_test::mismatch(Tensor arg) -> (Tensor, int)", [] (Tensor) -> std::tuple<Tensor, int64_t> {return {};});

  // 然后是一组不匹配的模式
  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> (Tensor, float)", [] (Tensor) -> std::tuple<Tensor, int64_t> {return {};});
    }, "Type mismatch in return 2: float vs int"
  );

  // 然后是一组不匹配的模式
  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> (int, int)", [] (Tensor) -> std::tuple<Tensor, int64_t> {return {};});
    }, "Type mismatch in return 1: int vs Tensor"
  );
}
```