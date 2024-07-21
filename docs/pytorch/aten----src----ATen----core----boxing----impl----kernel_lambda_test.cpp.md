# `.\pytorch\aten\src\ATen\core\boxing\impl\kernel_lambda_test.cpp`

```py
#include <gtest/gtest.h>  // 引入 Google Test 框架的头文件

#include <ATen/core/boxing/impl/test_helpers.h>  // 引入 ATen 的测试辅助函数头文件

#include <ATen/core/op_registration/op_registration.h>  // 引入 ATen 的操作注册相关头文件
#include <ATen/core/Tensor.h>  // 引入 ATen 的 Tensor 类头文件
#include <torch/csrc/jit/frontend/function_schema_parser.h>  // 引入 Torch 的函数 schema 解析头文件
#include <torch/library.h>  // 引入 Torch 库头文件

#include <ATen/core/LegacyTypeDispatch.h>  // 引入 ATen 的 LegacyTypeDispatch 头文件

using c10::RegisterOperators;  // 使用 c10 命名空间中的 RegisterOperators 类
using c10::DispatchKey;  // 使用 c10 命名空间中的 DispatchKey 枚举
using c10::Stack;  // 使用 c10 命名空间中的 Stack 类
using std::make_unique;  // 使用 std 命名空间中的 make_unique 函数
using c10::intrusive_ptr;  // 使用 c10 命名空间中的 intrusive_ptr 类
using c10::Dict;  // 使用 c10 命名空间中的 Dict 类
using at::Tensor;  // 使用 at 命名空间中的 Tensor 类
using std::string;  // 使用 std 命名空间中的 string 类
using std::unique_ptr;  // 使用 std 命名空间中的 unique_ptr 类

namespace {

void expectCallsIncrement(DispatchKey dispatch_key) {
  at::AutoDispatchBelowAutograd mode;  // 设置自动分发模式为禁用 Autograd

  // 断言是否存在指定的操作 schema 和 CPU 内核
  auto op = c10::Dispatcher::singleton().findSchema({"_test::my_op", ""});
  ASSERT_TRUE(op.has_value());  // 确保操作存在

  // 调用指定操作，并验证返回结果
  auto result = callOp(*op, dummyTensor(dispatch_key), 5);
  EXPECT_EQ(1, result.size());  // 确保结果大小为 1
  EXPECT_EQ(6, result[0].toInt());  // 确保返回值为 6
}

void expectCallsDecrement(DispatchKey dispatch_key) {
  at::AutoDispatchBelowAutograd mode;  // 设置自动分发模式为禁用 Autograd

  // 断言是否存在指定的操作 schema 和 CPU 内核
  auto op = c10::Dispatcher::singleton().findSchema({"_test::my_op", ""});
  ASSERT_TRUE(op.has_value());  // 确保操作存在

  // 调用指定操作，并验证返回结果
  auto result = callOp(*op, dummyTensor(dispatch_key), 5);
  EXPECT_EQ(1, result.size());  // 确保结果大小为 1
  EXPECT_EQ(4, result[0].toInt());  // 确保返回值为 4
}

TEST(OperatorRegistrationTestLambdaBasedKernel, givenKernel_whenRegistered_thenCanBeCalled) {
  // 注册一个使用 Lambda 表达式定义的 CPU 内核的操作
  auto registrar = RegisterOperators().op("_test::my_op(Tensor dummy, int input) -> int", RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor, int64_t i) {return i+1;}));
  expectCallsIncrement(DispatchKey::CPU);  // 调用并期望调用递增
}

TEST(OperatorRegistrationTestLambdaBasedKernel, givenOutOfLineKernel_whenRegistered_thenCanBeCalled) {
  auto my_kernel = [] (Tensor, int64_t i) {return i+1;};  // 定义一个 Lambda 表达式作为 CPU 内核
  // 注册一个使用 Lambda 表达式定义的 CPU 内核的操作
  auto registrar = RegisterOperators().op("_test::my_op(Tensor dummy, int input) -> int", RegisterOperators::options().kernel(DispatchKey::CPU, my_kernel));
  expectCallsIncrement(DispatchKey::CPU);  // 调用并期望调用递增
}

TEST(OperatorRegistrationTestLambdaBasedKernel, givenMultipleOperatorsAndKernels_whenRegisteredInOneRegistrar_thenCallsRightKernel) {
  // 在一个注册器中注册多个操作及其内核，包括 CPU 和 CUDA 内核
  auto registrar = RegisterOperators()
      .op("_test::my_op(Tensor dummy, int input) -> int", RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor, int64_t i) {return i+1;})
                                                                                      .kernel(DispatchKey::CUDA, [] (Tensor, int64_t) -> int64_t {EXPECT_TRUE(false); return 0;}))
      .op("_test::error(Tensor dummy, int input) -> int", RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor, int64_t) -> int64_t {EXPECT_TRUE(false); return 0;})
                                                                                      .kernel(DispatchKey::CUDA, [] (Tensor, int64_t) -> int64_t {EXPECT_TRUE(false); return 0;}));
  expectCallsIncrement(DispatchKey::CPU);  // 调用并期望调用递增
}
// 定义测试函数 OperatorRegistrationTestLambdaBasedKernel，测试注册带有 lambda 表达式内核的操作符行为
TEST(OperatorRegistrationTestLambdaBasedKernel, givenMultipleOperatorsAndKernels_whenRegisteredInMultipleRegistrars_thenCallsRightKernel) {
  // 注册第一个操作符 "_test::my_op"，包含 CPU 和 CUDA 两个内核
  auto registrar1 = RegisterOperators().op("_test::my_op(Tensor dummy, int input) -> int", 
    RegisterOperators::options()
      .kernel(DispatchKey::CPU, [] (Tensor, int64_t i) {return i+1;})
      .kernel(DispatchKey::CUDA, [] (Tensor, int64_t) -> int64_t {EXPECT_TRUE(false); return 0;}));
  
  // 注册第二个操作符 "_test::error"，同样包含 CPU 和 CUDA 两个内核
  auto registrar3 = RegisterOperators().op("_test::error(Tensor dummy, int input) -> int",
    RegisterOperators::options()
      .kernel(DispatchKey::CPU, [] (Tensor, int64_t) -> int64_t {EXPECT_TRUE(false); return 0;})
      .kernel(DispatchKey::CUDA, [] (Tensor, int64_t) -> int64_t {EXPECT_TRUE(false); return 0;}));
  
  // 验证调用 CPU 内核的正确性
  expectCallsIncrement(DispatchKey::CPU);
}

// 定义测试函数 OperatorRegistrationTestLambdaBasedKernel，测试 lambda 内核注册超出作用域后的行为
TEST(OperatorRegistrationTestLambdaBasedKernel, givenKernel_whenRegistrationRunsOutOfScope_thenCannotBeCalledAnymore) {
  {
    // 定义 TorchScript 库 m，用于注册 "_test::my_op" 的操作符架构
    auto m = MAKE_TORCH_LIBRARY(_test);
    m.def("_test::my_op(Tensor dummy, int input) -> int");
    
    // 在 CPU 上注册 "_test::my_op" 的内核函数
    auto m_cpu = MAKE_TORCH_LIBRARY_IMPL(_test, CPU);
    m_cpu.impl("my_op", DispatchKey::CPU, [] (Tensor, int64_t i) {return i+1;});
    
    {
      // 在 CUDA 上注册 "_test::my_op" 的内核函数
      auto m_cuda = MAKE_TORCH_LIBRARY_IMPL(_test, CUDA);
      m_cuda.impl("my_op", DispatchKey::CUDA, [] (Tensor, int64_t i) {return i-1;});

      // 断言架构和 CPU 内核存在
      expectCallsIncrement(DispatchKey::CPU);
      // 断言 CUDA 内核不存在
      expectCallsDecrement(DispatchKey::CUDA);
    }

    // 现在 m_cuda 注册器已经被销毁。断言架构仍然存在但 CPU 内核不存在
    expectCallsIncrement(DispatchKey::CPU);
    expectDoesntFindKernel("_test::my_op", DispatchKey::CUDA);
  }

  // 现在两个注册器都已被销毁。断言整个架构已经不存在
  expectDoesntFindOperator("_test::my_op");
}

// 声明一个全局变量，用于测试用例
bool was_called = false;

// 定义测试函数 OperatorRegistrationTestLambdaBasedKernel，测试注册没有输出的 lambda 内核的行为
TEST(OperatorRegistrationTestLambdaBasedKernel, givenKernelWithoutOutput_whenRegistered_thenCanBeCalled) {
  // 注册 "_test::no_return" 操作符，带有 CPU 内核
  auto registrar = RegisterOperators().op("_test::no_return(Tensor dummy) -> ()",
    RegisterOperators::options()
      .kernel(DispatchKey::CPU, [] (const Tensor&) -> void {was_called = true;}));

  // 查找操作符 "_test::no_return" 的架构
  auto op = c10::Dispatcher::singleton().findSchema({"_test::no_return", ""});
  ASSERT_TRUE(op.has_value());
  
  // 重置 was_called 标志位
  was_called = false;
  
  // 调用操作符，并验证 CPU 内核被调用
  auto result = callOp(*op, dummyTensor(DispatchKey::CPU));
  EXPECT_TRUE(was_called);
  // 验证结果的大小为 0
  EXPECT_EQ(0, result.size());
}

// 定义测试函数 OperatorRegistrationTestLambdaBasedKernel，测试注册带有零输出的 lambda 内核的行为
TEST(OperatorRegistrationTestLambdaBasedKernel, givenKernelWithZeroOutputs_whenRegistered_thenCanBeCalled) {
  // 注册 "_test::zero_outputs" 操作符，带有 CPU 内核
  auto registrar = RegisterOperators().op("_test::zero_outputs(Tensor dummy) -> ()",
    // 获取 CPU 分发键对应的操作符选项，并注册一个内核函数，当调用时设置 `was_called` 为 true，无返回值
    RegisterOperators::options().kernel(DispatchKey::CPU, [] (const Tensor&) -> std::tuple<> {was_called = true; return {};}));

    // 使用 Dispatcher 单例对象查找指定的操作模式（schema），此处查找 "_test::zero_outputs" 的操作
    auto op = c10::Dispatcher::singleton().findSchema({"_test::zero_outputs", ""});
    // 断言确保找到了相应的操作模式
    ASSERT_TRUE(op.has_value());

    // 重置 `was_called` 标志为 false，准备进行操作调用
    was_called = false;

    // 调用找到的操作模式，并期望设置 `was_called` 为 true，同时检查返回结果的大小应为 0
    auto result = callOp(*op, dummyTensor(DispatchKey::CPU));
    EXPECT_TRUE(was_called);
    EXPECT_EQ(0, result.size());
}

TEST(OperatorRegistrationTestLambdaBasedKernel, givenKernelWithIntOutput_whenRegistered_thenCanBeCalled) {
  // 创建一个操作注册器对象，用于注册自定义操作
  auto registrar = RegisterOperators()
      // 注册一个接受 Tensor 和两个整数参数的操作，并指定其 CPU 分发的内核函数
      .op("_test::int_output(Tensor dummy, int a, int b) -> int",
        RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor, int64_t a, int64_t b) {return a+b;}));

  // 查找注册的操作模式
  auto op = c10::Dispatcher::singleton().findSchema({"_test::int_output", ""});
  // 断言操作模式已找到
  ASSERT_TRUE(op.has_value());

  // 调用注册的操作，并验证返回结果
  auto result = callOp(*op, dummyTensor(DispatchKey::CPU), 3, 6);
  EXPECT_EQ(1, result.size());
  EXPECT_EQ(9, result[0].toInt());
}

TEST(OperatorRegistrationTestLambdaBasedKernel, givenKernelWithTensorOutput_whenRegistered_thenCanBeCalled) {
  // 创建一个操作注册器对象，用于注册自定义操作
  auto registrar = RegisterOperators()
      // 注册一个接受一个 Tensor 输入并返回 Tensor 的操作，并指定其 CPU 和 CUDA 分发的内核函数
      .op("_test::returning_tensor(Tensor input) -> Tensor",
        RegisterOperators::options().kernel(DispatchKey::CPU, [] (const Tensor& a) {return a;})
                                    .kernel(DispatchKey::CUDA, [] (const Tensor& a) {return a;}));

  // 查找注册的操作模式
  auto op = c10::Dispatcher::singleton().findSchema({"_test::returning_tensor", ""});
  // 断言操作模式已找到
  ASSERT_TRUE(op.has_value());

  // 调用注册的操作，并验证返回结果
  auto result = callOp(*op, dummyTensor(DispatchKey::CPU));
  EXPECT_EQ(1, result.size());
  EXPECT_EQ(DispatchKey::CPU, extractDispatchKey(result[0].toTensor()));

  // 再次调用注册的操作，验证 CUDA 分发的返回结果
  result = callOp(*op, dummyTensor(DispatchKey::CUDA));
  EXPECT_EQ(1, result.size());
  EXPECT_EQ(DispatchKey::CUDA, extractDispatchKey(result[0].toTensor()));
}

TEST(OperatorRegistrationTestLambdaBasedKernel, givenKernelWithTensorListOutput_whenRegistered_thenCanBeCalled) {
  // 创建一个操作注册器对象，用于注册自定义操作
  auto registrar = RegisterOperators()
      // 注册一个接受三个 Tensor 输入并返回 Tensor 列表的操作，并指定其 CUDA 分发的内核函数
      .op("_test::list_output(Tensor input1, Tensor input2, Tensor input3) -> Tensor[]",
        RegisterOperators::options().kernel(DispatchKey::CUDA, [] (const Tensor& a, const Tensor& b, const Tensor& c) -> c10::List<Tensor> {return c10::List<Tensor>({a, b, c});}));

  // 查找注册的操作模式
  auto op = c10::Dispatcher::singleton().findSchema({"_test::list_output", ""});
  // 断言操作模式已找到
  ASSERT_TRUE(op.has_value());

  // 调用注册的操作，并验证返回结果
  auto result = callOp(*op, dummyTensor(DispatchKey::CPU), dummyTensor(DispatchKey::CUDA), dummyTensor(DispatchKey::CPU));
  EXPECT_EQ(1, result.size());
  EXPECT_EQ(3, result[0].toTensorVector().size());
  EXPECT_EQ(DispatchKey::CPU, extractDispatchKey(result[0].toTensorVector()[0]));
  EXPECT_EQ(DispatchKey::CUDA, extractDispatchKey(result[0].toTensorVector()[1]));
  EXPECT_EQ(DispatchKey::CPU, extractDispatchKey(result[0].toTensorVector()[2]));
}
// 测试用例定义：测试带有整数列表输出的内核注册和调用情况
TEST(OperatorRegistrationTestLambdaBasedKernel, givenKernelWithIntListOutput_whenRegistered_thenCanBeCalled) {
  // 注册操作符，并指定对应的内核函数
  auto registrar = RegisterOperators()
      .op("_test::list_output(Tensor dummy, int input1, int input2, int input3) -> int[]",
        RegisterOperators::options().kernel(DispatchKey::CPU, [] (const Tensor&, int64_t a, int64_t b, int64_t c) -> c10::List<int64_t> {return c10::List<int64_t>({a,b,c});}));

  // 获取注册后的操作符模式
  auto op = c10::Dispatcher::singleton().findSchema({"_test::list_output", ""});
  // 断言操作符模式存在
  ASSERT_TRUE(op.has_value());

  // 调用操作符，传入参数并接收结果
  auto result = callOp(*op, dummyTensor(DispatchKey::CPU), 2, 4, 6);
  // 断言结果集大小为1
  EXPECT_EQ(1, result.size());
  // 断言第一个结果的整数向量大小为3
  EXPECT_EQ(3, result[0].toIntVector().size());
  // 断言第一个结果的整数向量元素值分别为2、4、6
  EXPECT_EQ(2, result[0].toIntVector()[0]);
  EXPECT_EQ(4, result[0].toIntVector()[1]);
  EXPECT_EQ(6, result[0].toIntVector()[2]);
}

// 测试用例定义：测试带有多个输出的内核注册和调用情况
TEST(OperatorRegistrationTestLambdaBasedKernel, givenKernelWithMultipleOutputs_whenRegistered_thenCanBeCalled) {
  // 注册操作符，并指定对应的内核函数
  auto registrar = RegisterOperators()
     .op("_test::multiple_outputs(Tensor dummy) -> (Tensor, int, Tensor[], int?, Dict(str, Tensor))",
       RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor) -> std::tuple<Tensor, int64_t, c10::List<Tensor>, std::optional<int64_t>, Dict<string, Tensor>> {
         // 创建并填充字典对象
         Dict<string, Tensor> dict;
         dict.insert("first", dummyTensor(DispatchKey::CPU));
         dict.insert("second", dummyTensor(DispatchKey::CUDA));
         // 返回包含多个输出的元组
         return std::tuple<Tensor, int64_t, c10::List<Tensor>, std::optional<int64_t>, Dict<string, Tensor>>(
           dummyTensor(DispatchKey::CUDA),    // 第一个输出：CUDA设备上的张量
           5,                                 // 第二个输出：整数值5
           c10::List<Tensor>({dummyTensor(DispatchKey::CPU), dummyTensor(DispatchKey::CUDA)}),  // 第三个输出：张量列表
           std::optional<int64_t>(std::in_place, 0),   // 第四个输出：可选的整数值
           dict                               // 第五个输出：字典对象
         );
       }));

  // 获取注册后的操作符模式
  auto op = c10::Dispatcher::singleton().findSchema({"_test::multiple_outputs", ""});
  // 断言操作符模式存在
  ASSERT_TRUE(op.has_value());

  // 调用操作符，传入参数并接收结果
  auto result = callOp(*op, dummyTensor(DispatchKey::CPU));
  // 断言结果集大小为5
  EXPECT_EQ(5, result.size());
  // 断言第一个结果的张量的调度键为CUDA
  EXPECT_EQ(DispatchKey::CUDA, extractDispatchKey(result[0].toTensor()));
  // 断言第二个结果的整数值为5
  EXPECT_EQ(5, result[1].toInt());
  // 断言第三个结果的张量列表大小为2
  EXPECT_EQ(2, result[2].toTensorVector().size());
  // 断言第三个结果的张量列表中第一个张量的调度键为CPU，第二个张量的调度键为CUDA
  EXPECT_EQ(DispatchKey::CPU, extractDispatchKey(result[2].toTensorVector()[0]));
  EXPECT_EQ(DispatchKey::CUDA, extractDispatchKey(result[2].toTensorVector()[1]));
  // 断言第四个结果的整数值为0
  EXPECT_EQ(0, result[3].toInt());
  // 将第五个结果转换为类型化字典，并断言其大小为2
  auto result_dict = c10::impl::toTypedDict<string, Tensor>(result[4].toGenericDict());
  EXPECT_EQ(2, result_dict.size());
  // 断言字典中"first"键对应的张量的调度键为CPU，"second"键对应的张量的调度键为CUDA
  EXPECT_EQ(DispatchKey::CPU, extractDispatchKey(result_dict.at("first")));
  EXPECT_EQ(DispatchKey::CUDA, extractDispatchKey(result_dict.at("second")));
}
TEST(OperatorRegistrationTestLambdaBasedKernel, givenKernelWithTensorInputByReference_withOutput_whenRegistered_thenCanBeCalled) {
  // 创建注册器对象，并注册带有引用参数的操作函数
  auto registrar = RegisterOperators()
      .op("_test::tensor_input(Tensor input) -> Tensor",
        // 设置 CPU 和 CUDA 的内核函数，返回输入张量本身
        RegisterOperators::options().kernel(DispatchKey::CPU, [] (const Tensor& a) {return a;})
                                    .kernel(DispatchKey::CUDA, [] (const Tensor& a) {return a;}));

  // 查找指定操作的模式对象
  auto op = c10::Dispatcher::singleton().findSchema({"_test::tensor_input", ""});
  ASSERT_TRUE(op.has_value());  // 断言确保找到了操作的模式对象

  // 调用操作函数，传入 CPU 上的虚拟张量，并检查输出
  auto result = callOp(*op, dummyTensor(DispatchKey::CPU));
  EXPECT_EQ(1, result.size());  // 确保输出结果的大小为 1
  EXPECT_EQ(DispatchKey::CPU, extractDispatchKey(result[0].toTensor()));  // 检查输出张量的调度键为 CPU

  // 以 CUDA 上的虚拟张量再次调用操作函数，并检查输出
  result = callOp(*op, dummyTensor(DispatchKey::CUDA));
  EXPECT_EQ(1, result.size());  // 确保输出结果的大小为 1
  EXPECT_EQ(DispatchKey::CUDA, extractDispatchKey(result[0].toTensor()));  // 检查输出张量的调度键为 CUDA
}

TEST(OperatorRegistrationTestLambdaBasedKernel, givenKernelWithTensorInputByValue_withOutput_whenRegistered_thenCanBeCalled) {
  // 创建注册器对象，并注册带有值参数的操作函数
  auto registrar = RegisterOperators()
      .op("_test::tensor_input(Tensor input) -> Tensor",
        // 设置 CPU 和 CUDA 的内核函数，返回输入张量本身
        RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor a) {return a;})
                                    .kernel(DispatchKey::CUDA, [] (Tensor a) {return a;}));

  // 查找指定操作的模式对象
  auto op = c10::Dispatcher::singleton().findSchema({"_test::tensor_input", ""});
  ASSERT_TRUE(op.has_value());  // 断言确保找到了操作的模式对象

  // 调用操作函数，传入 CPU 上的虚拟张量，并检查输出
  auto result = callOp(*op, dummyTensor(DispatchKey::CPU));
  EXPECT_EQ(1, result.size());  // 确保输出结果的大小为 1
  EXPECT_EQ(DispatchKey::CPU, extractDispatchKey(result[0].toTensor()));  // 检查输出张量的调度键为 CPU

  // 以 CUDA 上的虚拟张量再次调用操作函数，并检查输出
  result = callOp(*op, dummyTensor(DispatchKey::CUDA));
  EXPECT_EQ(1, result.size());  // 确保输出结果的大小为 1
  EXPECT_EQ(DispatchKey::CUDA, extractDispatchKey(result[0].toTensor()));  // 检查输出张量的调度键为 CUDA
}

Tensor captured_input;  // 声明一个全局张量 captured_input，用于捕获输入值

TEST(OperatorRegistrationTestLambdaBasedKernel, givenKernelWithTensorInputByReference_withoutOutput_whenRegistered_thenCanBeCalled) {
  // 创建注册器对象，并注册带有引用参数的操作函数，但没有输出
  auto registrar = RegisterOperators()
      .op("_test::tensor_input(Tensor input) -> ()",
        // 设置 CPU 和 CUDA 的内核函数，用于捕获输入张量到全局变量 captured_input
        RegisterOperators::options().kernel(DispatchKey::CPU, [] (const Tensor& a) -> void {captured_input = a;})
                                    .kernel(DispatchKey::CUDA, [] (const Tensor& a) -> void {captured_input = a;}));

  // 查找指定操作的模式对象
  auto op = c10::Dispatcher::singleton().findSchema({"_test::tensor_input", ""});
  ASSERT_TRUE(op.has_value());  // 断言确保找到了操作的模式对象

  // 调用操作函数，传入 CPU 上的虚拟张量，并检查没有输出
  auto outputs = callOp(*op, dummyTensor(DispatchKey::CPU));
  EXPECT_EQ(0, outputs.size());  // 确保没有输出结果

  // 检查捕获的输入张量的调度键为 CPU
  EXPECT_EQ(DispatchKey::CPU, extractDispatchKey(captured_input));

  // 以 CUDA 上的虚拟张量再次调用操作函数，并检查没有输出
  outputs = callOp(*op, dummyTensor(DispatchKey::CUDA));
  EXPECT_EQ(0, outputs.size());  // 确保没有输出结果

  // 检查捕获的输入张量的调度键为 CUDA
  EXPECT_EQ(DispatchKey::CUDA, extractDispatchKey(captured_input));
}
TEST(OperatorRegistrationTestLambdaBasedKernel, givenKernelWithTensorInputByValue_withoutOutput_whenRegistered_thenCanBeCalled) {
  // 创建操作注册器，并注册具有不返回值的张量输入操作
  auto registrar = RegisterOperators()
      .op("_test::tensor_input(Tensor input) -> ()",
        RegisterOperators::options()
          .kernel(DispatchKey::CPU, [] (Tensor a) -> void {captured_input = a;})
          .kernel(DispatchKey::CUDA, [] (Tensor a) -> void {captured_input = a;}));

  // 查找已注册的操作模式
  auto op = c10::Dispatcher::singleton().findSchema({"_test::tensor_input", ""});
  ASSERT_TRUE(op.has_value());

  // 调用操作并检查输出
  auto outputs = callOp(*op, dummyTensor(DispatchKey::CPU));
  EXPECT_EQ(0, outputs.size());
  EXPECT_EQ(DispatchKey::CPU, extractDispatchKey(captured_input));

  // 再次调用操作并检查输出
  outputs = callOp(*op, dummyTensor(DispatchKey::CUDA));
  EXPECT_EQ(0, outputs.size());
  EXPECT_EQ(DispatchKey::CUDA, extractDispatchKey(captured_input));
}

int64_t captured_int_input = 0;

TEST(OperatorRegistrationTestLambdaBasedKernel, givenKernelWithIntInput_withoutOutput_whenRegistered_thenCanBeCalled) {
  // 创建操作注册器，并注册具有整数输入但不返回值的操作
  auto registrar = RegisterOperators()
      .op("_test::int_input(Tensor dummy, int input) -> ()",
        RegisterOperators::options()
          .kernel(DispatchKey::CPU, [] (Tensor, int64_t a) -> void {captured_int_input = a;}));

  // 查找已注册的操作模式
  auto op = c10::Dispatcher::singleton().findSchema({"_test::int_input", ""});
  ASSERT_TRUE(op.has_value());

  captured_int_input = 0;
  // 调用操作并检查输出
  auto outputs = callOp(*op, dummyTensor(DispatchKey::CPU), 3);
  EXPECT_EQ(0, outputs.size());
  EXPECT_EQ(3, captured_int_input);
}

TEST(OperatorRegistrationTestLambdaBasedKernel, givenKernelWithIntInput_withOutput_whenRegistered_thenCanBeCalled) {
  // 创建操作注册器，并注册具有整数输入和返回整数的操作
  auto registrar = RegisterOperators()
      .op("_test::int_input(Tensor dummy, int input) -> int",
        RegisterOperators::options()
          .kernel(DispatchKey::CPU, [] (Tensor, int64_t a) {return a + 1;}));

  // 查找已注册的操作模式
  auto op = c10::Dispatcher::singleton().findSchema({"_test::int_input", ""});
  ASSERT_TRUE(op.has_value());

  // 调用操作并检查输出
  auto outputs = callOp(*op, dummyTensor(DispatchKey::CPU), 3);
  EXPECT_EQ(1, outputs.size());
  EXPECT_EQ(4, outputs[0].toInt());
}

int64_t captured_input_list_size = 0;

TEST(OperatorRegistrationTestLambdaBasedKernel, givenKernelWithIntListInput_withoutOutput_whenRegistered_thenCanBeCalled) {
  // 创建操作注册器，并注册具有整数列表输入但不返回值的操作
  auto registrar = RegisterOperators()
      .op("_test::int_list_input(Tensor dummy, int[] input) -> ()",
        RegisterOperators::options()
          .kernel(DispatchKey::CPU, [] (Tensor, const c10::List<int64_t>& a) {captured_input_list_size = a.size();}));

  // 查找已注册的操作模式
  auto op = c10::Dispatcher::singleton().findSchema({"_test::int_list_input", ""});
  ASSERT_TRUE(op.has_value());

  captured_input_list_size = 0;
  // 调用操作并检查输出
  auto outputs = callOp(*op, dummyTensor(DispatchKey::CPU), c10::List<int64_t>({2, 4, 6}));
  EXPECT_EQ(0, outputs.size());
  EXPECT_EQ(3, captured_input_list_size);
}
TEST(OperatorRegistrationTestLambdaBasedKernel, givenKernelWithIntListInput_withOutput_whenRegistered_thenCanBeCalled) {
  // 创建操作符注册器对象，注册具有整数列表输入和输出的操作符
  auto registrar = RegisterOperators()
      .op("_test::int_list_input(Tensor dummy, int[] input) -> int",
        RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor, const c10::List<int64_t>& a) -> int64_t {return a.size();}));

  // 查找已注册的操作符架构
  auto op = c10::Dispatcher::singleton().findSchema({"_test::int_list_input", ""});
  // 断言确保找到操作符
  ASSERT_TRUE(op.has_value());

  // 调用注册的操作符，并获取输出
  auto outputs = callOp(*op, dummyTensor(DispatchKey::CPU), c10::List<int64_t>({2, 4, 6}));
  // 断言输出的数量为1
  EXPECT_EQ(1, outputs.size());
  // 断言第一个输出的值为3
  EXPECT_EQ(3, outputs[0].toInt());
}

TEST(OperatorRegistrationTestLambdaBasedKernel, givenKernelWithTensorListInput_withoutOutput_whenRegistered_thenCanBeCalled) {
  // 创建操作符注册器对象，注册具有张量列表输入但没有输出的操作符
  auto registrar = RegisterOperators()
      .op("_test::tensor_list_input(Tensor[] input) -> ()",
        RegisterOperators::options().kernel(DispatchKey::CPU, [] (const c10::List<Tensor>& a) -> void {captured_input_list_size = a.size();}));

  // 查找已注册的操作符架构
  auto op = c10::Dispatcher::singleton().findSchema({"_test::tensor_list_input", ""});
  // 断言确保找到操作符
  ASSERT_TRUE(op.has_value());

  // 初始化捕获的输入列表大小
  captured_input_list_size = 0;
  // 调用注册的操作符，并获取输出
  auto outputs = callOp(*op, c10::List<Tensor>({dummyTensor(DispatchKey::CPU), dummyTensor(DispatchKey::CPU)}));
  // 断言输出的数量为0
  EXPECT_EQ(0, outputs.size());
  // 断言捕获的输入列表大小为2
  EXPECT_EQ(2, captured_input_list_size);
}

TEST(OperatorRegistrationTestLambdaBasedKernel, givenKernelWithTensorListInput_withOutput_whenRegistered_thenCanBeCalled) {
  // 创建操作符注册器对象，注册具有张量列表输入和整数输出的操作符
  auto registrar = RegisterOperators()
      .op("_test::tensor_list_input(Tensor[] input) -> int",
         RegisterOperators::options().kernel(DispatchKey::CPU, [] (const c10::List<Tensor>& a) -> int64_t {return a.size();}));

  // 查找已注册的操作符架构
  auto op = c10::Dispatcher::singleton().findSchema({"_test::tensor_list_input", ""});
  // 断言确保找到操作符
  ASSERT_TRUE(op.has_value());

  // 调用注册的操作符，并获取输出
  auto outputs = callOp(*op, c10::List<Tensor>({dummyTensor(DispatchKey::CPU), dummyTensor(DispatchKey::CPU)}));
  // 断言输出的数量为1
  EXPECT_EQ(1, outputs.size());
  // 断言输出的值为2
  EXPECT_EQ(2, outputs[0].toInt());
}

int captured_dict_size = 0;

TEST(OperatorRegistrationTestLambdaBasedKernel, givenKernelWithDictInput_withoutOutput_whenRegistered_thenCanBeCalled) {
  // 创建操作符注册器对象，注册具有字典输入但没有输出的操作符
  auto registrar = RegisterOperators()
      .op("_test::dict_input(Dict(str, Tensor) input) -> ()", RegisterOperators::options().catchAllKernel([] (Dict<string, Tensor> input1) {
        captured_dict_size = input1.size();
      }));

  // 查找已注册的操作符架构
  auto op = c10::Dispatcher::singleton().findSchema({"_test::dict_input", ""});
  // 断言确保找到操作符
  ASSERT_TRUE(op.has_value());

  // 初始化捕获的字典大小
  captured_dict_size = 0;
  // 创建测试用的字典对象
  Dict<string, Tensor> dict;
  dict.insert("key1", dummyTensor(DispatchKey::CPU));
  dict.insert("key2", dummyTensor(DispatchKey::CUDA));
  // 调用注册的操作符，并获取输出
  auto outputs = callOp(*op, dict);
  // 断言输出的数量为0
  EXPECT_EQ(0, outputs.size());
  // 断言捕获的字典大小为2
  EXPECT_EQ(2, captured_dict_size);
}
TEST(OperatorRegistrationTestLambdaBasedKernel, givenKernelWithDictInput_withOutput_whenRegistered_thenCanBeCalled) {
  // 创建操作符注册器，注册接受字典输入和返回字符串输出的操作符
  auto registrar = RegisterOperators()
      .op("_test::dict_input(Dict(str, str) input) -> str", RegisterOperators::options().catchAllKernel([] (Dict<string, string> input1) {
        // 在输入字典中查找并返回 "key2" 对应的值
        return input1.at("key2");
      }));

  // 查找已注册的操作符模式
  auto op = c10::Dispatcher::singleton().findSchema({"_test::dict_input", ""});
  ASSERT_TRUE(op.has_value());

  // 创建一个包含键值对 "key1"="value1" 和 "key2"="value2" 的字典
  Dict<string, string> dict;
  dict.insert("key1", "value1");
  dict.insert("key2", "value2");
  // 调用操作符，并期望返回单个输出
  auto outputs = callOp(*op, dict);
  EXPECT_EQ(1, outputs.size());
  EXPECT_EQ("value2", outputs[0].toStringRef());
}

TEST(OperatorRegistrationTestLambdaBasedKernel, givenKernelWithDictOutput_whenRegistered_thenCanBeCalled) {
  // 创建操作符注册器，注册接受字典输入并返回字典输出的操作符
  auto registrar = RegisterOperators()
    .op("_test::dict_output(Dict(str, str) input) -> Dict(str, str)", RegisterOperators::options().catchAllKernel([] (Dict<string, string> input) {
      // 直接返回输入字典作为输出
      return input;
    }));

  // 查找已注册的操作符模式
  auto op = c10::Dispatcher::singleton().findSchema({"_test::dict_output", ""});
  ASSERT_TRUE(op.has_value());

  // 创建一个包含键值对 "key1"="value1" 和 "key2"="value2" 的字典
  Dict<string, string> dict;
  dict.insert("key1", "value1");
  dict.insert("key2", "value2");
  // 调用操作符，并期望返回单个输出
  auto outputs = callOp(*op, dict);
  EXPECT_EQ(1, outputs.size());
  // 将输出转换为键值对形式的字典，并期望包含两个键值对
  auto output = c10::impl::toTypedDict<string, string>(outputs[0].toGenericDict());
  EXPECT_EQ(2, output.size());
  EXPECT_EQ("value1", output.at("key1"));
  EXPECT_EQ("value2", output.at("key2"));
}

bool called = false;

TEST(OperatorRegistrationTestLambdaBasedKernel, givenFallbackKernelWithoutAnyArguments_whenRegistered_thenCanBeCalled) {
  // 注意: 没有张量参数的非回退内核无法工作，因为没有办法获取调度键。
  // 对于只有回退内核的操作符，这种方式必须适用于向后兼容性。
  // 创建操作符注册器，注册不带任何参数的操作符并设置回退内核，只需设置一个标志为 true
  auto registrar = RegisterOperators()
      .op("_test::no_tensor_args() -> ()", RegisterOperators::options().catchAllKernel([] () {called = true;}));

  // 查找已注册的操作符模式
  auto op = c10::Dispatcher::singleton().findSchema({"_test::no_tensor_args", ""});
  ASSERT_TRUE(op.has_value());

  // 重置标志
  called = false;
  // 调用操作符，期望设置标志为 true
  auto outputs = callOp(*op);
  EXPECT_TRUE(called);
}

TEST(OperatorRegistrationTestLambdaBasedKernel, givenFallbackKernelWithoutTensorArguments_whenRegistered_thenCanBeCalled) {
  // 注意: 没有张量参数的非回退内核无法工作，因为没有办法获取调度键。
  // 对于只有回退内核的操作符，这种方式必须适用于向后兼容性。
  // 创建操作符注册器，注册不带张量参数的操作符并设置回退内核，增加输入参数值并返回值加 1
  auto registrar = RegisterOperators()
      .op("_test::no_tensor_args(int arg) -> int", RegisterOperators::options().catchAllKernel([] (int64_t arg) {return arg + 1;}));

  // 查找已注册的操作符模式
  auto op = c10::Dispatcher::singleton().findSchema({"_test::no_tensor_args", ""});
  ASSERT_TRUE(op.has_value());

  // 调用操作符，传递参数 3，期望返回值列表的第一个元素为 4
  auto outputs = callOp(*op, 3);
  EXPECT_EQ(1, outputs.size());
  EXPECT_EQ(4, outputs[0].toInt());
}

std::optional<Tensor> called_arg2 = c10::nullopt;
// 初始化一个可选的 int64_t 类型变量，未指定初值
std::optional<int64_t> called_arg3 = c10::nullopt;
// 初始化一个可选的 std::string 类型变量，未指定初值
std::optional<std::string> called_arg4 = c10::nullopt;

// 定义测试用例 OperatorRegistrationTestLambdaBasedKernel，测试具有可选输入但无输出的内核注册和调用情况
TEST(OperatorRegistrationTestLambdaBasedKernel, givenKernelWithOptionalInputs_withoutOutput_whenRegistered_thenCanBeCalled) {
  // 注册一个操作符，指定操作符的签名和相应的内核函数
  auto registrar = RegisterOperators().op(
    "_test::opt_input(Tensor arg1, Tensor? arg2, int? arg3, str? arg4) -> ()",
    RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor arg1, const std::optional<Tensor>& arg2, std::optional<int64_t> arg3, std::optional<std::string> arg4) {
      // 设置一个标志表示内核函数被调用
      called = true;
      // 记录被传入的可选参数值
      called_arg2 = arg2;
      called_arg3 = arg3;
      called_arg4 = arg4;
    }));

  // 查找注册的操作符模式，确定其是否存在
  auto op = c10::Dispatcher::singleton().findSchema({"_test::opt_input", ""});
  ASSERT_TRUE(op.has_value());

  // 重置调用标志
  called = false;
  // 调用操作符，并检查返回的输出
  auto outputs = callOp(*op, dummyTensor(DispatchKey::CPU), dummyTensor(DispatchKey::CPU), c10::IValue(), std::string("text"));
  EXPECT_EQ(0, outputs.size());

  // 断言内核函数确实被调用
  EXPECT_TRUE(called);
  // 断言可选参数 arg2 的值已经被设置
  EXPECT_TRUE(called_arg2.has_value());
  // 断言 arg2 的分发键是 DispatchKey::CPU
  EXPECT_EQ(extractDispatchKey(*called_arg2), DispatchKey::CPU);
  // 断言可选参数 arg3 的值未被设置
  EXPECT_FALSE(called_arg3.has_value());
  // 断言可选参数 arg4 的值已经被设置
  EXPECT_TRUE(called_arg4.has_value());
  // 断言 arg4 的值等于 "text"
  EXPECT_EQ(*called_arg4, "text");

  // 重置调用标志
  called = false;
  // 再次调用操作符，并检查返回的输出
  outputs = callOp(*op, dummyTensor(DispatchKey::CPU), c10::IValue(), 4, c10::IValue());
  EXPECT_EQ(0, outputs.size());

  // 断言内核函数确实被调用
  EXPECT_TRUE(called);
  // 断言可选参数 arg2 的值未被设置
  EXPECT_FALSE(called_arg2.has_value());
  // 断言可选参数 arg3 的值已经被设置
  EXPECT_TRUE(called_arg3.has_value());
  // 断言 arg3 的值等于 4
  EXPECT_EQ(*called_arg3, 4);
  // 断言可选参数 arg4 的值未被设置
  EXPECT_FALSE(called_arg4.has_value());
}

// 定义测试用例 OperatorRegistrationTestLambdaBasedKernel，测试具有可选输入和输出的内核注册和调用情况
TEST(OperatorRegistrationTestLambdaBasedKernel, givenKernelWithOptionalInputs_withOutput_whenRegistered_thenCanBeCalled) {
  // 注册一个操作符，指定操作符的签名、相应的内核函数以及输出
  auto registrar = RegisterOperators().op(
    "_test::opt_input(Tensor arg1, Tensor? arg2, int? arg3, str? arg4) -> Tensor?",
    RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor arg1, const std::optional<Tensor>& arg2, std::optional<int64_t> arg3, std::optional<std::string> arg4) {
      // 设置一个标志表示内核函数被调用
      called = true;
      // 记录被传入的可选参数值
      called_arg2 = arg2;
      called_arg3 = arg3;
      called_arg4 = arg4;
      // 返回 arg2 作为输出
      return arg2;
    }));
    }));

该行代码结尾，未能提供上下文信息，需要更多上下文才能提供准确注释。


  auto op = c10::Dispatcher::singleton().findSchema({"_test::opt_input", ""});

从调度程序中查找名称为 "_test::opt_input" 的模式，并将结果存储在 op 中。


  ASSERT_TRUE(op.has_value());

断言确保 op 中有有效值，即成功找到了 "_test::opt_input" 对应的模式。


  called = false;

将变量 called 设置为 false，用于后续的状态跟踪。


  auto outputs = callOp(*op, dummyTensor(DispatchKey::CPU), dummyTensor(DispatchKey::CPU), c10::IValue(), std::string("text"));

调用 callOp 函数，使用 op 模式对象以及其他参数来生成输出，并将结果存储在 outputs 中。


  EXPECT_EQ(1, outputs.size());

断言确保 outputs 中的元素数量为 1。


  EXPECT_EQ(DispatchKey::CPU, extractDispatchKey(outputs[0].toTensor()));

断言确保 outputs 的第一个元素的分发键为 DispatchKey::CPU。


  EXPECT_TRUE(called);

断言确保 called 变量为 true，表示 callOp 函数已被调用。


  EXPECT_TRUE(called_arg2.has_value());

断言确保 called_arg2 变量有值。


  EXPECT_EQ(extractDispatchKey(*called_arg2), DispatchKey::CPU);

断言确保 called_arg2 的值的分发键为 DispatchKey::CPU。


  EXPECT_FALSE(called_arg3.has_value());

断言确保 called_arg3 变量没有值。


  EXPECT_TRUE(called_arg4.has_value());

断言确保 called_arg4 变量有值。


  EXPECT_EQ(*called_arg4, "text");

断言确保 called_arg4 的值等于 "text"。


  called = false;

将 called 变量重新设置为 false，用于下一次状态跟踪。


  outputs = callOp(*op, dummyTensor(DispatchKey::CPU), c10::IValue(), 4, c10::IValue());

再次调用 callOp 函数，使用 op 模式对象和其他参数来生成输出，并将结果存储在 outputs 中。


  EXPECT_EQ(1, outputs.size());

断言确保 outputs 中的元素数量为 1。


  EXPECT_TRUE(outputs[0].isNone());

断言确保 outputs 的第一个元素是 None 类型。


  EXPECT_TRUE(called);

断言确保 called 变量为 true，表示 callOp 函数再次被调用。


  EXPECT_FALSE(called_arg2.has_value());

断言确保 called_arg2 变量没有值。


  EXPECT_TRUE(called_arg3.has_value());

断言确保 called_arg3 变量有值。


  EXPECT_EQ(*called_arg3, 4);

断言确保 called_arg3 的值为 4。


  EXPECT_FALSE(called_arg4.has_value());

断言确保 called_arg4 变量没有值。
TEST(OperatorRegistrationTestLambdaBasedKernel, givenKernelWithOptionalInputs_withMultipleOutputs_whenRegistered_thenCanBeCalled) {
  // 创建一个操作注册器，并注册一个具有可选输入和多个输出的内核函数
  auto registrar = RegisterOperators().op(
    "_test::opt_input(Tensor arg1, Tensor? arg2, int? arg3, str? arg4) -> (Tensor?, int?, str?)",
    RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor arg1, const std::optional<Tensor>& arg2, std::optional<int64_t> arg3, std::optional<std::string> arg4) {
      // 在 lambda 函数中，返回一个包含所有可选输入的元组
      return std::make_tuple(arg2, arg3, arg4);
    }));
  // 查找并验证刚注册的操作模式是否存在
  auto op = c10::Dispatcher::singleton().findSchema({"_test::opt_input", ""});
  ASSERT_TRUE(op.has_value());

  // 调用注册的操作模式，使用虚拟数据，并验证输出结果的数量和内容
  auto outputs = callOp(*op, dummyTensor(DispatchKey::CPU), dummyTensor(DispatchKey::CPU), c10::IValue(), std::string("text"));
  EXPECT_EQ(3, outputs.size());
  EXPECT_EQ(DispatchKey::CPU, extractDispatchKey(outputs[0].toTensor()));
  EXPECT_TRUE(outputs[1].isNone());
  EXPECT_EQ("text", outputs[2].toStringRef());

  // 使用不同的虚拟数据再次调用注册的操作模式，并验证输出结果的数量和内容
  outputs = callOp(*op, dummyTensor(DispatchKey::CPU), c10::IValue(), 4, c10::IValue());
  EXPECT_EQ(3, outputs.size());
  EXPECT_TRUE(outputs[0].isNone());
  EXPECT_EQ(4, outputs[1].toInt());
  EXPECT_TRUE(outputs[2].isNone());
}

void expectCallsConcatUnboxed(DispatchKey dispatch_key) {
  // 进入自动调度以下自动梯度模式
  at::AutoDispatchBelowAutograd mode;

  // 断言指定的模式和 CPU 内核是否存在
  auto op = c10::Dispatcher::singleton().findSchema({"_test::my_op", ""});
  ASSERT_TRUE(op.has_value());
  // 调用非装箱版本的操作，并验证返回的字符串结果
  std::string result = callOpUnboxed<std::string, const Tensor&, std::string, const std::string&, int64_t>(*op, dummyTensor(dispatch_key), "1", "2", 3);
  EXPECT_EQ("123", result);
}

TEST(OperatorRegistrationTestLambdaBasedKernel, givenKernel_whenRegistered_thenCanBeCalledUnboxed) {
  // 创建一个操作注册器，并注册一个带有 CPU 内核的操作
  auto registrar = RegisterOperators().op("_test::my_op(Tensor dummy, str a, str b, int c) -> str", torch::RegisterOperators::options()
    .kernel(DispatchKey::CPU, [] (const Tensor& tensor1, std::string a, const std::string& b, int64_t c) {
      // 在 lambda 函数中返回字符串连接结果
      return a + b + std::to_string(c);
    }));
  // 调用一个期望执行非装箱调用的函数，使用 CPU 分发密钥
  expectCallsConcatUnboxed(DispatchKey::CPU);
}

TEST(OperatorRegistrationTestLambdaBasedKernel, givenKernel_whenRegisteredWithoutSpecifyingSchema_thenInfersSchema) {
  // 创建一个操作注册器，并注册一个没有显式指定模式的操作
  auto registrar = RegisterOperators()
      .op("_test::no_schema_specified", RegisterOperators::options().catchAllKernel([] (Tensor arg1, int64_t arg2, const c10::List<Tensor>& arg3) -> std::tuple<int64_t, Tensor> {return {};}));

  // 查找并验证刚注册的操作模式是否存在
  auto op = c10::Dispatcher::singleton().findSchema({"_test::no_schema_specified", ""});
  ASSERT_TRUE(op.has_value());

  // 检查推断的模式和手动解析的模式之间的差异，期望没有差异
  std::optional<std::string> differences = c10::findSchemaDifferences(torch::jit::parseSchema("_test::no_schema_specified(Tensor arg1, int arg2, Tensor[] arg3) -> (int, Tensor)"), op->schema());
  EXPECT_FALSE(differences.has_value());
}
TEST(OperatorRegistrationTestLambdaBasedKernel, givenMismatchedKernel_withDifferentNumArguments_whenRegistering_thenFails) {
  // 注册一个操作符，定义了一个带有一个参数的内核函数
  RegisterOperators()
      .op("_test::mismatch(Tensor arg) -> int", RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor) -> int64_t {return {};}));

  // 预期会抛出异常，因为注册的操作符定义与内核函数的参数个数不匹配
  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg, Tensor arg2) -> int", RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor) -> int64_t {return {};}));
    }, "The number of arguments is different. 2 vs 1"
  );

  // 注册一个操作符，定义了一个带有两个参数的内核函数
  RegisterOperators()
      .op("_test::mismatch(Tensor arg, Tensor arg2) -> ()", RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor, Tensor) -> void {}));

  // 预期会抛出异常，因为注册的操作符定义与内核函数的参数个数不匹配
  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch() -> ()", RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor, Tensor) -> void {}));
    }, "The number of arguments is different. 0 vs 2"
  );

  // 预期会抛出异常，因为注册的操作符定义与内核函数的参数个数不匹配
  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> ()", RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor, Tensor) -> void {}));
    }, "The number of arguments is different. 1 vs 2"
  );

  // 预期会抛出异常，因为注册的操作符定义与内核函数的参数个数不匹配
  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg, Tensor arg2, Tensor arg3) -> ()", RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor, Tensor) -> void {}));
    }, "The number of arguments is different. 3 vs 2"
  );
}

TEST(OperatorRegistrationTestLambdaBasedKernel, givenMismatchedKernel_withDifferentArgumentType_whenRegistering_thenFails) {
  // 注册一个操作符，定义了一个带有一个 Tensor 和一个 int 参数的内核函数
  RegisterOperators()
      .op("_test::mismatch(Tensor arg1, int arg2) -> int", RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor, int64_t) -> int64_t {return {};}));

  // 预期会抛出异常，因为注册的操作符定义中第二个参数的类型与内核函数中不匹配
  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg1, float arg2) -> int", RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor, int64_t) -> int64_t {return {};}));
    }, "Type mismatch in argument 2: float vs int"
  );

  // 预期会抛出异常，因为注册的操作符定义中第一个参数的类型与内核函数中不匹配
  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(int arg1, int arg2) -> int", RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor, int64_t) -> int64_t {return {};}));
    }, "Type mismatch in argument 1: int vs Tensor"
  );
}
// 定义测试用例 OperatorRegistrationTestLambdaBasedKernel.givenMismatchedKernel_withDifferentNumReturns_whenRegistering_thenFails
TEST(OperatorRegistrationTestLambdaBasedKernel, givenMismatchedKernel_withDifferentNumReturns_whenRegistering_thenFails) {
  // 断言这不会失败，因为匹配成功
  RegisterOperators()
      .op("_test::mismatch(Tensor arg) -> int", RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor) -> int64_t {return {};}));

  // 现在是一组不匹配的函数签名
  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> ()", RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor) -> int64_t {return {};}));
    }, "The number of returns is different. 0 vs 1"
  );

  // 现在是另一组不匹配的函数签名
  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> (int, int)", RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor) -> int64_t {return {};}));
    }, "The number of returns is different. 2 vs 1"
  );

  // 断言这不会失败，因为匹配成功
  RegisterOperators()
      .op("_test::mismatch(Tensor arg) -> ()", RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor) -> void {}));

  // 现在是一组不匹配的函数签名
  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> Tensor", RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor) -> void {}));
    }, "The number of returns is different. 1 vs 0"
  );

  // 现在是另一组不匹配的函数签名
  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> (Tensor, Tensor)", RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor) -> void {}));
    }, "The number of returns is different. 2 vs 0"
  );

  // 断言这不会失败，因为匹配成功
  RegisterOperators()
      .op("_test::mismatch(Tensor arg) -> (Tensor, Tensor)", RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor) -> std::tuple<Tensor, Tensor> {return {};}));

  // 现在是一组不匹配的函数签名
  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> ()", RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor) -> std::tuple<Tensor, Tensor> {return {};}));
    }, "The number of returns is different. 0 vs 2"
  );

  // 现在是另一组不匹配的函数签名
  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> Tensor", RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor) -> std::tuple<Tensor, Tensor> {return {};}));
    }, "The number of returns is different. 1 vs 2"
  );

  // 最后一组不匹配的函数签名
  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> (Tensor, Tensor, Tensor)", RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor) -> std::tuple<Tensor, Tensor> {return {};}));
    }, "The number of returns is different. 3 vs 2"
  );
}
TEST(OperatorRegistrationTestLambdaBasedKernel, givenMismatchedKernel_withDifferentReturnTypes_whenRegistering_thenFails) {
  // 注册一个操作符，其输入为 Tensor 类型，输出为 int 类型的内核函数
  RegisterOperators()
      .op("_test::mismatch(Tensor arg) -> int", RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor) -> int64_t {return {};}));

  // 预期抛出 c10::Error 异常，因为返回类型不匹配
  expectThrows<c10::Error>([] {
    // 注册一个操作符，其输入为 Tensor 类型，输出为 Tensor 类型的内核函数
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> Tensor", RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor) -> int64_t {return {};}));
    }, "Type mismatch in return 1: Tensor vs int"
  );

  // 预期抛出 c10::Error 异常，因为返回类型不匹配
  expectThrows<c10::Error>([] {
    // 注册一个操作符，其输入为 Tensor 类型，输出为 float 类型的内核函数
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> float", RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor) -> int64_t {return {};}));
    }, "Type mismatch in return 1: float vs int"
  );

  // 注册一个操作符，其输入为 Tensor 类型，输出为 Tensor 类型的内核函数
  RegisterOperators()
      .op("_test::mismatch(Tensor arg) -> Tensor", RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor) -> Tensor {return {};}));

  // 预期抛出 c10::Error 异常，因为返回类型不匹配
  expectThrows<c10::Error>([] {
    // 注册一个操作符，其输入为 Tensor 类型，输出为 float 类型的内核函数
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> float", RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor) -> Tensor {return {};}));
    }, "Type mismatch in return 1: float vs Tensor"
  );

  // 注册一个操作符，其输入为 Tensor 类型，输出为 (Tensor, int) 元组的内核函数
  RegisterOperators()
      .op("_test::mismatch(Tensor arg) -> (Tensor, int)", RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor) -> std::tuple<Tensor, int64_t> {return {};}));

  // 预期抛出 c10::Error 异常，因为返回类型不匹配
  expectThrows<c10::Error>([] {
    // 注册一个操作符，其输入为 Tensor 类型，输出为 (Tensor, float) 元组的内核函数
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> (Tensor, float)", RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor) -> std::tuple<Tensor, int64_t> {return {};}));
    }, "Type mismatch in return 2: float vs int"
  );

  // 预期抛出 c10::Error 异常，因为返回类型不匹配
  expectThrows<c10::Error>([] {
    // 注册一个操作符，其输入为 Tensor 类型，输出为 (int, int) 元组的内核函数
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> (int, int)", RegisterOperators::options().kernel(DispatchKey::CPU, [] (Tensor) -> std::tuple<Tensor, int64_t> {return {};}));
    }, "Type mismatch in return 1: int vs Tensor"
  );
}
```