# `.\pytorch\aten\src\ATen\core\boxing\impl\make_boxed_from_unboxed_functor_test.cpp`

```py
#include <gtest/gtest.h>
#include <ATen/core/boxing/impl/test_helpers.h>

#include <ATen/core/op_registration/op_registration.h>
#include <ATen/core/Tensor.h>
#include <torch/csrc/jit/frontend/function_schema_parser.h>
#include <torch/library.h>

#include <ATen/core/LegacyTypeDispatch.h>

using c10::RegisterOperators;
using c10::OperatorKernel;
using c10::DispatchKey;
using c10::Stack;
using std::make_unique;
using c10::intrusive_ptr;
using c10::Dict;
using at::Tensor;
using std::unique_ptr;
using std::string;

namespace {

// 自定义操作核心，用于测试错误情况
struct ErrorKernel final : public OperatorKernel {
  // 当前操作核心的执行函数，断言永远不应该调用这个核心
  int64_t operator()(const Tensor&, int64_t) {
    EXPECT_TRUE(false); // this kernel should never be called
    return 0;
  }
};

// 自增操作核心
struct IncrementKernel final : OperatorKernel {
  // 自增操作的执行函数，返回输入值加一
  int64_t operator()(const Tensor& tensor, int64_t input) {
    return input + 1;
  }
};

// 自减操作核心
struct DecrementKernel final : OperatorKernel {
  // 自减操作的执行函数，返回输入值减一
  int64_t operator()(const Tensor& tensor, int64_t input) {
    return input - 1;
  }
};

// 辅助函数，验证调用增量操作是否成功
void expectCallsIncrement(DispatchKey dispatch_key) {
  at::AutoDispatchBelowAutograd mode;

  // 断言操作模式和 CPU 核心已注册
  auto op = c10::Dispatcher::singleton().findSchema({"_test::my_op", ""});
  ASSERT_TRUE(op.has_value());
  
  // 调用注册的操作，期望返回一个值，验证增量操作结果
  auto result = callOp(*op, dummyTensor(dispatch_key), 5);
  EXPECT_EQ(1, result.size());
  EXPECT_EQ(6, result[0].toInt());
}

// 测试用例，验证基于函数对象的操作核心注册后能否被正确调用
TEST(OperatorRegistrationTestFunctorBasedKernel, givenKernel_whenRegistered_thenCanBeCalled) {
  // 注册增量操作核心，并通过给定的 CPU 调度键注册操作
  auto registrar = RegisterOperators().op("_test::my_op(Tensor dummy, int input) -> int", RegisterOperators::options().kernel<IncrementKernel>(DispatchKey::CPU));
  expectCallsIncrement(DispatchKey::CPU);
}

// 测试用例，验证多个操作核心和操作在一个注册器中注册后，能够正确调用指定的核心
TEST(OperatorRegistrationTestFunctorBasedKernel, givenMultipleOperatorsAndKernels_whenRegisteredInOneRegistrar_thenCallsRightKernel) {
  // 注册增量和错误操作核心，并分别通过 CPU 和 CUDA 调度键注册两个不同的操作
  auto registrar = RegisterOperators()
      .op("_test::my_op(Tensor dummy, int input) -> int", RegisterOperators::options().kernel<IncrementKernel>(DispatchKey::CPU)
                                                                                      .kernel<ErrorKernel>(DispatchKey::CUDA))
      .op("_test::error(Tensor dummy, int input) -> int", RegisterOperators::options().kernel<ErrorKernel>(DispatchKey::CPU)
                                                                                      .kernel<ErrorKernel>(DispatchKey::CUDA));
  expectCallsIncrement(DispatchKey::CPU);
}
// 定义测试函数 OperatorRegistrationTestFunctorBasedKernel，测试多个运算符和内核的注册，验证在多个注册器中注册时调用正确的内核
TEST(OperatorRegistrationTestFunctorBasedKernel, givenMultipleOperatorsAndKernels_whenRegisteredInMultipleRegistrars_thenCallsRightKernel) {
  // 第一个运算符注册，注册一个名为 "_test::my_op" 的操作符，接受一个 Tensor 和一个 int 参数，并返回 int
  auto registrar1 = RegisterOperators().op("_test::my_op(Tensor dummy, int input) -> int",
                                           RegisterOperators::options()
                                             .kernel<IncrementKernel>(DispatchKey::CPU)  // 使用 IncrementKernel 作为 CPU 分发键的内核
                                             .kernel<ErrorKernel>(DispatchKey::CUDA));    // 使用 ErrorKernel 作为 CUDA 分发键的内核
  
  // 第二个运算符注册，注册一个名为 "_test::error" 的操作符，接受一个 Tensor 和一个 int 参数，并返回 int
  auto registrar2 = RegisterOperators().op("_test::error(Tensor dummy, int input) -> int",
                                           RegisterOperators::options()
                                             .kernel<ErrorKernel>(DispatchKey::CPU)      // 使用 ErrorKernel 作为 CPU 分发键的内核
                                             .kernel<ErrorKernel>(DispatchKey::CUDA));    // 使用 ErrorKernel 作为 CUDA 分发键的内核
  
  // 预期调用增量内核的函数，使用 CPU 分发键
  expectCallsIncrement(DispatchKey::CPU);
}

// 定义全局变量 was_called，用于标记是否调用过某个函数
bool was_called = false;

// 定义结构体 KernelWithoutOutput，实现 OperatorKernel 接口，用于没有输出的内核操作
struct KernelWithoutOutput final : OperatorKernel {
  // 实现操作符重载，接受一个 Tensor 参数，标记 was_called 为 true
  void operator()(const Tensor&) {
    was_called = true;
  }
};

// 测试 KernelWithoutOutput 内核的注册和调用
TEST(OperatorRegistrationTestFunctorBasedKernel, givenKernelWithoutOutput_whenRegistered_thenCanBeCalled) {
  // 注册一个名为 "_test::no_return" 的操作符，接受一个 Tensor 参数，并且没有返回值
  auto registrar = RegisterOperators().op("_test::no_return(Tensor dummy) -> ()",
                                          RegisterOperators::options()
                                            .kernel<KernelWithoutOutput>(DispatchKey::CPU));  // 使用 KernelWithoutOutput 作为 CPU 分发键的内核
  
  // 查找名为 "_test::no_return" 的操作符模式
  auto op = c10::Dispatcher::singleton().findSchema({"_test::no_return", ""});
  ASSERT_TRUE(op.has_value());  // 断言确保找到了操作符模式
  
  was_called = false;  // 重置 was_called 标志位
  // 调用名为 "_test::no_return" 的操作符，传入一个虚拟的 Tensor，期望 was_called 被设置为 true
  auto result = callOp(*op, dummyTensor(DispatchKey::CPU));
  EXPECT_TRUE(was_called);  // 验证 was_called 被正确设置为 true
  EXPECT_EQ(0, result.size());  // 验证结果集的大小为 0
}

// 定义结构体 KernelWithZeroOutputs，实现 OperatorKernel 接口，用于有零输出的内核操作
struct KernelWithZeroOutputs final : OperatorKernel {
  // 实现操作符重载，接受一个 Tensor 参数，标记 was_called 为 true，并返回一个空的 std::tuple
  std::tuple<> operator()(const Tensor&) {
    was_called = true;
    return std::make_tuple();
  }
};

// 测试 KernelWithZeroOutputs 内核的注册和调用
TEST(OperatorRegistrationTestFunctorBasedKernel, givenKernelWithZeroOutputs_whenRegistered_thenCanBeCalled) {
  // 注册一个名为 "_test::zero_outputs" 的操作符，接受一个 Tensor 参数，并且返回值为空的 std::tuple
  auto registrar = RegisterOperators().op("_test::zero_outputs(Tensor dummy) -> ()",
                                          RegisterOperators::options()
                                            .kernel<KernelWithZeroOutputs>(DispatchKey::CPU));  // 使用 KernelWithZeroOutputs 作为 CPU 分发键的内核
  
  // 查找名为 "_test::zero_outputs" 的操作符模式
  auto op = c10::Dispatcher::singleton().findSchema({"_test::zero_outputs", ""});
  ASSERT_TRUE(op.has_value());  // 断言确保找到了操作符模式
  
  was_called = false;  // 重置 was_called 标志位
  // 调用名为 "_test::zero_outputs" 的操作符，传入一个虚拟的 Tensor，期望 was_called 被设置为 true
  auto result = callOp(*op, dummyTensor(DispatchKey::CPU));
  EXPECT_TRUE(was_called);  // 验证 was_called 被正确设置为 true
  EXPECT_EQ(0, result.size());  // 验证结果集的大小为 0
}

// 定义结构体 KernelWithIntOutput，实现 OperatorKernel 接口，用于返回整数输出的内核操作
struct KernelWithIntOutput final : OperatorKernel {
  // 实现操作符重载，接受一个 Tensor 和两个整数参数，返回这两个整数的和
  int64_t operator()(Tensor, int64_t a, int64_t b) {
    return a + b;
  }
};

// 测试 KernelWithIntOutput 内核的注册和调用
TEST(OperatorRegistrationTestFunctorBasedKernel, givenKernelWithIntOutput_whenRegistered_thenCanBeCalled) {
  // 注册一个名为 "_test::int_output" 的操作符，接受一个 Tensor 和两个整数参数，并返回一个整数
  auto registrar = RegisterOperators()
      .op("_test::int_output(Tensor dummy, int a, int b) -> int",
          RegisterOperators::options()
            .kernel<KernelWithIntOutput>(DispatchKey::CPU));  // 使用 KernelWithIntOutput 作为 CPU 分发键的内核
  
  // 查找名为 "_test::int_output" 的操作符模式
  auto op = c10::Dispatcher::singleton().findSchema({"_test::int_output", ""});
  ASSERT_TRUE(op.has_value());  // 断言确保找到了操作符模式
  
  // 调用名为 "_test::int_output" 的操作符，传入一个虚拟的 Tensor 和参数 3, 6，期望返回值为 9
  auto result = callOp(*op, dummyTensor(DispatchKey::CPU), 3, 6);
  EXPECT_EQ(1, result.size());  // 验证结果集的大小为 1
  EXPECT_EQ(9, result[0].toInt());  // 验证返回的整数结果为 9
}

// 定义结构体 KernelWithTensorOutput，实现 OperatorKernel 接口，用于返回与输入相同的 Tensor 输出的内核操作
struct KernelWithTensorOutput final : OperatorKernel {
  // 实现操作符重载，接受一个 Tensor 参数，返回相同的输入 Tensor
  Tensor operator()(const Tensor& input) {
    return input;
  }
};
TEST(OperatorRegistrationTestFunctorBasedKernel, givenKernelWithTensorOutput_whenRegistered_thenCanBeCalled) {
  // 创建 RegisterOperators 对象，用于注册运算符
  auto registrar = RegisterOperators()
      // 注册具有张量输出的操作，指定 CPU 和 CUDA 的内核
      .op("_test::returning_tensor(Tensor input) -> Tensor", RegisterOperators::options().kernel<KernelWithTensorOutput>(DispatchKey::CPU)
                                                                                         .kernel<KernelWithTensorOutput>(DispatchKey::CUDA));

  // 查找注册的操作模式
  auto op = c10::Dispatcher::singleton().findSchema({"_test::returning_tensor", ""});
  // 确保找到对应的操作模式
  ASSERT_TRUE(op.has_value());

  // 调用注册的操作，并传入一个 CPU 的虚拟张量
  auto result = callOp(*op, dummyTensor(DispatchKey::CPU));
  // 确保返回结果的大小为 1
  EXPECT_EQ(1, result.size());
  // 确保返回结果的张量使用了正确的 DispatchKey：CPU
  EXPECT_EQ(DispatchKey::CPU, extractDispatchKey(result[0].toTensor()));

  // 再次调用注册的操作，传入一个 CUDA 的虚拟张量
  result = callOp(*op, dummyTensor(DispatchKey::CUDA));
  // 确保返回结果的大小为 1
  EXPECT_EQ(1, result.size());
  // 确保返回结果的张量使用了正确的 DispatchKey：CUDA
  EXPECT_EQ(DispatchKey::CUDA, extractDispatchKey(result[0].toTensor()));
}

struct KernelWithTensorListOutput final : OperatorKernel {
  // 实现操作核心，返回一个张量列表
  c10::List<Tensor> operator()(const Tensor& input1, const Tensor& input2, const Tensor& input3) {
    return c10::List<Tensor>({input1, input2, input3});
  }
};

TEST(OperatorRegistrationTestFunctorBasedKernel, givenKernelWithTensorListOutput_whenRegistered_thenCanBeCalled) {
  // 创建 RegisterOperators 对象，用于注册运算符
  auto registrar = RegisterOperators()
      // 注册具有张量列表输出的操作，指定 CUDA 的内核
      .op("_test::list_output(Tensor input1, Tensor input2, Tensor input3) -> Tensor[]", RegisterOperators::options().kernel<KernelWithTensorListOutput>(DispatchKey::CUDA));

  // 查找注册的操作模式
  auto op = c10::Dispatcher::singleton().findSchema({"_test::list_output", ""});
  // 确保找到对应的操作模式
  ASSERT_TRUE(op.has_value());

  // 调用注册的操作，并传入三个虚拟张量（CPU, CUDA, CPU）
  auto result = callOp(*op, dummyTensor(DispatchKey::CPU), dummyTensor(DispatchKey::CUDA), dummyTensor(DispatchKey::CPU));
  // 确保返回结果的大小为 1
  EXPECT_EQ(1, result.size());
  // 确保返回结果的张量列表大小为 3
  EXPECT_EQ(3, result[0].toTensorVector().size());
  // 确保返回结果的张量列表中第一个张量使用了正确的 DispatchKey：CPU
  EXPECT_EQ(DispatchKey::CPU, extractDispatchKey(result[0].toTensorVector()[0]));
  // 确保返回结果的张量列表中第二个张量使用了正确的 DispatchKey：CUDA
  EXPECT_EQ(DispatchKey::CUDA, extractDispatchKey(result[0].toTensorVector()[1]));
  // 确保返回结果的张量列表中第三个张量使用了正确的 DispatchKey：CPU
  EXPECT_EQ(DispatchKey::CPU, extractDispatchKey(result[0].toTensorVector()[2]));
}

struct KernelWithIntListOutput final : OperatorKernel {
  // 实现操作核心，返回一个整数列表
  c10::List<int64_t> operator()(const Tensor&, int64_t input1, int64_t input2, int64_t input3) {
    return c10::List<int64_t>({input1, input2, input3});
  }
};

TEST(OperatorRegistrationTestFunctorBasedKernel, givenKernelWithIntListOutput_whenRegistered_thenCanBeCalled) {
  // 创建 RegisterOperators 对象，用于注册运算符
  auto registrar = RegisterOperators()
      // 注册具有整数列表输出的操作，指定 CPU 的内核
      .op("_test::list_output(Tensor dummy, int input1, int input2, int input3) -> int[]", RegisterOperators::options().kernel<KernelWithIntListOutput>(DispatchKey::CPU));

  // 查找注册的操作模式
  auto op = c10::Dispatcher::singleton().findSchema({"_test::list_output", ""});
  // 确保找到对应的操作模式
  ASSERT_TRUE(op.has_value());

  // 调用注册的操作，并传入四个整数参数
  auto result = callOp(*op, dummyTensor(DispatchKey::CPU), 2, 4, 6);
  // 确保返回结果的大小为 1
  EXPECT_EQ(1, result.size());
  // 确保返回结果的整数列表大小为 3
  EXPECT_EQ(3, result[0].toIntVector().size());
  // 确保返回结果的整数列表中第一个整数为 2
  EXPECT_EQ(2, result[0].toIntVector()[0]);
  // 确保返回结果的整数列表中第二个整数为 4
  EXPECT_EQ(4, result[0].toIntVector()[1]);
  // 确保返回结果的整数列表中第三个整数为 6
  EXPECT_EQ(6, result[0].toIntVector()[2]);
}
// 定义一个继承自 OperatorKernel 的结构体 KernelWithMultipleOutputs
struct KernelWithMultipleOutputs final : OperatorKernel {
  // 重载操作符()，接受一个 Tensor 输入并返回一个包含多个值的元组
  std::tuple<Tensor, int64_t, c10::List<Tensor>, std::optional<int64_t>, Dict<string, Tensor>> operator()(Tensor) {
    // 创建一个空的字典 dict
    Dict<string, Tensor> dict;
    // 向字典中插入名为 "first" 的键值对，值是通过 dummyTensor 创建的 CPU 类型的 Tensor
    dict.insert("first", dummyTensor(DispatchKey::CPU));
    // 向字典中插入名为 "second" 的键值对，值是通过 dummyTensor 创建的 CUDA 类型的 Tensor
    dict.insert("second", dummyTensor(DispatchKey::CUDA));
    // 返回一个包含多个值的元组，分别是一个 CUDA 类型的 Tensor、一个 int64_t 类型的整数 5、一个 Tensor 列表、一个可选的 int64_t 类型整数、以及前面创建的字典 dict
    return std::tuple<Tensor, int64_t, c10::List<Tensor>, std::optional<int64_t>, Dict<string, Tensor>>(
      dummyTensor(DispatchKey::CUDA),  // CUDA 类型的 Tensor
      5,  // 整数 5
      c10::List<Tensor>({dummyTensor(DispatchKey::CPU), dummyTensor(DispatchKey::CUDA)}),  // 包含两个 Tensor 的列表，分别是 CPU 和 CUDA 类型的 Tensor
      std::optional<int64_t>(std::in_place, 0),  // 可选的 int64_t 类型整数，值为 0
      dict  // 包含两个键值对的字典，每个值是一个 Tensor
    );
  }
};

// 测试 OperatorRegistrationTestFunctorBasedKernel 中的 KernelWithMultipleOutputs 是否可以成功注册并调用
TEST(OperatorRegistrationTestFunctorBasedKernel, givenKernelWithMultipleOutputs_whenRegistered_thenCanBeCalled) {
  // 注册一个操作符 "_test::multiple_outputs(Tensor dummy) -> (Tensor, int, Tensor[], int?, Dict(str, Tensor))"，并使用 KernelWithMultipleOutputs 处理器在 CPU 上进行注册
  auto registrar = RegisterOperators()
     .op("_test::multiple_outputs(Tensor dummy) -> (Tensor, int, Tensor[], int?, Dict(str, Tensor))", RegisterOperators::options().kernel<KernelWithMultipleOutputs>(DispatchKey::CPU));

  // 查找已注册的操作符 "_test::multiple_outputs"
  auto op = c10::Dispatcher::singleton().findSchema({"_test::multiple_outputs", ""});
  ASSERT_TRUE(op.has_value());

  // 调用已注册的操作符，并传入一个通过 dummyTensor 创建的 CPU 类型的 Tensor
  auto result = callOp(*op, dummyTensor(DispatchKey::CPU));

  // 检查结果的各个部分是否符合预期
  EXPECT_EQ(5, result.size());  // 检查结果的元素个数是否为 5
  EXPECT_EQ(DispatchKey::CUDA, extractDispatchKey(result[0].toTensor()));  // 检查第一个返回的 Tensor 是否为 CUDA 类型
  EXPECT_EQ(5, result[1].toInt());  // 检查第二个返回的整数是否为 5
  EXPECT_EQ(2, result[2].toTensorVector().size());  // 检查第三个返回的 Tensor 列表是否包含 2 个元素
  EXPECT_EQ(DispatchKey::CPU, extractDispatchKey(result[2].toTensorVector()[0]));  // 检查列表中第一个 Tensor 是否为 CPU 类型
  EXPECT_EQ(DispatchKey::CUDA, extractDispatchKey(result[2].toTensorVector()[1]));  // 检查列表中第二个 Tensor 是否为 CUDA 类型
  EXPECT_EQ(0, result[3].toInt());  // 检查第四个返回的整数是否为 0
  // 将第五个返回值转换为字典，并检查其是否包含 2 个键值对
  auto result_dict = c10::impl::toTypedDict<string, Tensor>(result[4].toGenericDict());
  EXPECT_EQ(2, result_dict.size());
  EXPECT_EQ(DispatchKey::CPU, extractDispatchKey(result_dict.at("first")));  // 检查字典中名为 "first" 的值是否为 CPU 类型
  EXPECT_EQ(DispatchKey::CUDA, extractDispatchKey(result_dict.at("second")));  // 检查字典中名为 "second" 的值是否为 CUDA 类型
}

// 定义一个继承自 OperatorKernel 的结构体 KernelWithTensorInputByReferenceWithOutput
struct KernelWithTensorInputByReferenceWithOutput final : OperatorKernel {
  // 重载操作符()，接受一个 Tensor 引用作为输入，并返回相同的 Tensor
  Tensor operator()(const Tensor& input1) {
    return input1;  // 直接返回输入的 Tensor
  }
};

// 定义一个继承自 OperatorKernel 的结构体 KernelWithTensorInputByValueWithOutput
struct KernelWithTensorInputByValueWithOutput final : OperatorKernel {
  // 重载操作符()，接受一个 Tensor 值作为输入，并返回相同的 Tensor
  Tensor operator()(Tensor input1) {
    return input1;  // 直接返回输入的 Tensor
  }
};
// 定义一个名为 captured_input 的全局 Tensor 变量，用于捕获输入的值
Tensor captured_input;

// 定义一个继承自 OperatorKernel 的结构体 KernelWithTensorInputByReferenceWithoutOutput
struct KernelWithTensorInputByReferenceWithoutOutput final : OperatorKernel {
  // 重载括号运算符，接收一个 Tensor 的常引用作为参数，并将其赋值给 captured_input
  void operator()(const Tensor& input1) {
    captured_input = input1;
  }
};

// 定义一个继承自 OperatorKernel 的结构体 KernelWithTensorInputByValueWithoutOutput
struct KernelWithTensorInputByValueWithoutOutput final : OperatorKernel {
  // 重载括号运算符，接收一个 Tensor 值作为参数，并将其赋值给 captured_input
  void operator()(Tensor input1) {
    captured_input = input1;
  }
}
TEST(OperatorRegistrationTestFunctorBasedKernel, givenKernelWithTensorInputByReference_withoutOutput_whenRegistered_thenCanBeCalled) {
  // 注册操作符，接受一个输入张量（按引用传递），没有输出
  auto registrar = RegisterOperators()
      .op("_test::tensor_input(Tensor input) -> ()", RegisterOperators::options().kernel<KernelWithTensorInputByReferenceWithoutOutput>(DispatchKey::CPU)
                                                                                 .kernel<KernelWithTensorInputByReferenceWithoutOutput>(DispatchKey::CUDA));

  // 查找操作符的模式
  auto op = c10::Dispatcher::singleton().findSchema({"_test::tensor_input", ""});
  // 确保操作符模式存在
  ASSERT_TRUE(op.has_value());

  // 调用操作符，传入一个 CPU 上的虚拟张量
  auto outputs = callOp(*op, dummyTensor(DispatchKey::CPU));
  // 验证输出的大小为0
  EXPECT_EQ(0, outputs.size());
  // 验证捕获的输入分发键为 CPU
  EXPECT_EQ(DispatchKey::CPU, extractDispatchKey(captured_input));

  // 再次调用操作符，传入一个 CUDA 上的虚拟张量
  outputs = callOp(*op, dummyTensor(DispatchKey::CUDA));
  // 验证输出的大小为0
  EXPECT_EQ(0, outputs.size());
  // 验证捕获的输入分发键为 CUDA
  EXPECT_EQ(DispatchKey::CUDA, extractDispatchKey(captured_input));
}

TEST(OperatorRegistrationTestFunctorBasedKernel, givenKernelWithTensorInputByValue_withoutOutput_whenRegistered_thenCanBeCalled) {
  // 注册操作符，接受一个输入张量（按值传递），没有输出
  auto registrar = RegisterOperators()
      .op("_test::tensor_input(Tensor input) -> ()", RegisterOperators::options().kernel<KernelWithTensorInputByValueWithoutOutput>(DispatchKey::CPU)
                                                                                 .kernel<KernelWithTensorInputByValueWithoutOutput>(DispatchKey::CUDA));

  // 查找操作符的模式
  auto op = c10::Dispatcher::singleton().findSchema({"_test::tensor_input", ""});
  // 确保操作符模式存在
  ASSERT_TRUE(op.has_value());

  // 调用操作符，传入一个 CPU 上的虚拟张量
  auto outputs = callOp(*op, dummyTensor(DispatchKey::CPU));
  // 验证输出的大小为0
  EXPECT_EQ(0, outputs.size());
  // 验证捕获的输入分发键为 CPU
  EXPECT_EQ(DispatchKey::CPU, extractDispatchKey(captured_input));

  // 再次调用操作符，传入一个 CUDA 上的虚拟张量
  outputs = callOp(*op, dummyTensor(DispatchKey::CUDA));
  // 验证输出的大小为0
  EXPECT_EQ(0, outputs.size());
  // 验证捕获的输入分发键为 CUDA
  EXPECT_EQ(DispatchKey::CUDA, extractDispatchKey(captured_input));
}

int64_t captured_int_input = 0;

struct KernelWithIntInputWithoutOutput final : OperatorKernel {
  // 操作符核心逻辑，接受一个张量和一个整数作为输入，无输出
  void operator()(Tensor, int64_t input1) {
    captured_int_input = input1;
  }
};

TEST(OperatorRegistrationTestFunctorBasedKernel, givenKernelWithIntInput_withoutOutput_whenRegistered_thenCanBeCalled) {
  // 注册操作符，接受一个虚拟张量和一个整数输入，没有输出
  auto registrar = RegisterOperators()
      .op("_test::int_input(Tensor dummy, int input) -> ()", RegisterOperators::options().kernel<KernelWithIntInputWithoutOutput>(DispatchKey::CPU));

  // 查找操作符的模式
  auto op = c10::Dispatcher::singleton().findSchema({"_test::int_input", ""});
  // 确保操作符模式存在
  ASSERT_TRUE(op.has_value());

  // 初始化捕获的整数输入为0
  captured_int_input = 0;
  // 调用操作符，传入一个 CPU 上的虚拟张量和整数输入 3
  auto outputs = callOp(*op, dummyTensor(DispatchKey::CPU), 3);
  // 验证输出的大小为0
  EXPECT_EQ(0, outputs.size());
  // 验证捕获的整数输入为 3
  EXPECT_EQ(3, captured_int_input);
}

struct KernelWithIntInputWithOutput final : OperatorKernel {
  // 操作符核心逻辑，接受一个张量和一个整数作为输入，返回整数加1
  int64_t operator()(Tensor, int64_t input1) {
    return input1 + 1;
  }
};
TEST(OperatorRegistrationTestFunctorBasedKernel, givenKernelWithIntInput_withOutput_whenRegistered_thenCanBeCalled) {
  // 创建操作注册器，并注册具有整数输入和输出的自定义内核
  auto registrar = RegisterOperators()
      .op("_test::int_input(Tensor dummy, int input) -> int", RegisterOperators::options().kernel<KernelWithIntInputWithOutput>(DispatchKey::CPU));

  // 查找名为 "_test::int_input" 的操作模式
  auto op = c10::Dispatcher::singleton().findSchema({"_test::int_input", ""});
  // 断言是否成功找到操作模式
  ASSERT_TRUE(op.has_value());

  // 调用操作，并获取输出结果
  auto outputs = callOp(*op, dummyTensor(DispatchKey::CPU), 3);
  // 断言输出的大小为1
  EXPECT_EQ(1, outputs.size());
  // 断言输出的值为4
  EXPECT_EQ(4, outputs[0].toInt());
}

int64_t captured_input_list_size = 0;

struct KernelWithIntListInputWithoutOutput final : OperatorKernel {
  // 自定义操作内核，处理张量和整数列表输入，并捕获列表的大小
  void operator()(Tensor, const c10::List<int64_t>& input1) {
    captured_input_list_size = input1.size();
  }
};

TEST(OperatorRegistrationTestFunctorBasedKernel, givenKernelWithIntListInput_withoutOutput_whenRegistered_thenCanBeCalled) {
  // 创建操作注册器，并注册具有整数列表输入和无输出的自定义内核
  auto registrar = RegisterOperators()
      .op("_test::int_list_input(Tensor dummy, int[] input) -> ()", RegisterOperators::options().kernel<KernelWithIntListInputWithoutOutput>(DispatchKey::CPU));

  // 查找名为 "_test::int_list_input" 的操作模式
  auto op = c10::Dispatcher::singleton().findSchema({"_test::int_list_input", ""});
  // 断言是否成功找到操作模式
  ASSERT_TRUE(op.has_value());

  // 重置捕获的输入列表大小
  captured_input_list_size = 0;
  // 调用操作，并传入整数列表作为输入
  auto outputs = callOp(*op, dummyTensor(DispatchKey::CPU), c10::List<int64_t>({2, 4, 6}));
  // 断言输出的大小为0
  EXPECT_EQ(0, outputs.size());
  // 断言捕获的输入列表大小为3
  EXPECT_EQ(3, captured_input_list_size);
}

struct KernelWithIntListInputWithOutput final : OperatorKernel {
  // 自定义操作内核，处理张量和整数列表输入，并返回列表的大小
  int64_t operator()(Tensor, const c10::List<int64_t>& input1) {
    return input1.size();
  }
};

TEST(OperatorRegistrationTestFunctorBasedKernel, givenKernelWithIntListInput_withOutput_whenRegistered_thenCanBeCalled) {
  // 创建操作注册器，并注册具有整数列表输入和整数输出的自定义内核
  auto registrar = RegisterOperators()
      .op("_test::int_list_input(Tensor dummy, int[] input) -> int", RegisterOperators::options().kernel<KernelWithIntListInputWithOutput>(DispatchKey::CPU));

  // 查找名为 "_test::int_list_input" 的操作模式
  auto op = c10::Dispatcher::singleton().findSchema({"_test::int_list_input", ""});
  // 断言是否成功找到操作模式
  ASSERT_TRUE(op.has_value());

  // 调用操作，并传入整数列表作为输入
  auto outputs = callOp(*op, dummyTensor(DispatchKey::CPU), c10::List<int64_t>({2, 4, 6}));
  // 断言输出的大小为1
  EXPECT_EQ(1, outputs.size());
  // 断言输出的值为3，即整数列表的大小
  EXPECT_EQ(3, outputs[0].toInt());
}

struct KernelWithTensorListInputWithoutOutput final : OperatorKernel {
  // 自定义操作内核，处理张量列表输入，并捕获列表的大小
  void operator()(const c10::List<Tensor>& input1) {
    captured_input_list_size = input1.size();
  }
};
TEST(OperatorRegistrationTestFunctorBasedKernel, givenKernelWithTensorListInput_withoutOutput_whenRegistered_thenCanBeCalled) {
  // 注册运算符，指定输入为张量列表，无输出，使用特定的内核处理器类型（CPU）
  auto registrar = RegisterOperators()
      .op("_test::tensor_list_input(Tensor[] input) -> ()", RegisterOperators::options().kernel<KernelWithTensorListInputWithoutOutput>(DispatchKey::CPU));

  // 查找指定名称和空字符串的运算符模式
  auto op = c10::Dispatcher::singleton().findSchema({"_test::tensor_list_input", ""});
  // 断言确保找到对应的运算符模式
  ASSERT_TRUE(op.has_value());

  // 设置捕获的输入列表大小为0
  captured_input_list_size = 0;
  // 调用运算符，并传入两个虚拟张量作为参数，期望返回值为空
  auto outputs = callOp(*op, c10::List<Tensor>({dummyTensor(DispatchKey::CPU), dummyTensor(DispatchKey::CPU)}));
  // 预期输出的大小为0
  EXPECT_EQ(0, outputs.size());
  // 预期捕获的输入列表大小为2
  EXPECT_EQ(2, captured_input_list_size);
}

struct KernelWithTensorListInputWithOutput final : OperatorKernel {
  // 操作符内核实现，接受张量列表输入，返回列表的大小作为整数
  int64_t operator()(const c10::List<Tensor>& input1) {
    return input1.size();
  }
};

TEST(OperatorRegistrationTestFunctorBasedKernel, givenKernelWithTensorListInput_withOutput_whenRegistered_thenCanBeCalled) {
  // 注册运算符，指定输入为张量列表，输出为整数，使用特定的内核处理器类型（CPU）
  auto registrar = RegisterOperators()
      .op("_test::tensor_list_input(Tensor[] input) -> int", RegisterOperators::options().kernel<KernelWithTensorListInputWithOutput>(DispatchKey::CPU));

  // 查找指定名称和空字符串的运算符模式
  auto op = c10::Dispatcher::singleton().findSchema({"_test::tensor_list_input", ""});
  // 断言确保找到对应的运算符模式
  ASSERT_TRUE(op.has_value());

  // 调用运算符，并传入两个虚拟张量作为参数，期望返回一个包含列表大小的输出
  auto outputs = callOp(*op, c10::List<Tensor>({dummyTensor(DispatchKey::CPU), dummyTensor(DispatchKey::CPU)}));
  // 预期输出的大小为1
  EXPECT_EQ(1, outputs.size());
  // 预期输出的第一个元素为2
  EXPECT_EQ(2, outputs[0].toInt());
}

int captured_dict_size = 0;

struct KernelWithDictInputWithoutOutput final : OperatorKernel {
  // 操作符内核实现，接受字典输入，捕获字典的大小作为成员变量
  void operator()(Dict<string, Tensor> input1) {
    captured_dict_size = input1.size();
  }
};

TEST(OperatorRegistrationTestFunctorBasedKernel, givenKernelWithDictInput_withoutOutput_whenRegistered_thenCanBeCalled) {
  // 注册运算符，指定输入为字符串键值对的字典，无输出，使用通用内核处理器类型
  auto registrar = RegisterOperators()
      .op("_test::dict_input(Dict(str, Tensor) input) -> ()", RegisterOperators::options().catchAllKernel<KernelWithDictInputWithoutOutput>());

  // 查找指定名称和空字符串的运算符模式
  auto op = c10::Dispatcher::singleton().findSchema({"_test::dict_input", ""});
  // 断言确保找到对应的运算符模式
  ASSERT_TRUE(op.has_value());

  // 设置捕获的字典大小为0
  captured_dict_size = 0;
  // 创建一个包含两个虚拟张量的字典，作为运算符的参数，期望返回值为空
  Dict<string, Tensor> dict;
  dict.insert("key1", dummyTensor(DispatchKey::CPU));
  dict.insert("key2", dummyTensor(DispatchKey::CUDA));
  auto outputs = callOp(*op, dict);
  // 预期输出的大小为0
  EXPECT_EQ(0, outputs.size());
  // 预期捕获的字典大小为2
  EXPECT_EQ(2, captured_dict_size);
}

struct KernelWithDictInputWithOutput final : OperatorKernel {
  // 操作符内核实现，接受字符串键值对的字典输入，返回字典中指定键的字符串值
  string operator()(Dict<string, string> input1) {
    return input1.at("key2");
  }
};
TEST(OperatorRegistrationTestFunctorBasedKernel, givenKernelWithDictInput_withOutput_whenRegistered_thenCanBeCalled) {
  // 创建一个操作符注册对象，注册一个接受字典输入和返回字符串的操作符
  auto registrar = RegisterOperators()
      .op("_test::dict_input(Dict(str, str) input) -> str", RegisterOperators::options().catchAllKernel<KernelWithDictInputWithOutput>());

  // 查询已注册的操作符，通过操作符名称和空字符串作为命名空间
  auto op = c10::Dispatcher::singleton().findSchema({"_test::dict_input", ""});
  // 确保操作符成功找到
  ASSERT_TRUE(op.has_value());

  // 创建一个字典对象，插入两个键值对
  Dict<string, string> dict;
  dict.insert("key1", "value1");
  dict.insert("key2", "value2");
  // 调用操作符，并传入字典作为参数，获取输出
  auto outputs = callOp(*op, dict);
  // 验证输出的大小为1
  EXPECT_EQ(1, outputs.size());
  // 验证输出的第一个元素为"value2"
  EXPECT_EQ("value2", outputs[0].toStringRef());
}

struct KernelWithDictOutput final : OperatorKernel {
  // 定义一个操作符核函数，接受字典输入并返回相同的字典
  Dict<string, string> operator()(Dict<string, string> input) {
    return input;
  }
};

TEST(OperatorRegistrationTestFunctorBasedKernel, givenKernelWithDictOutput_whenRegistered_thenCanBeCalled) {
  // 创建一个操作符注册对象，注册一个接受字典输入并返回字典的操作符
  auto registrar = RegisterOperators()
      .op("_test::dict_output(Dict(str, str) input) -> Dict(str, str)", RegisterOperators::options().catchAllKernel<KernelWithDictOutput>());

  // 查询已注册的操作符，通过操作符名称和空字符串作为命名空间
  auto op = c10::Dispatcher::singleton().findSchema({"_test::dict_output", ""});
  // 确保操作符成功找到
  ASSERT_TRUE(op.has_value());

  // 创建一个字典对象，插入两个键值对
  Dict<string, string> dict;
  dict.insert("key1", "value1");
  dict.insert("key2", "value2");
  // 调用操作符，并传入字典作为参数，获取输出
  auto outputs = callOp(*op, dict);
  // 验证输出的大小为1
  EXPECT_EQ(1, outputs.size());
  // 将输出转换为类型化的字典对象，以便进行进一步验证
  auto output = c10::impl::toTypedDict<string, string>(outputs[0].toGenericDict());

  // 验证输出字典的大小为2
  EXPECT_EQ(2, output.size());
  // 验证输出字典中"key1"对应的值为"value1"
  EXPECT_EQ("value1", output.at("key1"));
  // 验证输出字典中"key2"对应的值为"value2"
  EXPECT_EQ("value2", output.at("key2"));
}

class KernelWithCache final : public OperatorKernel {
public:
  // 定义一个操作符核函数，每次调用返回一个递增的计数器值
  KernelWithCache(): counter(3) {}

  int64_t operator()(Tensor) {
    return ++counter;
  }
private:
  int64_t counter;
};

struct KernelWithTupleInput final : OperatorKernel {
  // 定义一个操作符核函数，接受元组输入并返回其中第一个字符串元素
  string operator()(std::tuple<string, int64_t, double> input1) {
    return std::get<0>(input1);
  }
};

TEST(OperatorRegistrationTestFunctorBasedKernel, givenKernelWithTupleInput_withOutput_whenRegistered_thenCanBeCalled) {
  // 创建一个操作符注册对象，注册一个接受元组输入和返回字符串的操作符
  auto registrar = RegisterOperators()
      .op("_test::tuple_input((str, int, float) input) -> str", RegisterOperators::options().catchAllKernel<KernelWithTupleInput>());

  // 查询已注册的操作符，通过操作符名称和空字符串作为命名空间
  auto op = c10::Dispatcher::singleton().findSchema({"_test::tuple_input", ""});
  // 确保操作符成功找到
  ASSERT_TRUE(op.has_value());

  // 创建一个包含字符串、整数和浮点数的元组
  std::tuple<string, int64_t, float> tup{"foobar", 123, 420.1337};
  // 调用操作符，并传入元组作为参数，获取输出
  auto outputs = callOp(*op, tup);
  // 验证输出的大小为1
  EXPECT_EQ(1, outputs.size());
  // 验证输出的第一个元素为"foobar"
  EXPECT_EQ("foobar", outputs[0].toStringRef());
}
TEST(OperatorRegistrationTestFunctorBasedKernel, givenKernelWithCache_thenCacheIsKeptCorrectly) {
  // 创建操作注册器，注册一个名为 "_test::cache_op" 的操作，使用 KernelWithCache 类处理 CPU 分发
  auto registrar = RegisterOperators()
      .op("_test::cache_op(Tensor input) -> int", RegisterOperators::options().kernel<KernelWithCache>(DispatchKey::CPU));

  // 查找名为 "_test::cache_op" 的操作的模式
  auto op = c10::Dispatcher::singleton().findSchema({"_test::cache_op", ""});
  ASSERT_TRUE(op.has_value());

  // 预期第一次调用返回 4（缓存中的初始值为4）
  auto stack = makeStack(dummyTensor(DispatchKey::CPU));
  op->callBoxed(&stack);
  EXPECT_EQ(1, stack.size());
  EXPECT_EQ(4, stack[0].toInt());

  // 预期第二次调用返回 5
  stack = makeStack(dummyTensor(DispatchKey::CPU));
  op->callBoxed(&stack);
  EXPECT_EQ(1, stack.size());
  EXPECT_EQ(5, stack[0].toInt());

  // 预期第三次调用返回 6
  stack = makeStack(dummyTensor(DispatchKey::CPU));
  op->callBoxed(&stack);
  EXPECT_EQ(1, stack.size());
  EXPECT_EQ(6, stack[0].toInt());
}

class KernelWithConstructorArg final : public OperatorKernel {
public:
  // 显式构造函数，接受一个 int64_t 类型参数作为偏移量
  explicit KernelWithConstructorArg(int64_t offset)
  : offset_(offset) {}

  // 操作符重载，对输入进行偏移并返回结果
  int64_t operator()(const Tensor&, int64_t input) {
    return input + offset_;
  }

private:
  int64_t offset_;  // 存储偏移量的私有成员变量
};

TEST(OperatorRegistrationTestFunctorBasedKernel, givenKernelWithConstructorArg_whenRegistered_thenCanBeCalled) {
  // 创建操作注册器，注册名为 "_test::offset_op" 的操作，使用 KernelWithConstructorArg 类处理 CPU 和 CUDA 分发
  auto registrar = RegisterOperators()
      .op("_test::offset_op(Tensor tensor, int input) -> int", RegisterOperators::options().kernel<KernelWithConstructorArg>(DispatchKey::CPU, 2)
                                                                                           .kernel<KernelWithConstructorArg>(DispatchKey::CUDA, 4));

  // 查找名为 "_test::offset_op" 的操作的模式
  auto op = c10::Dispatcher::singleton().findSchema({"_test::offset_op", ""});
  ASSERT_TRUE(op.has_value());

  // 在 CPU 上调用操作，预期输出为 6 (4 + 2)
  auto outputs = callOp(*op, dummyTensor(DispatchKey::CPU), 4);
  EXPECT_EQ(1, outputs.size());
  EXPECT_EQ(6, outputs[0].toInt());

  // 在 CUDA 上调用操作，预期输出为 8 (4 + 4)
  outputs = callOp(*op, dummyTensor(DispatchKey::CUDA), 4);
  EXPECT_EQ(1, outputs.size());
  EXPECT_EQ(8, outputs[0].toInt());
}

class KernelWithMultipleConstructorArgs final : public OperatorKernel {
public:
  // 显式构造函数，接受两个 int64_t 类型参数作为偏移量，将它们相加得到 offset_
  explicit KernelWithMultipleConstructorArgs(int64_t offset1, int64_t offset2)
  : offset_(offset1 + offset2) {}

  // 操作符重载，对输入进行偏移并返回结果
  int64_t operator()(const Tensor&, int64_t input) {
    return input + offset_;
  }

private:
  int64_t offset_;  // 存储偏移量的私有成员变量
};
TEST(OperatorRegistrationTestFunctorBasedKernel, givenKernelWithMultipleConstructorArgs_whenRegistered_thenCanBeCalled) {
  // 创建注册器对象，注册带有多个构造参数的操作符，并指定CPU和CUDA的内核函数
  auto registrar = RegisterOperators()
      .op("_test::offset_op(Tensor tensor, int input) -> int", RegisterOperators::options().kernel<KernelWithMultipleConstructorArgs>(DispatchKey::CPU, 2, 3)
                                                                                           .kernel<KernelWithMultipleConstructorArgs>(DispatchKey::CUDA, 4, 5));

  // 查找名称为 "_test::offset_op" 的操作符模式
  auto op = c10::Dispatcher::singleton().findSchema({"_test::offset_op", ""});
  // 断言找到操作符模式
  ASSERT_TRUE(op.has_value());

  // 调用操作符，并验证输出结果
  auto outputs = callOp(*op, dummyTensor(DispatchKey::CPU), 4);
  EXPECT_EQ(1, outputs.size());
  EXPECT_EQ(9, outputs[0].toInt());

  // 再次调用操作符，使用不同的调度键（CUDA），并验证输出结果
  outputs = callOp(*op, dummyTensor(DispatchKey::CUDA), 4);
  EXPECT_EQ(1, outputs.size());
  EXPECT_EQ(13, outputs[0].toInt());
}

bool called = false;

struct KernelWithoutInputs final : OperatorKernel {
  // 定义操作符内核函数，设置标志为已调用
  void operator()() {
    called = true;
  }
};

TEST(OperatorRegistrationTestFunctorBasedKernel, givenFallbackKernelWithoutAnyArguments_whenRegistered_thenCanBeCalled) {
  // 注意：没有张量参数的非回退内核无法工作，因为无法获取调度键。
  // 对于仅有回退内核的操作符，这对向后兼容性至关重要。
  auto registrar = RegisterOperators()
      .op("_test::no_tensor_args() -> ()", RegisterOperators::options().catchAllKernel<KernelWithoutInputs>());

  // 查找名称为 "_test::no_tensor_args" 的操作符模式
  auto op = c10::Dispatcher::singleton().findSchema({"_test::no_tensor_args", ""});
  // 断言找到操作符模式
  ASSERT_TRUE(op.has_value());

  // 重置调用标志，调用操作符，并验证已调用标志为真
  called = false;
  auto outputs = callOp(*op);
  EXPECT_TRUE(called);
}

struct KernelWithoutTensorInputs final : OperatorKernel {
  // 定义操作符内核函数，接受一个整数参数，并返回增加1后的结果
  int64_t operator()(int64_t arg) {
    return arg + 1;
  }
};

TEST(OperatorRegistrationTestFunctorBasedKernel, givenFallbackKernelWithoutTensorArguments_whenRegistered_thenCanBeCalled) {
  // 注意：没有张量参数的非回退内核无法工作，因为无法获取调度键。
  // 对于仅有回退内核的操作符，这对向后兼容性至关重要。
  auto registrar = RegisterOperators()
      .op("_test::no_tensor_args(int arg) -> int", RegisterOperators::options().catchAllKernel<KernelWithoutTensorInputs>());

  // 查找名称为 "_test::no_tensor_args" 的操作符模式
  auto op = c10::Dispatcher::singleton().findSchema({"_test::no_tensor_args", ""});
  // 断言找到操作符模式
  ASSERT_TRUE(op.has_value());

  // 调用操作符，并验证输出结果
  auto outputs = callOp(*op, 3);
  EXPECT_EQ(1, outputs.size());
  EXPECT_EQ(4, outputs[0].toInt());
}

std::optional<Tensor> called_arg2 = c10::nullopt;
std::optional<int64_t> called_arg3 = c10::nullopt;
std::optional<std::string> called_arg4 = c10::nullopt;

struct KernelWithOptInputWithoutOutput final : OperatorKernel {
  // 定义操作符内核函数，接受张量参数和可选的其他参数，并设置全局变量以便后续验证
  void operator()(Tensor arg1, const std::optional<Tensor>& arg2, std::optional<int64_t> arg3, std::optional<std::string> arg4) {
    called = true;
    called_arg2 = arg2;
    called_arg3 = arg3;
    called_arg4 = arg4;
  }
};
TEST(OperatorRegistrationTestFunctorBasedKernel, givenKernelWithOptionalInputs_withoutOutput_whenRegistered_thenCanBeCalled) {
  // 注册一个具有可选输入但无输出的操作符，并指定其调度到 CPU 的内核
  auto registrar = RegisterOperators().op("_test::opt_input(Tensor arg1, Tensor? arg2, int? arg3, str? arg4) -> ()", RegisterOperators::options().kernel<KernelWithOptInputWithoutOutput>(DispatchKey::CPU));
  // 查找已注册的操作符模式
  auto op = c10::Dispatcher::singleton().findSchema({"_test::opt_input", ""});
  // 断言操作符模式已找到
  ASSERT_TRUE(op.has_value());

  // 初始化测试状态
  called = false;
  // 调用操作符，并验证输出结果为空
  auto outputs = callOp(*op, dummyTensor(DispatchKey::CPU), dummyTensor(DispatchKey::CPU), c10::IValue(), std::string("text"));
  EXPECT_EQ(0, outputs.size());

  // 验证测试状态
  EXPECT_TRUE(called);
  // 验证可选输入参数 arg2 已被调用
  EXPECT_TRUE(called_arg2.has_value());
  EXPECT_EQ(extractDispatchKey(*called_arg2), DispatchKey::CPU);
  // 验证可选输入参数 arg3 未被调用
  EXPECT_FALSE(called_arg3.has_value());
  // 验证可选输入参数 arg4 已被调用
  EXPECT_TRUE(called_arg4.has_value());
  EXPECT_EQ(*called_arg4, "text");

  // 重置测试状态
  called = false;
  // 再次调用操作符，并验证输出结果为空
  outputs = callOp(*op, dummyTensor(DispatchKey::CPU), c10::IValue(), 4, c10::IValue());
  EXPECT_EQ(0, outputs.size());

  // 验证测试状态
  EXPECT_TRUE(called);
  // 验证可选输入参数 arg2 未被调用
  EXPECT_FALSE(called_arg2.has_value());
  // 验证可选输入参数 arg3 已被调用
  EXPECT_TRUE(called_arg3.has_value());
  EXPECT_EQ(*called_arg3, 4);
  // 验证可选输入参数 arg4 未被调用
  EXPECT_FALSE(called_arg4.has_value());
}

struct KernelWithOptInputWithOutput final : OperatorKernel {
  std::optional<Tensor> operator()(Tensor arg1, const std::optional<Tensor>& arg2, std::optional<int64_t> arg3, std::optional<std::string> arg4) {
    // 标记操作符已被调用
    called = true;
    // 记录调用时的可选输入参数值
    called_arg2 = arg2;
    called_arg3 = arg3;
    called_arg4 = arg4;
    // 返回可选输入参数 arg2 的值
    return arg2;
  }
};

TEST(OperatorRegistrationTestFunctorBasedKernel, givenKernelWithOptionalInputs_withOutput_whenRegistered_thenCanBeCalled) {
  // 注册一个具有可选输入和输出的操作符，并指定其调度到 CPU 的内核
  auto registrar = RegisterOperators().op("_test::opt_input(Tensor arg1, Tensor? arg2, int? arg3, str? arg4) -> Tensor?", RegisterOperators::options().kernel<KernelWithOptInputWithOutput>(DispatchKey::CPU));
  // 查找已注册的操作符模式
  auto op = c10::Dispatcher::singleton().findSchema({"_test::opt_input", ""});
  // 断言操作符模式已找到
  ASSERT_TRUE(op.has_value());

  // 初始化测试状态
  called = false;
  // 调用操作符，并验证输出结果包含一个张量
  auto outputs = callOp(*op, dummyTensor(DispatchKey::CPU), dummyTensor(DispatchKey::CPU), c10::IValue(), std::string("text"));
  EXPECT_EQ(1, outputs.size());
  EXPECT_EQ(DispatchKey::CPU, extractDispatchKey(outputs[0].toTensor()));

  // 验证测试状态
  EXPECT_TRUE(called);
  // 验证可选输入参数 arg2 已被调用
  EXPECT_TRUE(called_arg2.has_value());
  EXPECT_EQ(extractDispatchKey(*called_arg2), DispatchKey::CPU);
  // 验证可选输入参数 arg3 未被调用
  EXPECT_FALSE(called_arg3.has_value());
  // 验证可选输入参数 arg4 已被调用
  EXPECT_TRUE(called_arg4.has_value());
  EXPECT_EQ(*called_arg4, "text");

  // 重置测试状态
  called = false;
  // 再次调用操作符，并验证输出结果包含一个空值
  outputs = callOp(*op, dummyTensor(DispatchKey::CPU), c10::IValue(), 4, c10::IValue());
  EXPECT_EQ(1, outputs.size());
  EXPECT_TRUE(outputs[0].isNone());

  // 验证测试状态
  EXPECT_TRUE(called);
  // 验证可选输入参数 arg2 未被调用
  EXPECT_FALSE(called_arg2.has_value());
  // 验证可选输入参数 arg3 已被调用
  EXPECT_TRUE(called_arg3.has_value());
  EXPECT_EQ(*called_arg3, 4);
  // 验证可选输入参数 arg4 未被调用
  EXPECT_FALSE(called_arg4.has_value());
}
// 定义了一个名为 KernelWithOptInputWithMultipleOutputs 的结构体，继承自 OperatorKernel
struct KernelWithOptInputWithMultipleOutputs final : OperatorKernel {
  // 重载操作符()，接受多个参数，返回一个元组，其中包含三个可选类型的对象：Tensor、int64_t 和 std::string
  std::tuple<std::optional<Tensor>, std::optional<int64_t>, std::optional<std::string>>
  operator()(Tensor arg1, const std::optional<Tensor>& arg2, std::optional<int64_t> arg3, std::optional<std::string> arg4) {
    return std::make_tuple(arg2, arg3, arg4);
  }
};

// 测试函数 OperatorRegistrationTestFunctorBasedKernel，测试注册带有多个可选输入和多个输出的操作核心的情况
TEST(OperatorRegistrationTestFunctorBasedKernel, givenKernelWithOptionalInputs_withMultipleOutputs_whenRegistered_thenCanBeCalled) {
  // 注册操作符 "_test::opt_input"，指定了输入参数和输出类型，以及使用的 KernelWithOptInputWithMultipleOutputs 作为内核
  auto registrar = RegisterOperators().op("_test::opt_input(Tensor arg1, Tensor? arg2, int? arg3, str? arg4) -> (Tensor?, int?, str?)", RegisterOperators::options().kernel<KernelWithOptInputWithMultipleOutputs>(DispatchKey::CPU));
  // 查找注册的操作符 "_test::opt_input"
  auto op = c10::Dispatcher::singleton().findSchema({"_test::opt_input", ""});
  ASSERT_TRUE(op.has_value());

  // 调用注册的操作符，传入参数调用，期望返回一个包含三个元素的列表
  auto outputs = callOp(*op, dummyTensor(DispatchKey::CPU), dummyTensor(DispatchKey::CPU), c10::IValue(), std::string("text"));
  // 断言输出元素数量为3
  EXPECT_EQ(3, outputs.size());
  // 断言第一个输出元素的调度键为 CPU
  EXPECT_EQ(DispatchKey::CPU, extractDispatchKey(outputs[0].toTensor()));
  // 断言第二个输出元素为 None 类型
  EXPECT_TRUE(outputs[1].isNone());
  // 断言第三个输出元素为字符串 "text"
  EXPECT_EQ("text", outputs[2].toStringRef());

  // 再次调用注册的操作符，使用不同的参数组合
  outputs = callOp(*op, dummyTensor(DispatchKey::CPU), c10::IValue(), 4, c10::IValue());
  // 断言输出元素数量为3
  EXPECT_EQ(3, outputs.size());
  // 断言第一个输出元素为 None 类型
  EXPECT_TRUE(outputs[0].isNone());
  // 断言第二个输出元素为整数 4
  EXPECT_EQ(4, outputs[1].toInt());
  // 断言第三个输出元素为 None 类型
  EXPECT_TRUE(outputs[2].isNone());
}

// 定义了一个名为 ConcatKernel 的结构体，继承自 OperatorKernel
struct ConcatKernel final : OperatorKernel {
  // 显式构造函数，接受一个字符串作为前缀
  explicit ConcatKernel(std::string prefix): prefix_(std::move(prefix)) {}

  // 重载操作符()，接受多个参数，返回一个字符串，将前缀、参数 a、b 和 c 组合成一个字符串
  std::string operator()(const Tensor& tensor1, std::string a, const std::string& b, int64_t c) {
    return prefix_ + a + b + std::to_string(c);
  }

  std::string prefix_;
};

// 辅助函数，测试在给定调度键下调用注册的操作符，预期结果为 "prefix123"
void expectCallsConcatUnboxed(DispatchKey dispatch_key) {
  // 进入自动分发模式
  at::AutoDispatchBelowAutograd mode;

  // 断言可以找到指定的操作符 "_test::my_op"
  auto op = c10::Dispatcher::singleton().findSchema({"_test::my_op", ""});
  ASSERT_TRUE(op.has_value());

  // 调用注册的操作符，传入参数调用，预期返回结果为 "prefix123"
  std::string result = callOpUnboxed<std::string, const Tensor&, std::string, const std::string&, int64_t>(*op, dummyTensor(dispatch_key), "1", "2", 3);
  EXPECT_EQ("prefix123", result);
}

// 测试函数 OperatorRegistrationTestFunctorBasedKernel，测试注册一个带有特定前缀的 ConcatKernel 操作核心
TEST(OperatorRegistrationTestFunctorBasedKernel, givenKernel_whenRegistered_thenCanBeCalledUnboxed) {
  // 注册操作符 "_test::my_op"，指定输入参数和输出类型，并使用指定前缀的 ConcatKernel 作为内核
  auto registrar = RegisterOperators().op("_test::my_op(Tensor dummy, str a, str b, int c) -> str", RegisterOperators::options().kernel<ConcatKernel>(DispatchKey::CPU, "prefix"));
  // 调用辅助函数，使用 CPU 调度键调用操作符
  expectCallsConcatUnboxed(DispatchKey::CPU);
}

// 定义了一个名为 KernelForSchemaInference 的结构体，继承自 OperatorKernel
struct KernelForSchemaInference final : OperatorKernel {
  // 重载操作符()，接受多个参数，返回一个空元组
  std::tuple<int64_t, Tensor> operator()(Tensor arg1, int64_t arg2, const c10::List<Tensor>& arg3) {
    return {};
  }
};
TEST(OperatorRegistrationTestFunctorBasedKernel, givenKernel_whenRegisteredWithoutSpecifyingSchema_thenInfersSchema) {
  // 注册操作符 "_test::no_schema_specified"，使用默认选项注册一个需要推断模式的内核
  auto registrar = RegisterOperators()
      .op("_test::no_schema_specified", RegisterOperators::options().kernel<KernelForSchemaInference>(DispatchKey::CPU));

  // 查找注册的操作符的模式（schema）
  auto op = c10::Dispatcher::singleton().findSchema({"_test::no_schema_specified", ""});
  // 断言操作符模式存在
  ASSERT_TRUE(op.has_value());

  // 检查注册的模式与预期模式之间的差异
  std::optional<std::string> differences = c10::findSchemaDifferences(torch::jit::parseSchema("_test::no_schema_specified(Tensor arg1, int arg2, Tensor[] arg3) -> (int, Tensor)"), op->schema());
  // 预期模式与注册的模式没有差异
  EXPECT_FALSE(differences.has_value());
}

TEST(OperatorRegistrationTestFunctorBasedKernel, givenKernel_whenRegisteredCatchAllWithoutSpecifyingSchema_thenInfersSchema) {
  // 注册操作符 "_test::no_schema_specified"，使用默认选项注册一个需要推断模式的 catch-all 内核
  auto registrar = RegisterOperators()
      .op("_test::no_schema_specified", RegisterOperators::options().catchAllKernel<KernelForSchemaInference>());

  // 查找注册的操作符的模式（schema）
  auto op = c10::Dispatcher::singleton().findSchema({"_test::no_schema_specified", ""});
  // 断言操作符模式存在
  ASSERT_TRUE(op.has_value());

  // 检查注册的模式与预期模式之间的差异
  std::optional<std::string> differences = c10::findSchemaDifferences(torch::jit::parseSchema("_test::no_schema_specified(Tensor arg1, int arg2, Tensor[] arg3) -> (int, Tensor)"), op->schema());
  // 预期模式与注册的模式没有差异
  EXPECT_FALSE(differences.has_value());
}

template<class Return, class... Args> struct KernelFunc final : OperatorKernel{
  Return operator()(Args...) { return {}; }
};
template<class... Args> struct KernelFunc<void, Args...> final : OperatorKernel {
  void operator()(Args...) {}
};

TEST(OperatorRegistrationTestFunctorBasedKernel, givenMismatchedKernel_withDifferentNumArguments_whenRegistering_thenFails) {
  // 注册操作符 "_test::mismatch(Tensor arg) -> int"，使用指定内核
  // 预期成功，因为注册的模式与期望模式匹配
  RegisterOperators()
      .op("_test::mismatch(Tensor arg) -> int", RegisterOperators::options().kernel<KernelFunc<int64_t, Tensor>>(DispatchKey::CPU));

  // 注册操作符 "_test::mismatch(Tensor arg, Tensor arg2) -> int"，使用指定内核
  // 预期抛出异常，因为参数数量不匹配
  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg, Tensor arg2) -> int", RegisterOperators::options().kernel<KernelFunc<int64_t, Tensor>>(DispatchKey::CPU));
    }, "The number of arguments is different. 2 vs 1"
  );

  // 注册操作符 "_test::mismatch(Tensor arg, Tensor arg2) -> ()"，使用指定内核
  // 预期成功，因为注册的模式与期望模式匹配
  RegisterOperators()
      .op("_test::mismatch(Tensor arg, Tensor arg2) -> ()", RegisterOperators::options().kernel<KernelFunc<void, Tensor, Tensor>>(DispatchKey::CPU));

  // 注册操作符 "_test::mismatch() -> ()"，使用指定内核
  // 预期抛出异常，因为参数数量不匹配
  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch() -> ()", RegisterOperators::options().kernel<KernelFunc<void, Tensor, Tensor>>(DispatchKey::CPU));
    }, "The number of arguments is different. 0 vs 2"
  );

  // 注册操作符 "_test::mismatch(Tensor arg) -> ()"，使用指定内核
  // 预期抛出异常，因为参数数量不匹配
  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> ()", RegisterOperators::options().kernel<KernelFunc<void, Tensor, Tensor>>(DispatchKey::CPU));
    }, "The number of arguments is different. 1 vs 2"
  );

  expectThrows<c10::Error>([] {
    RegisterOperators()
        // 注册自定义操作 "_test::mismatch"，接受三个 Tensor 类型的参数，并指定 CPU 上的内核函数
        .op("_test::mismatch(Tensor arg, Tensor arg2, Tensor arg3) -> ()", RegisterOperators::options().kernel<KernelFunc<void, Tensor, Tensor>>(DispatchKey::CPU));
    // 注册操作失败时的错误信息，说明参数个数不匹配，应该有3个参数，但实际传递了2个
    }, "The number of arguments is different. 3 vs 2"
);
TEST(OperatorRegistrationTestFunctorBasedKernel, givenMismatchedKernel_withDifferentArgumentType_whenRegistering_thenFails) {
  // 在注册操作时，使用指定的核函数和调度键来注册一个操作
  RegisterOperators()
      .op("_test::mismatch(Tensor arg1, int arg2) -> int", RegisterOperators::options().kernel<KernelFunc<int64_t, Tensor, int64_t>>(DispatchKey::CPU));

  // 现在是一组不匹配的模式
  expectThrows<c10::Error>([] {
    // 尝试注册一个参数类型与预期不匹配的操作，预期会抛出错误
    RegisterOperators()
        .op("_test::mismatch(Tensor arg1, float arg2) -> int", RegisterOperators::options().kernel<KernelFunc<int64_t, Tensor, int64_t>>(DispatchKey::CPU));
    }, "Type mismatch in argument 2: float vs int"
  );

  expectThrows<c10::Error>([] {
    // 尝试注册一个参数类型与预期不匹配的操作，预期会抛出错误
    RegisterOperators()
        .op("_test::mismatch(int arg1, int arg2) -> int", RegisterOperators::options().kernel<KernelFunc<int64_t, Tensor, int64_t>>(DispatchKey::CPU));
    }, "Type mismatch in argument 1: int vs Tensor"
  );
}

TEST(OperatorRegistrationTestFunctorBasedKernel, givenMismatchedKernel_withDifferentNumReturns_whenRegistering_thenFails) {
  // 在注册操作时，使用指定的核函数和调度键来注册一个操作
  RegisterOperators()
      .op("_test::mismatch(Tensor arg) -> int", RegisterOperators::options().kernel<KernelFunc<int64_t, Tensor>>(DispatchKey::CPU));

  // 现在是一组不匹配的模式
  expectThrows<c10::Error>([] {
    // 尝试注册一个返回值数量与预期不匹配的操作，预期会抛出错误
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> ()", RegisterOperators::options().kernel<KernelFunc<int64_t, Tensor>>(DispatchKey::CPU));
    }, "The number of returns is different. 0 vs 1"
  );

  expectThrows<c10::Error>([] {
    // 尝试注册一个返回值数量与预期不匹配的操作，预期会抛出错误
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> (int, int)", RegisterOperators::options().kernel<KernelFunc<int64_t, Tensor>>(DispatchKey::CPU));
    }, "The number of returns is different. 2 vs 1"
  );

  // 在注册操作时，使用指定的核函数和调度键来注册一个操作
  RegisterOperators()
      .op("_test::mismatch(Tensor arg) -> ()", RegisterOperators::options().kernel<KernelFunc<void, Tensor>>(DispatchKey::CPU));

  // 现在是一组不匹配的模式
  expectThrows<c10::Error>([] {
    // 尝试注册一个返回值数量与预期不匹配的操作，预期会抛出错误
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> Tensor", RegisterOperators::options().kernel<KernelFunc<void, Tensor>>(DispatchKey::CPU));
    }, "The number of returns is different. 1 vs 0"
  );

  expectThrows<c10::Error>([] {
    // 尝试注册一个返回值数量与预期不匹配的操作，预期会抛出错误
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> (Tensor, Tensor)", RegisterOperators::options().kernel<KernelFunc<void, Tensor>>(DispatchKey::CPU));
    }, "The number of returns is different. 2 vs 0"
  );

  // 在注册操作时，使用指定的核函数和调度键来注册一个操作
  RegisterOperators()
      .op("_test::mismatch(Tensor arg) -> (Tensor, Tensor)", RegisterOperators::options().kernel<KernelFunc<std::tuple<Tensor, Tensor>, Tensor>>(DispatchKey::CPU));

  // 现在是一组不匹配的模式
  expectThrows<c10::Error>([] {
    // 注册一个操作符 "_test::mismatch(Tensor arg) -> ()"
    RegisterOperators()
        // 使用 RegisterOperators::options() 创建操作符选项，并设置 CPU 派发的核函数
        .op("_test::mismatch(Tensor arg) -> ()", RegisterOperators::options().kernel<KernelFunc<std::tuple<Tensor, Tensor>, Tensor>>(DispatchKey::CPU));
    // 期望抛出 c10::Error，因为返回值的数量不匹配，预期为 0 个返回值
    }, "The number of returns is different. 0 vs 2"
  );

  // 期望抛出 c10::Error，因为返回值的数量不匹配，预期为 1 个返回值
  expectThrows<c10::Error>([] {
    RegisterOperators()
        // 注册一个操作符 "_test::mismatch(Tensor arg) -> Tensor"
        .op("_test::mismatch(Tensor arg) -> Tensor", RegisterOperators::options().kernel<KernelFunc<std::tuple<Tensor, Tensor>, Tensor>>(DispatchKey::CPU));
    }, "The number of returns is different. 1 vs 2"
  );

  // 期望抛出 c10::Error，因为返回值的数量不匹配，预期为 3 个返回值
  expectThrows<c10::Error>([] {
    RegisterOperators()
        // 注册一个操作符 "_test::mismatch(Tensor arg) -> (Tensor, Tensor, Tensor)"
        .op("_test::mismatch(Tensor arg) -> (Tensor, Tensor, Tensor)", RegisterOperators::options().kernel<KernelFunc<std::tuple<Tensor, Tensor>, Tensor>>(DispatchKey::CPU));
    }, "The number of returns is different. 3 vs 2"
  );
// 定义一个测试函数 OperatorRegistrationTestFunctorBasedKernel，用于测试在注册操作符时处理不匹配的内核
TEST(OperatorRegistrationTestFunctorBasedKernel, givenMismatchedKernel_withDifferentReturnTypes_whenRegistering_thenFails) {
  // 断言此处不会失败，因为类型匹配
  RegisterOperators()
      .op("_test::mismatch(Tensor arg) -> int", RegisterOperators::options().kernel<KernelFunc<int64_t, Tensor>>(DispatchKey::CPU));

  // 然后测试一组不匹配的模式
  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> Tensor", RegisterOperators::options().kernel<KernelFunc<int64_t, Tensor>>(DispatchKey::CPU));
    }, "Type mismatch in return 1: Tensor vs int"
  );

  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> float", RegisterOperators::options().kernel<KernelFunc<int64_t, Tensor>>(DispatchKey::CPU));
    }, "Type mismatch in return 1: float vs int"
  );

  // 断言此处不会失败，因为类型匹配
  RegisterOperators()
      .op("_test::mismatch(Tensor arg) -> Tensor", RegisterOperators::options().kernel<KernelFunc<Tensor, Tensor>>(DispatchKey::CPU));

  // 然后测试一组不匹配的模式
  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> float", RegisterOperators::options().kernel<KernelFunc<Tensor, Tensor>>(DispatchKey::CPU));
    }, "Type mismatch in return 1: float vs Tensor"
  );

  // 断言此处不会失败，因为类型匹配
  RegisterOperators()
      .op("_test::mismatch(Tensor arg) -> (Tensor, int)", RegisterOperators::options().kernel<KernelFunc<std::tuple<Tensor, int64_t>, Tensor>>(DispatchKey::CPU));

  // 然后测试一组不匹配的模式
  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> (Tensor, float)", RegisterOperators::options().kernel<KernelFunc<std::tuple<Tensor, int64_t>, Tensor>>(DispatchKey::CPU));
    }, "Type mismatch in return 2: float vs int"
  );

  expectThrows<c10::Error>([] {
    RegisterOperators()
        .op("_test::mismatch(Tensor arg) -> (int, int)", RegisterOperators::options().kernel<KernelFunc<std::tuple<Tensor, int64_t>, Tensor>>(DispatchKey::CPU));
    }, "Type mismatch in return 1: int vs Tensor"
  );
}
```