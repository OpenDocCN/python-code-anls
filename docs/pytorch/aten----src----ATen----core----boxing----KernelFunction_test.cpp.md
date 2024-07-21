# `.\pytorch\aten\src\ATen\core\boxing\KernelFunction_test.cpp`

```py
#include <gtest/gtest.h> // 包含 Google Test 框架的头文件
#include <ATen/ATen.h> // 包含 ATen 张量库的头文件
#include <ATen/core/boxing/KernelFunction.h> // 包含 ATen 核函数相关的头文件
#include <ATen/core/boxing/impl/test_helpers.h> // 包含 ATen 测试辅助函数的头文件
#include <ATen/core/op_registration/op_registration.h> // 包含 ATen 操作注册相关的头文件

using std::vector; // 使用 std 命名空间中的 vector 类
using std::tuple; // 使用 std 命名空间中的 tuple 类
using std::optional; // 使用 std 命名空间中的 optional 类
using c10::IValue; // 使用 c10 命名空间中的 IValue 类
using c10::OperatorKernel; // 使用 c10 命名空间中的 OperatorKernel 类
using c10::OperatorHandle; // 使用 c10 命名空间中的 OperatorHandle 类
using c10::Stack; // 使用 c10 命名空间中的 Stack 类
using c10::KernelFunction; // 使用 c10 命名空间中的 KernelFunction 类

namespace {

namespace kernels {
// 此命名空间包含几个虚构的核函数。
// 其中一些核函数期望以两个 int64_t 参数调用，并将这些参数存储在 called_with_args 中。
// 核函数可以返回单个值、多个值或无返回值。
// 只返回单个值的核函数返回整数值 5。
// 下面的 expectXXX() 函数利用这些不变性检查调用特定核函数的正确性。

optional<tuple<int64_t, int64_t>> called_with_args;

// 调度程序中的调用约定要求 KernelFunction::call()/callBoxed() 接受 DispatchKeySet。
// 对于不需要 DispatchKeySet 参数的所有测试来说，这个值本身是无意义的。
// 有关详细信息，请参阅“Plumbing Keys Through The Dispatcher”注释。
c10::DispatchKeySet CPU_TEST_SET = c10::DispatchKeySet(c10::DispatchKey::CPU);

void boxed_func_with_return(const OperatorHandle& /*opHandle*/, Stack* stack) {
  EXPECT_EQ(2, stack->size()); // 断言堆栈大小为2
  EXPECT_TRUE(stack->at(0).isInt()); // 断言第一个元素是整数类型
  EXPECT_TRUE(stack->at(1).isInt()); // 断言第二个元素是整数类型
  called_with_args = tuple<int64_t, int64_t>(stack->at(0).toInt(), stack->at(1).toInt()); // 存储调用参数

  stack->clear(); // 清空堆栈
  stack->push_back(5); // 压入整数值 5
}

void boxed_func_without_return(const OperatorHandle& /*opHandle*/, Stack* stack) {
  EXPECT_EQ(2, stack->size()); // 断言堆栈大小为2
  EXPECT_TRUE(stack->at(0).isInt()); // 断言第一个元素是整数类型
  EXPECT_TRUE(stack->at(1).isInt()); // 断言第二个元素是整数类型
  called_with_args = tuple<int64_t, int64_t>(stack->at(0).toInt(), stack->at(1).toInt()); // 存储调用参数

  stack->clear(); // 清空堆栈
}

void boxed_func_with_multi_return(const OperatorHandle& /*opHandle*/, Stack* stack) {
  EXPECT_EQ(2, stack->size()); // 断言堆栈大小为2
  EXPECT_TRUE(stack->at(0).isInt()); // 断言第一个元素是整数类型
  int64_t a = stack->at(0).toInt(); // 将第一个元素转换为 int64_t 类型
  EXPECT_TRUE(stack->at(1).isInt()); // 断言第二个元素是整数类型
  int64_t b = stack->at(1).toInt(); // 将第二个元素转换为 int64_t 类型
  called_with_args = tuple<int64_t, int64_t>(a, b); // 存储调用参数

  stack->clear(); // 清空堆栈
  torch::jit::push(stack, a + b); // 压入 a + b 的结果
  torch::jit::push(stack, a * b); // 压入 a * b 的结果
}

struct unboxed_functor_with_return final : OperatorKernel {
  int64_t operator()(int64_t a, int64_t b) {
    called_with_args = tuple<int64_t, int64_t>(a, b); // 存储调用参数
    return 5; // 返回整数值 5
  }
};

struct unboxed_functor_without_return final : OperatorKernel {
  void operator()(int64_t a, int64_t b) {
    called_with_args = tuple<int64_t, int64_t>(a, b); // 存储调用参数
  }
};

struct unboxed_functor_with_return_factory final {
  std::unique_ptr<OperatorKernel> operator()() {
    return std::make_unique<unboxed_functor_with_return>(); // 返回带返回值的 OperatorKernel 实例
  }
};

struct unboxed_functor_without_return_factory final {
  std::unique_ptr<OperatorKernel> operator()() {
    return std::make_unique<unboxed_functor_without_return>(); // 返回无返回值的 OperatorKernel 实例
  }
};
// 定义一个返回 int64_t 类型的函数，接受两个 int64_t 类型参数，并返回 5
int64_t unboxed_function_with_return(int64_t a, int64_t b) {
  // 将参数 a 和 b 封装成一个元组
  called_with_args = tuple<int64_t, int64_t>(a, b);
  // 返回整数 5
  return 5;
}

// 定义一个没有返回值的函数，接受两个 int64_t 类型参数
void unboxed_function_without_return(int64_t a, int64_t b) {
  // 将参数 a 和 b 封装成一个元组
  called_with_args = tuple<int64_t, int64_t>(a, b);
}

// 定义一个返回 int64_t 类型的 Lambda 函数，接受两个 int64_t 类型参数，并返回 5
auto unboxed_lambda_with_return = [] (int64_t a, int64_t b) -> int64_t {
  // 将参数 a 和 b 封装成一个元组
  called_with_args = tuple<int64_t, int64_t>(a, b);
  // 返回整数 5
  return 5;
};

// 定义一个没有返回值的 Lambda 函数，接受两个 int64_t 类型参数
auto unboxed_lambda_without_return = [] (int64_t a, int64_t b) -> void{
  // 将参数 a 和 b 封装成一个元组
  called_with_args = tuple<int64_t, int64_t>(a, b);
};

// 返回一个 OperatorHandle 对象，用于注册一个名为 "my::dummy() -> ()" 的操作
OperatorHandle makeDummyOperatorHandle() {
  static auto registry = torch::RegisterOperators().op("my::dummy() -> ()");
  // 查找并返回名称为 "my::dummy" 的操作的模式
  return c10::Dispatcher::singleton().findSchema({"my::dummy", ""}).value();
}

//
// 以下是一些返回引用到张量参数的箱化核心，类似于 inplace/outplace 核心
//

// 对于 inplace 操作的箱化函数，接受操作句柄和堆栈作为参数
void boxed_func_for_inplace_op(const OperatorHandle& /*opHandle*/, Stack* stack) {
  // 确保堆栈的大小为 2
  EXPECT_EQ(2, stack->size());

  // 确保第一个元素是张量
  ASSERT_TRUE(stack->at(0).isTensor());
  auto t = stack->at(0).toTensor();

  // 确保第二个元素是标量
  ASSERT_TRUE(stack->at(1).isScalar());
  auto s = stack->at(1).toScalar();

  // 在张量上执行 inplace 加法操作
  t.add_(s);

  // 清空堆栈
  stack->clear();
  // 将修改后的张量压入堆栈
  torch::jit::push(stack, t);
}

// 对于 out-of-place 操作的箱化函数，接受操作句柄和堆栈作为参数
void boxed_func_for_outofplace_op(const OperatorHandle& /*opHandle*/, Stack* stack) {
  // 确保堆栈的大小为 2
  EXPECT_EQ(2, stack->size());

  // 确保第一个元素是标量
  ASSERT_TRUE(stack->at(0).isScalar());
  auto s = stack->at(0).toScalar();

  // 确保第二个元素是张量
  ASSERT_TRUE(stack->at(1).isTensor());
  auto t = stack->at(1).toTensor();

  // 在张量上执行 out-of-place 加法操作
  t.add_(s);

  // 清空堆栈
  stack->clear();
  // 将修改后的张量压入堆栈
  torch::jit::push(stack, t);
}

// 对于多个 out-of-place 操作的箱化函数，接受操作句柄和堆栈作为参数
void boxed_func_for_outofplace_multi_op(const OperatorHandle& /*opHandle*/, Stack* stack) {
  // 确保堆栈的大小为 4
  EXPECT_EQ(4, stack->size());

  // 确保前两个元素是标量
  ASSERT_TRUE(stack->at(0).isScalar());
  auto s1 = stack->at(0).toScalar();

  ASSERT_TRUE(stack->at(1).isScalar());
  auto s2 = stack->at(1).toScalar();

  // 确保后两个元素是张量
  ASSERT_TRUE(stack->at(2).isTensor());
  auto t1 = stack->at(2).toTensor();

  ASSERT_TRUE(stack->at(3).isTensor());
  auto t2 = stack->at(3).toTensor();

  // 在张量上执行 out-of-place 加法操作
  t1.add_(s1);
  t2.add_(s2);

  // 清空堆栈
  stack->clear();
  // 将修改后的张量压入堆栈
  torch::jit::push(stack, t1);
  torch::jit::push(stack, t2);
}

//
// 箱化调用的测试
//

// 测试函数，用于验证带返回值的箱化调用是否正常工作，接受一个 KernelFunction 参数
void expectBoxedCallingWithReturnWorks(const KernelFunction& func) {
  // 将 called_with_args 初始化为 null
  called_with_args = c10::nullopt;
  // 创建一个包含两个整数的 IValue 向量
  vector<IValue> stack {3, 4};
  // 创建一个虚拟的操作句柄
  OperatorHandle dummy = makeDummyOperatorHandle();

  // 调用指定的箱化函数
  func.callBoxed(dummy, CPU_TEST_SET, &stack);

  // 断言 called_with_args 已被赋值
  EXPECT_TRUE(called_with_args.has_value());
  // 断言调用参数与预期的元组相符
  EXPECT_EQ((tuple<int64_t, int64_t>(3, 4)), *called_with_args);
  // 断言堆栈中只有一个元素
  EXPECT_EQ(1, stack.size());
  // 断言堆栈中的元素为整数类型
  EXPECT_TRUE(stack[0].isInt());
  // 断言堆栈中的整数为 5
  EXPECT_EQ(5, stack[0].toInt());
}
// 在不返回结果的情况下期望盒子调用正常工作
void expectBoxedCallingWithoutReturnWorks(const KernelFunction& func) {
  // 初始化一个空的参数调用
  called_with_args = c10::nullopt;
  // 创建包含两个整数的堆栈
  vector<IValue> stack {3, 4};
  // 创建一个虚拟的操作符句柄
  OperatorHandle dummy = makeDummyOperatorHandle();

  // 执行盒子调用
  func.callBoxed(dummy, CPU_TEST_SET, &stack);

  // 断言是否已成功调用
  EXPECT_TRUE(called_with_args.has_value());
  // 断言参数是否被正确传递
  EXPECT_EQ((tuple<int64_t, int64_t>(3, 4)), *called_with_args);
  // 断言堆栈是否为空
  EXPECT_EQ(0, stack.size());
}

// 在返回多个值的情况下期望盒子调用正常工作
void expectBoxedCallingWithMultiReturnWorks(const KernelFunction& func) {
  // 初始化一个空的参数调用
  called_with_args = c10::nullopt;
  // 创建包含两个整数的堆栈
  vector<IValue> stack {3, 4};
  // 创建一个虚拟的操作符句柄
  OperatorHandle dummy = makeDummyOperatorHandle();

  // 执行盒子调用
  func.callBoxed(dummy, CPU_TEST_SET, &stack);

  // 断言是否已成功调用
  EXPECT_TRUE(called_with_args.has_value());
  // 断言参数是否被正确传递
  EXPECT_EQ((tuple<int64_t, int64_t>(3, 4)), *called_with_args);
  // 断言堆栈的大小是否为2
  EXPECT_EQ(2, stack.size());

  // 断言第一个返回值是整数7
  EXPECT_TRUE(stack[0].isInt());
  EXPECT_EQ(7, stack[0].toInt());

  // 断言第二个返回值是整数12
  EXPECT_TRUE(stack[1].isInt());
  EXPECT_EQ(12, stack[1].toInt());
}

// in/out

// 期望就地盒子调用正常工作
void expectInPlaceBoxedCallingWorks(const KernelFunction& func) {
  // 创建一个虚拟的操作符句柄
  OperatorHandle dummy = makeDummyOperatorHandle();

  // 创建一个大小为1的零张量
  auto t = at::zeros({1});
  // 创建一个浮点数1.0
  auto s = 1.0f;
  // 创建包含张量和浮点数的堆栈
  vector<IValue> stack {t, s};
  
  // 执行盒子调用
  func.callBoxed(dummy, CPU_TEST_SET, &stack);

  // 核函数应该已更新输出参数并将其返回
  EXPECT_EQ(t.item().toFloat(), 1.0f);
  EXPECT_EQ(1, stack.size());
  EXPECT_TRUE(stack[0].isTensor());
  EXPECT_TRUE(stack[0].toTensor().is_same(t));
}

// 期望非就地盒子调用正常工作
void expectOutOfPlaceBoxedCallingWorks(const KernelFunction& func) {
  // 创建一个虚拟的操作符句柄
  OperatorHandle dummy = makeDummyOperatorHandle();

  // 创建一个浮点数1.0
  auto s = 1.0f;
  // 创建一个大小为1的零张量
  auto t = at::zeros({1});
  // 创建包含浮点数和张量的堆栈
  vector<IValue> stack {s, t};
  
  // 执行盒子调用
  func.callBoxed(dummy, CPU_TEST_SET, &stack);

  // 核函数应该已更新输出参数并在堆栈上返回它
  EXPECT_EQ(t.item().toFloat(), 1.0f);
  EXPECT_EQ(1, stack.size());
  EXPECT_TRUE(stack[0].isTensor());
  EXPECT_TRUE(stack[0].toTensor().is_same(t));
}

// 期望非就地多输出盒子调用正常工作
void expectOutOfPlaceMultiBoxedCallingWorks(const KernelFunction& func) {
  // 创建一个虚拟的操作符句柄
  OperatorHandle dummy = makeDummyOperatorHandle();

  // 创建两个浮点数和两个大小为1的零张量
  auto s1 = 1.0f;
  auto s2 = 2.0f;
  auto t1 = at::zeros({1});
  auto t2 = at::zeros({1});
  // 创建包含四个元素的堆栈
  vector<IValue> stack {s1, s2, t1, t2};
  
  // 执行盒子调用
  func.callBoxed(dummy, CPU_TEST_SET, &stack);

  // 核函数应该已更新输出参数并在堆栈上返回它们
  EXPECT_EQ(t1.item().toFloat(), 1.0f);
  EXPECT_EQ(t2.item().toFloat(), 2.0f);
  EXPECT_EQ(2, stack.size());
  EXPECT_TRUE(stack[0].isTensor());
  EXPECT_TRUE(stack[0].toTensor().is_same(t1));
  EXPECT_TRUE(stack[1].isTensor());
  EXPECT_TRUE(stack[1].toTensor().is_same(t2));
}

//
// unboxed calling tests:
//

// 期望无盒子返回值正常工作的非盒子调用
void expectUnboxedCallingWithReturnWorks(const KernelFunction& func) {
  // 初始化一个空的参数调用
  called_with_args = c10::nullopt;
  // 创建一个虚拟的操作符句柄
  OperatorHandle dummy = makeDummyOperatorHandle();

  // 执行无盒子返回值的非盒子调用，期望返回值为int64_t
  int64_t result = func.call<int64_t, int64_t, int64_t>(dummy, CPU_TEST_SET, 3, 4);

  // 断言是否已成功调用
  EXPECT_TRUE(called_with_args.has_value());
  // 断言参数是否被正确传递
  EXPECT_EQ((tuple<int64_t, int64_t>(3, 4)), *called_with_args);
  // 断言结果是否为5
  EXPECT_EQ(5, result);
}
// make an unboxed call to a kernel that returns nothing.
//
void expectUnboxedCallingWithoutReturnWorks(const KernelFunction& func) {
  called_with_args = c10::nullopt;  // 初始化一个空的optional变量，用于记录调用参数
  OperatorHandle dummy = makeDummyOperatorHandle();  // 创建一个虚拟的操作符句柄

  func.call<void, int64_t, int64_t>(dummy, CPU_TEST_SET, 3, 4);  // 调用内核函数，传入两个int64_t参数，无返回值

  EXPECT_TRUE(called_with_args.has_value());  // 断言检查是否记录了调用的参数
  EXPECT_EQ((tuple<int64_t, int64_t>(3, 4)), *called_with_args);  // 断言检查记录的参数值是否正确
}

// make an unboxed call to a kernel that returns two values.
// When calling unboxed, multiple values are returned as a tuple.
//
void expectUnboxedCallingWithMultiReturnWorks(const KernelFunction& func) {
  called_with_args = c10::nullopt;  // 初始化一个空的optional变量，用于记录调用参数
  OperatorHandle dummy = makeDummyOperatorHandle();  // 创建一个虚拟的操作符句柄

  auto result = func.call<std::tuple<int64_t, int64_t>, int64_t, int64_t>(dummy, CPU_TEST_SET, 3, 4);  // 调用内核函数，传入两个int64_t参数，返回一个包含两个int64_t值的tuple

  EXPECT_TRUE(called_with_args.has_value());  // 断言检查是否记录了调用的参数
  EXPECT_EQ((tuple<int64_t, int64_t>(3, 4)), *called_with_args);  // 断言检查记录的参数值是否正确

  EXPECT_EQ((tuple<int64_t, int64_t>(7, 12)), result);  // 断言检查返回的tuple值是否正确
}

// in/out

void expectInPlaceUnboxedCallingWorks(const KernelFunction& func) {
  OperatorHandle dummy = makeDummyOperatorHandle();  // 创建一个虚拟的操作符句柄

  auto t = at::zeros({1});  // 创建一个大小为1的零张量
  at::Tensor& t_out = func.call<at::Tensor&, at::Tensor&, at::Scalar>(dummy, CPU_TEST_SET, t, 1.0f);  // 调用内核函数，在原地更新第一个参数张量，并返回更新后的张量

  // should have updated first arg and returned it
  EXPECT_EQ(t.item().toFloat(), 1.0f);  // 断言检查原地更新的张量值是否为1.0f
  EXPECT_EQ(&t, &t_out);  // 断言检查返回的张量是否和原始张量是同一个对象
}

void expectOutOfPlaceUnboxedCallingWorks(const KernelFunction& func) {
  OperatorHandle dummy = makeDummyOperatorHandle();  // 创建一个虚拟的操作符句柄

  auto t = at::zeros({1});  // 创建一个大小为1的零张量
  at::Tensor& t_out = func.call<at::Tensor&, at::Scalar, at::Tensor&>(dummy, CPU_TEST_SET, 1.0f, t);  // 调用内核函数，在外部更新第二个参数张量，并返回更新后的张量

  // should have updated out arg and returned it
  EXPECT_EQ(t.item().toFloat(), 1.0f);  // 断言检查外部更新的张量值是否为1.0f
  EXPECT_EQ(&t, &t_out);  // 断言检查返回的张量是否和原始张量是同一个对象
}

void expectOutOfPlaceMultiUnboxedCallingWorks(const KernelFunction& func) {
  OperatorHandle dummy = makeDummyOperatorHandle();  // 创建一个虚拟的操作符句柄

  auto s1 = 1.0f;  // 创建浮点数变量s1，并赋值为1.0f
  auto s2 = 2.0f;  // 创建浮点数变量s2，并赋值为2.0f
  auto t1 = at::zeros({1});  // 创建一个大小为1的零张量t1
  auto t2 = at::zeros({1});  // 创建一个大小为1的零张量t2

  std::tuple<at::Tensor&, at::Tensor&> tup = func.call<
    std::tuple<at::Tensor&, at::Tensor&>, at::Scalar, at::Scalar, at::Tensor&, at::Tensor&
  >(dummy, CPU_TEST_SET, s1, s2, t1, t2);  // 调用内核函数，传入两个浮点数和两个张量参数，并返回一个包含两个张量引用的tuple

  // kernel should have updated out args and returned them in a tuple
  EXPECT_EQ(t1.item().toFloat(), 1.0f);  // 断言检查第一个张量的值是否更新为1.0f
  EXPECT_EQ(t2.item().toFloat(), 2.0f);  // 断言检查第二个张量的值是否更新为2.0f

  auto t1_out = std::get<0>(tup);  // 获取tuple中第一个张量引用
  EXPECT_EQ(t1_out.item().toFloat(), 1.0f);  // 断言检查返回的第一个张量引用的值是否为1.0f
  EXPECT_TRUE(t1_out.is_same(t1));  // 断言检查返回的第一个张量引用是否和原始t1张量是同一个对象

  auto t2_out = std::get<1>(tup);  // 获取tuple中第二个张量引用
  EXPECT_EQ(t2_out.item().toFloat(), 2.0f);  // 断言检查返回的第二个张量引用的值是否为2.0f
  EXPECT_TRUE(t2_out.is_same(t2));  // 断言检查返回的第二个张量引用是否和原始t2张量是同一个对象
}
TEST(KernelFunctionTest, givenBoxedFunction_withoutReturn_whenCallingBoxed_thenWorks) {
  // 使用 KernelFunction 的静态方法从 boxed_func_without_return 函数生成 KernelFunction 对象
  KernelFunction func = KernelFunction::makeFromBoxedFunction<&kernels::boxed_func_without_return>();
  // 调用函数验证 boxed_func_without_return 的预期行为
  kernels::expectBoxedCallingWithoutReturnWorks(func);
}

TEST(KernelFunctionTest, givenBoxedFunction_withMultiReturn_whenCallingBoxed_thenWorks) {
  // 使用 KernelFunction 的静态方法从 boxed_func_with_multi_return 函数生成 KernelFunction 对象
  KernelFunction func = KernelFunction::makeFromBoxedFunction<&kernels::boxed_func_with_multi_return>();
  // 调用函数验证 boxed_func_with_multi_return 的预期行为
  kernels::expectBoxedCallingWithMultiReturnWorks(func);
}

// in/out, boxed calling

TEST(KernelFunctionTest, givenBoxedFunction_withInPlaceSignature_whenCallingBoxed_thenWorks) {
  // 使用 KernelFunction 的静态方法从 boxed_func_for_inplace_op 函数生成 KernelFunction 对象
  KernelFunction func = KernelFunction::makeFromBoxedFunction<&kernels::boxed_func_for_inplace_op>();
  // 调用函数验证 boxed_func_for_inplace_op 的预期 inplace 行为
  kernels::expectInPlaceBoxedCallingWorks(func);
}

TEST(KernelFunctionTest, givenBoxedFunction_withOutOfPlaceSignature_whenCallingBoxed_thenWorks) {
  // 使用 KernelFunction 的静态方法从 boxed_func_for_outofplace_op 函数生成 KernelFunction 对象
  KernelFunction func = KernelFunction::makeFromBoxedFunction<&kernels::boxed_func_for_outofplace_op>();
  // 调用函数验证 boxed_func_for_outofplace_op 的预期 out-of-place 行为
  kernels::expectOutOfPlaceBoxedCallingWorks(func);
}

TEST(KernelFunctionTest, givenBoxedFunction_withOutOfPlaceMultiSignature_whenCallingBoxed_thenWorks) {
  // 使用 KernelFunction 的静态方法从 boxed_func_for_outofplace_multi_op 函数生成 KernelFunction 对象
  KernelFunction func = KernelFunction::makeFromBoxedFunction<&kernels::boxed_func_for_outofplace_multi_op>();
  // 调用函数验证 boxed_func_for_outofplace_multi_op 的预期 out-of-place 多重签名行为
  kernels::expectOutOfPlaceMultiBoxedCallingWorks(func);
}

// functional, unboxed calling

TEST(KernelFunctionTest, givenBoxedFunction_withReturn_whenCallingUnboxed_thenWorks) {
  // 使用 KernelFunction 的静态方法从 boxed_func_with_return 函数生成 KernelFunction 对象
  KernelFunction func = KernelFunction::makeFromBoxedFunction<&kernels::boxed_func_with_return>();
  // 调用函数验证 boxed_func_with_return 的预期 unboxed 返回行为
  kernels::expectUnboxedCallingWithReturnWorks(func);
}

TEST(KernelFunctionTest, givenBoxedFunction_withoutReturn_whenCallingUnboxed_thenWorks) {
  // 使用 KernelFunction 的静态方法从 boxed_func_without_return 函数生成 KernelFunction 对象
  KernelFunction func = KernelFunction::makeFromBoxedFunction<&kernels::boxed_func_without_return>();
  // 调用函数验证 boxed_func_without_return 的预期 unboxed 无返回行为
  kernels::expectUnboxedCallingWithoutReturnWorks(func);
}

TEST(KernelFunctionTest, givenBoxedFunction_withMultiReturn_whenCallingUnboxed_thenWorks) {
  // 使用 KernelFunction 的静态方法从 boxed_func_with_multi_return 函数生成 KernelFunction 对象
  KernelFunction func = KernelFunction::makeFromBoxedFunction<&kernels::boxed_func_with_multi_return>();
  // 调用函数验证 boxed_func_with_multi_return 的预期 unboxed 多返回行为
  kernels::expectUnboxedCallingWithMultiReturnWorks(func);
}

// in/out, unboxed calling

TEST(KernelFunctionTest, givenBoxedFunction_withInPlaceSignature_whenCallingUnboxed_thenWorks) {
  // 使用 KernelFunction 的静态方法从 boxed_func_for_inplace_op 函数生成 KernelFunction 对象
  KernelFunction func = KernelFunction::makeFromBoxedFunction<&kernels::boxed_func_for_inplace_op>();
  // 调用函数验证 boxed_func_for_inplace_op 的预期 inplace unboxed 行为
  kernels::expectInPlaceUnboxedCallingWorks(func);
}

TEST(KernelFunctionTest, givenBoxedFunction_withOutOfPlaceSignature_whenCallingUnboxed_thenWorks) {
  // 使用 KernelFunction 的静态方法从 boxed_func_for_outofplace_op 函数生成 KernelFunction 对象
  KernelFunction func = KernelFunction::makeFromBoxedFunction<&kernels::boxed_func_for_outofplace_op>();
  // 调用函数验证 boxed_func_for_outofplace_op 的预期 out-of-place unboxed 行为
  kernels::expectOutOfPlaceUnboxedCallingWorks(func);
}

TEST(KernelFunctionTest, givenBoxedFunction_withOutOfPlaceMultiSignature_whenCallingUnboxed_thenWorks) {
  // 使用 KernelFunction 的静态方法从 boxed_func_for_outofplace_multi_op 函数生成 KernelFunction 对象
  KernelFunction func = KernelFunction::makeFromBoxedFunction<&kernels::boxed_func_for_outofplace_multi_op>();
  // 调用函数验证 boxed_func_for_outofplace_multi_op 的预期 out-of-place 多重签名 unboxed 行为
  kernels::expectOutOfPlaceMultiUnboxedCallingWorks(func);
}

// functors etc.
TEST(KernelFunctionTest, givenUnboxedFunctor_withReturn_whenCallingBoxed_thenWorks) {
  // 创建一个 KernelFunction 对象，从带有返回值的未装箱的函数对象创建
  KernelFunction func = KernelFunction::makeFromUnboxedFunctor<false, kernels::unboxed_functor_with_return>(std::unique_ptr<OperatorKernel>(std::make_unique<kernels::unboxed_functor_with_return>()));
  // 调用函数，验证带返回值的装箱调用工作正常
  kernels::expectBoxedCallingWithReturnWorks(func);
}

TEST(KernelFunctionTest, givenUnboxedFunctor_withoutReturn_whenCallingBoxed_thenWorks) {
  // 创建一个 KernelFunction 对象，从不带返回值的未装箱的函数对象创建
  KernelFunction func = KernelFunction::makeFromUnboxedFunctor<false, kernels::unboxed_functor_without_return>(std::unique_ptr<OperatorKernel>(std::make_unique<kernels::unboxed_functor_without_return>()));
  // 调用函数，验证不带返回值的装箱调用工作正常
  kernels::expectBoxedCallingWithoutReturnWorks(func);
}

TEST(KernelFunctionTest, givenUnboxedFunctor_withReturn_whenCallingUnboxed_thenWorks) {
  // 创建一个 KernelFunction 对象，从带有返回值的未装箱的函数对象创建
  KernelFunction func = KernelFunction::makeFromUnboxedFunctor<false, kernels::unboxed_functor_with_return>(std::unique_ptr<OperatorKernel>(std::make_unique<kernels::unboxed_functor_with_return>()));
  // 调用函数，验证带返回值的未装箱调用工作正常
  kernels::expectUnboxedCallingWithReturnWorks(func);
}

TEST(KernelFunctionTest, givenUnboxedFunctor_withoutReturn_whenCallingUnboxed_thenWorks) {
  // 创建一个 KernelFunction 对象，从不带返回值的未装箱的函数对象创建
  KernelFunction func = KernelFunction::makeFromUnboxedFunctor<false, kernels::unboxed_functor_without_return>(std::unique_ptr<OperatorKernel>(std::make_unique<kernels::unboxed_functor_without_return>()));
  // 调用函数，验证不带返回值的未装箱调用工作正常
  kernels::expectUnboxedCallingWithoutReturnWorks(func);
}

TEST(KernelFunctionTest, givenUnboxedFunction_withReturn_whenCallingBoxed_thenWorks) {
  // 创建一个 KernelFunction 对象，从带返回值的未装箱函数创建
  KernelFunction func = KernelFunction::makeFromUnboxedFunction(TORCH_FN(kernels::unboxed_function_with_return));
  // 调用函数，验证带返回值的装箱调用工作正常
  kernels::expectBoxedCallingWithReturnWorks(func);
}

TEST(KernelFunctionTest, givenUnboxedFunction_withoutReturn_whenCallingBoxed_thenWorks) {
  // 创建一个 KernelFunction 对象，从不带返回值的未装箱函数创建
  KernelFunction func = KernelFunction::makeFromUnboxedFunction(TORCH_FN(kernels::unboxed_function_without_return));
  // 调用函数，验证不带返回值的装箱调用工作正常
  kernels::expectBoxedCallingWithoutReturnWorks(func);
}

TEST(KernelFunctionTest, givenUnboxedFunction_withReturn_whenCallingUnboxed_thenWorks) {
  // 创建一个 KernelFunction 对象，从带返回值的未装箱函数创建
  KernelFunction func = KernelFunction::makeFromUnboxedFunction(TORCH_FN(kernels::unboxed_function_with_return));
  // 调用函数，验证带返回值的未装箱调用工作正常
  kernels::expectUnboxedCallingWithReturnWorks(func);
}

TEST(KernelFunctionTest, givenUnboxedFunction_withoutReturn_whenCallingUnboxed_thenWorks) {
  // 创建一个 KernelFunction 对象，从不带返回值的未装箱函数创建
  KernelFunction func = KernelFunction::makeFromUnboxedFunction(TORCH_FN(kernels::unboxed_function_without_return));
  // 调用函数，验证不带返回值的未装箱调用工作正常
  kernels::expectUnboxedCallingWithoutReturnWorks(func);
}

TEST(KernelFunctionTest, givenUnboxedRuntimeFunction_withReturn_whenCallingBoxed_thenWorks) {
  // 创建一个 KernelFunction 对象，从带返回值的运行时未装箱函数创建
  KernelFunction func = KernelFunction::makeFromUnboxedRuntimeFunction(&kernels::unboxed_function_with_return);
  // 调用函数，验证带返回值的装箱调用工作正常
  kernels::expectBoxedCallingWithReturnWorks(func);
}

TEST(KernelFunctionTest, givenUnboxedRuntimeFunction_withoutReturn_whenCallingBoxed_thenWorks) {
  // 创建一个 KernelFunction 对象，从不带返回值的运行时未装箱函数创建
  KernelFunction func = KernelFunction::makeFromUnboxedRuntimeFunction(&kernels::unboxed_function_without_return);
  // 调用函数，验证不带返回值的装箱调用工作正常
  kernels::expectBoxedCallingWithoutReturnWorks(func);
}
# 定义一个测试用例，测试使用未装箱的运行时函数并带有返回值时的情况
TEST(KernelFunctionTest, givenUnboxedRuntimeFunction_withReturn_whenCallingUnboxed_thenWorks) {
    # 从未装箱的运行时函数创建 KernelFunction 对象
    KernelFunction func = KernelFunction::makeFromUnboxedRuntimeFunction(&kernels::unboxed_function_with_return);
    # 调用期望未装箱调用带返回值函数正常工作的测试函数
    kernels::expectUnboxedCallingWithReturnWorks(func);
}

# 定义一个测试用例，测试使用未装箱的运行时函数并没有返回值时的情况
TEST(KernelFunctionTest, givenUnboxedRuntimeFunction_withoutReturn_whenCallingUnboxed_thenWorks) {
    # 从未装箱的运行时函数创建 KernelFunction 对象
    KernelFunction func = KernelFunction::makeFromUnboxedRuntimeFunction(&kernels::unboxed_function_without_return);
    # 调用期望未装箱调用没有返回值函数正常工作的测试函数
    kernels::expectUnboxedCallingWithoutReturnWorks(func);
}

# 定义一个测试用例，测试使用未装箱的 lambda 函数并带有返回值时的情况
TEST(KernelFunctionTest, givenUnboxedLambda_withReturn_whenCallingBoxed_thenWorks) {
    # 从未装箱的 lambda 函数创建 KernelFunction 对象
    KernelFunction func = KernelFunction::makeFromUnboxedLambda(kernels::unboxed_lambda_with_return);
    # 调用期望装箱调用带返回值函数正常工作的测试函数
    kernels::expectBoxedCallingWithReturnWorks(func);
}

# 定义一个测试用例，测试使用未装箱的 lambda 函数并没有返回值时的情况
TEST(KernelFunctionTest, givenUnboxedLambda_withoutReturn_whenCallingBoxed_thenWorks) {
    # 从未装箱的 lambda 函数创建 KernelFunction 对象
    KernelFunction func = KernelFunction::makeFromUnboxedLambda(kernels::unboxed_lambda_without_return);
    # 调用期望装箱调用没有返回值函数正常工作的测试函数
    kernels::expectBoxedCallingWithoutReturnWorks(func);
}

# 定义一个测试用例，测试使用未装箱的 lambda 函数并带有返回值时的情况
TEST(KernelFunctionTest, givenUnboxedLambda_withReturn_whenCallingUnboxed_thenWorks) {
    # 从未装箱的 lambda 函数创建 KernelFunction 对象
    KernelFunction func = KernelFunction::makeFromUnboxedLambda(kernels::unboxed_lambda_with_return);
    # 调用期望未装箱调用带返回值函数正常工作的测试函数
    kernels::expectUnboxedCallingWithReturnWorks(func);
}

# 定义一个测试用例，测试使用未装箱的 lambda 函数并没有返回值时的情况
TEST(KernelFunctionTest, givenUnboxedLambda_withoutReturn_whenCallingUnboxed_thenWorks) {
    # 从未装箱的 lambda 函数创建 KernelFunction 对象
    KernelFunction func = KernelFunction::makeFromUnboxedLambda(kernels::unboxed_lambda_without_return);
    # 调用期望未装箱调用没有返回值函数正常工作的测试函数
    kernels::expectUnboxedCallingWithoutReturnWorks(func);
}
```