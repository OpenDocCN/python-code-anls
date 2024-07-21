# `.\pytorch\test\cpp\jit\test_interpreter.cpp`

```
#include <gmock/gmock.h>  // 引入 Google Mock 框架的头文件
#include <gtest/gtest.h>  // 引入 Google Test 框架的头文件

#include <ATen/Parallel.h>  // 引入 ATen 并行计算相关头文件
#include <c10/core/DeviceType.h>  // 引入 c10 设备类型相关头文件
#include <test/cpp/jit/test_utils.h>  // 引入 Torch JIT 测试工具相关头文件
#include <torch/csrc/jit/runtime/instruction.h>  // 引入 Torch JIT 运行时指令头文件
#include <torch/jit.h>  // 引入 Torch JIT 模块头文件
#include <torch/script.h>  // 引入 Torch 脚本模块头文件
#include <torch/torch.h>  // 引入 Torch 核心头文件

namespace torch {
namespace jit {

class TypeCheckTest : public ::testing::Test {
 protected:
  TypeCheckTest() : interp(makeInterp()) {}

  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  InterpreterState interp;

 private:
  static InterpreterState makeInterp() {
    auto graph = std::make_shared<Graph>();  // 创建共享指针 graph 对象
    std::unordered_map<std::string, Value*> vmap;  // 创建无序映射 vmap

    // 解析输入的 IR 表达式并填充 graph 和 vmap
    parseIR(
        R"IR(
graph(%a.1 : Tensor,
      %b.1 : Tensor):
  %t0 : Float(2, 2, strides=[2, 1], device=cpu, requires_grad=1), %t1 : Float(3, 3, strides=[3, 1]), %type_matched : bool = prim::TypeCheck[types=[Float(2, 2, strides=[2, 1], device=cpu, requires_grad=1), Float(3, 3, strides=[3, 1])]](%a.1, %b.1)
  return (%t0, %t1, %type_matched)
  )IR",
        &*graph,
        vmap);

    Code function(graph, "");  // 创建 Code 对象 function，使用 graph 初始化
    return InterpreterState(function);  // 返回 InterpreterState 对象
  }
};

TEST_F(TypeCheckTest, MatchingType) {
  // TypeCheck yields to true! Shape, grad and device matches.
  auto a = at::zeros({2, 2}, at::kFloat);  // 创建大小为 (2, 2) 的 Float 类型张量 a
  auto b = at::ones({3, 3}, at::kFloat);   // 创建大小为 (3, 3) 的 Float 类型张量 b
  a.set_requires_grad(true);  // 设置张量 a 需要梯度计算
  a = a.to(at::kCPU);  // 将张量 a 移动到 CPU 上
  std::vector<IValue> stack({a, b});  // 创建包含 a 和 b 的 IValue 向量 stack
  interp.run(stack);  // 运行 interp 对象处理 stack 中的数据
  ASSERT_TRUE(exactlyEqual(stack[0].toTensor(), a));  // 断言 stack 中第一个元素与 a 相等
  ASSERT_TRUE(exactlyEqual(stack[1].toTensor(), b));  // 断言 stack 中第二个元素与 b 相等
  ASSERT_TRUE(stack[2].toBool());  // 断言 stack 中第三个元素为 true
}

TEST_F(TypeCheckTest, SizeMismatch) {
  auto a = at::zeros({2, 2}, at::kFloat);  // 创建大小为 (2, 2) 的 Float 类型张量 a
  auto b = at::ones({2, 2}, at::kFloat);   // 创建大小为 (2, 2) 的 Float 类型张量 b，大小不匹配
  a.set_requires_grad(true);  // 设置张量 a 需要梯度计算
  a = a.to(at::kCPU);  // 将张量 a 移动到 CPU 上
  std::vector<IValue> stack({a, b});  // 创建包含 a 和 b 的 IValue 向量 stack
  interp.run(stack);  // 运行 interp 对象处理 stack 中的数据
  ASSERT_FALSE(stack[2].toBool());  // 断言 stack 中第三个元素为 false
}

TEST_F(TypeCheckTest, GradientMismatch) {
  auto a = at::zeros({2, 2}, at::kFloat);  // 创建大小为 (2, 2) 的 Float 类型张量 a
  auto b = at::ones({3, 3}, at::kFloat);   // 创建大小为 (3, 3) 的 Float 类型张量 b
  a = a.to(at::kCPU);  // 将张量 a 移动到 CPU 上
  a.set_requires_grad(false);  // 设置张量 a 不需要梯度计算，梯度不匹配
  std::vector<IValue> stack({a, b});  // 创建包含 a 和 b 的 IValue 向量 stack
  interp.run(stack);  // 运行 interp 对象处理 stack 中的数据
  ASSERT_FALSE(stack[2].toBool());  // 断言 stack 中第三个元素为 false
}

TEST_F(TypeCheckTest, ScalarTypeMismatch) {
  auto a = at::zeros({2, 2}, at::kFloat);  // 创建大小为 (2, 2) 的 Float 类型张量 a
  auto b = at::ones({3, 3}, at::kFloat);   // 创建大小为 (3, 3) 的 Float 类型张量 b
  a = a.to(at::kCPU);  // 将张量 a 移动到 CPU 上
  a.set_requires_grad(true);  // 设置张量 a 需要梯度计算
  a = a.to(at::kInt);  // 将张量 a 的标量类型改为 Int，类型不匹配
  std::vector<IValue> stack({a, b});  // 创建包含 a 和 b 的 IValue 向量 stack
  interp.run(stack);  // 运行 interp 对象处理 stack 中的数据
  ASSERT_FALSE(stack[2].toBool());  // 断言 stack 中第三个元素为 false
}

TEST_F(TypeCheckTest, DeviceMismatch_CUDA) {
  auto a = at::zeros({2, 2}, at::kFloat);  // 创建大小为 (2, 2) 的 Float 类型张量 a
  auto b = at::ones({3, 3}, at::kFloat);   // 创建大小为 (3, 3) 的 Float 类型张量 b
  a.set_requires_grad(true);  // 设置张量 a 需要梯度计算
  a = a.to(at::kCUDA);  // 将张量 a 移动到 CUDA 上，设备不匹配
  std::vector<IValue> stack({a, b});  // 创建包含 a 和 b 的 IValue 向量 stack
  interp.run(stack);  // 运行 interp 对象处理 stack 中的数据
  ASSERT_FALSE(stack[2].toBool());  // 断言 stack 中第三个元素为 false
}

// TODO: These tests weren't doing anything.
// TEST(TypeCheckErrorTest, EmptyCheckRaises) {
//   // Test empty Typecheck raises an internal assertion
//   auto graph = std::make_shared<Graph>();
// 测试用例：InterpreterTest, Basic_CUDA
TEST(InterpreterTest, Basic_CUDA) {
  // 定义常量
  constexpr int batch_size = 4;
  constexpr int input_size = 256;
  constexpr int seq_len = 32;

  // 计算隐藏层大小
  int hidden_size = 2 * input_size;

  // 创建 CUDA 上的随机输入张量
  auto input = at::randn({seq_len, batch_size, input_size}, at::kCUDA);
  // 创建 CUDA 上的随机隐藏状态张量
  auto hx = at::randn({batch_size, hidden_size}, at::kCUDA);
  // 创建 CUDA 上的随机细胞状态张量
  auto cx = at::randn({batch_size, hidden_size}, at::kCUDA);
  // 创建 CUDA 上的随机输入权重张量
  auto w_ih = t_def(at::randn({4 * hidden_size, input_size}, at::kCUDA));
  // 创建 CUDA 上的随机隐藏状态权重张量
  auto w_hh = t_def(at::randn({4 * hidden_size, hidden_size}, at::kCUDA));

  // 构建 LSTM 图形
  auto lstm_g = build_lstm();
  // 用 LSTM 图形构建代码对象
  Code lstm_function(lstm_g, "");
  // 创建 LSTM 解释器状态
  InterpreterState lstm_interp(lstm_function);
  // 运行 LSTM 解释器，获取输出
  auto outputs = run(lstm_interp, {input[0], hx, cx, w_ih, w_hh});
  // 运行 LSTM 模型，更新隐藏状态和细胞状态
  std::tie(hx, cx) = lstm(input[0], hx, cx, w_ih, w_hh);

  // 断言输出的隐藏状态与 LSTM 运行结果一致
  ASSERT_TRUE(exactlyEqual(outputs[0], hx));
  // 断言输出的细胞状态与 LSTM 运行结果一致
  ASSERT_TRUE(exactlyEqual(outputs[1], cx));
}

// 测试用例：InterpreterTest, IgnorableArgsInSchema
TEST(InterpreterTest, IgnorableArgsInSchema) {
  // 构建移动端导出分析图
  auto graph = build_mobile_export_analysis_graph();
  // 创建移动端代码对象
  MobileCode function(graph, "");
  // 获取操作到指定参数数目的映射
  auto op_to_specified_args = function.op_to_num_specified_args();
  // 断言操作到指定参数数目的映射大小为2
  ASSERT_TRUE(op_to_specified_args.size() == 2);
  // 断言 slice.Tensor 操作的指定参数数目为4
  ASSERT_TRUE(op_to_specified_args["aten::slice.Tensor"] == 4);
  // 断言 slice.str 操作的指定参数数目为4

  ASSERT_TRUE(op_to_specified_args["aten::slice.str"] == 4);

  // 构建包含可变参数的移动端导出分析图
  auto graph_vararg = build_mobile_export_analysis_graph_with_vararg();
  // 创建移动端代码对象（包含可变参数）
  MobileCode function_vararg(graph_vararg, "");
  // 获取操作到指定参数数目的映射（包含可变参数）
  auto op_to_specified_args_vararg = function_vararg.op_to_num_specified_args();
  // 断言 prim::tolist 操作未注册
  ASSERT_TRUE(
      op_to_specified_args_vararg.find("prim::tolist") ==
      op_to_specified_args_vararg.end());

  // 构建嵌套的移动端导出分析图
  auto graph_nested = build_mobile_export_analysis_graph_nested();
  // 创建移动端代码对象（嵌套版本）
  MobileCode function_nested(graph_nested, "");
  // 获取操作到指定参数数目的映射（嵌套版本）
  auto op_to_specified_args_nested = function_nested.op_to_num_specified_args();
  // 断言 slice.Tensor 操作的指定参数数目为4
  ASSERT_TRUE(op_to_specified_args_nested["aten::slice.Tensor"] == 4);
  // 断言 slice.str 操作的指定参数数目为4
  ASSERT_TRUE(op_to_specified_args_nested["aten::slice.str"] == 4);

  // 构建非常量的移动端导出分析图
  auto graph_non_const = build_mobile_export_analysis_graph_non_const();
  // 创建移动端代码对象（非常量版本）
  MobileCode function_non_const(graph_non_const, "");
  // 获取操作到指定参数数目的映射（非常量版本）
  auto op_to_specified_args_non_const =
      function_non_const.op_to_num_specified_args();
  // 断言 conv2d 操作的指定参数数目为6
  ASSERT_TRUE(op_to_specified_args_non_const["aten::conv2d"] == 6);
}
TEST(InterpreterTest, IgnorableArgsInSchemaWithOut) {
  // 构建一个移动端导出图形
  auto graph = build_mobile_export_with_out();
  // 创建一个移动端代码对象
  MobileCode function(graph, "");
  // 获取操作到指定参数数量的映射
  auto op_to_specified_args = function.op_to_num_specified_args();
  // 断言操作到指定参数数量的映射大小为1
  ASSERT_TRUE(op_to_specified_args.size() == 1);
  // 当 add_out 标志设置为 True 时，此处应为3
  ASSERT_TRUE(op_to_specified_args["aten::add.out"] == 3);
}

TEST(InterpreterTest, runAsyncBasicTest) {
  /*
  TODO: 在涉及 fork 的 C++ 解析脚本程序中存在一些问题。
  目前使用以下测试模块替代。
  相关问题请查看：github.com/pytorch/pytorch/issues/46368
  测试模块文件生成如下：
    class DemoModule(torch.nn.Module):
      def forward(self):
        r1 = torch.jit.fork(torch.mm, torch.rand(100,100),torch.rand(100,100))
        r2 = torch.jit.fork(torch.mm, torch.rand(100,100),torch.rand(100,100))
        return r1.wait() + r2.wait()
  demo = DemoModule()
  torch.jit.save(torch.jit.script(demo), 'test_interpreter_async.pt')
  */
  // 获取当前文件路径
  std::string filePath(__FILE__);
  // 构造测试模型文件路径
  auto testModelFile = filePath.substr(0, filePath.find_last_of("/\\") + 1);
  testModelFile.append("test_interpreter_async.pt");
  // 加载模型
  auto model = load(testModelFile);
  // 获取模型前向方法对应的图形
  auto graph = model.get_method("forward").graph();
  // 创建代码对象
  Code function(graph, "");
  // 异步计数器
  auto asyncCounter = 0;
  // 互斥锁
  std::mutex mtx;
  // 虚拟的执行器，实际使用 at::launch，但会增加计数器
  auto launcher = [&](std::function<void()> f) {
    mtx.lock();
    ++asyncCounter;
    mtx.unlock();
    at::launch(f);
  };
  // 栈
  std::vector<IValue> stack;
  // 将模型转换为 IValue 并压入栈
  stack.push_back(model._ivalue());
  // 创建解释器状态对象
  InterpreterState interp(function, launcher);
  // 运行异步任务并等待完成
  interp.runAsync(stack)->wait();
  // 断言异步计数器大于0
  ASSERT_TRUE(asyncCounter > 0);
}

TEST(
    EnableRethrowCaughtExceptionTest,
    EnableRethrowCaughtExceptionTestRethrowsCaughtException) {
  // 创建一个图形对象
  auto graph = std::make_shared<Graph>();
  // 值到值指针的无序映射
  std::unordered_map<std::string, Value*> vmap;
  // 解析 IR 到图形对象中
  parseIR(
      R"IR(
graph(%0 : Tensor,
      %1 : Tensor):
  %2 : int = prim::Constant[value=2]()
  %3 : Tensor = aten::add(%0, %1, %2)
  return (%3)
  )IR",
      &*graph,
      vmap);
  // 创建代码对象
  Code function(graph, "");
  // 创建解释器状态对象
  InterpreterState interp = InterpreterState(function);
  // 创建张量 a 和 b
  auto a = at::zeros({2, 2}, at::kFloat);
  auto b = at::ones({2, 3}, at::kFloat);
  // 设置张量 a 需要梯度，并移到 CPU
  a.set_requires_grad(true);
  a = a.to(at::kCPU);
  // 创建栈并压入张量 a 和 b
  std::vector<IValue> stack({a, b});

  // 保存原始标志值
  bool original_flag_value = FLAGS_torch_jit_enable_rethrow_caught_exception;
  // 异常处理标志
  bool exception_handled = false;
  try {
    // 禁用重新抛出捕获的异常标志
    FLAGS_torch_jit_enable_rethrow_caught_exception = false;
    // 运行解释器栈
    interp.run(stack);
  } catch (std::runtime_error& e) {
    // 标记已处理异常
    exception_handled = true;
    // 获取异常消息
    std::string exception_msg = e.what();
    // 断言异常消息包含特定子字符串
    EXPECT_THAT(
        exception_msg,
        ::testing::HasSubstr("%3 : Tensor = aten::add(%0, %1, %2)"));
  EXPECT_THAT(
      exception_msg,
      ::testing::HasSubstr(
          "The size of tensor a (2) must match the size of tensor b (3) at non-singleton dimension 1"));

检查异常消息 `exception_msg` 是否包含特定子字符串，用于断言测试。


  }
  EXPECT_TRUE(exception_handled);

断言确保异常已被处理。


  exception_handled = false;
  try {
    FLAGS_torch_jit_enable_rethrow_caught_exception = true;
    interp.run(stack);
  } catch (c10::Error& e) {
    exception_handled = true;
    std::string exception_msg = e.what_without_backtrace();
    EXPECT_STREQ(
        exception_msg.c_str(),
        "The size of tensor a (2) must match the size of tensor b (3) at non-singleton dimension 1");
  }
  EXPECT_TRUE(exception_handled);

尝试运行 `interp.run(stack)` 并捕获可能的 `c10::Error` 异常。如果捕获到异常，将 `exception_handled` 设为 `true`，并检查异常消息是否符合预期。


  FLAGS_torch_jit_enable_rethrow_caught_exception = true;
  c10::intrusive_ptr<Future> future = interp.runAsync(stack);
  future->wait();
  ASSERT_TRUE(future->completed());
  ASSERT_TRUE(future->hasError());
  try {
    std::rethrow_exception(future->exception_ptr());
  } catch (c10::Error& e) {
    std::string exception_msg = e.what_without_backtrace();
    EXPECT_STREQ(
        exception_msg.c_str(),
        "The size of tensor a (2) must match the size of tensor b (3) at non-singleton dimension 1");
  }

启用异常重新抛出，并运行异步操作 `interp.runAsync(stack)`。等待异步操作完成，并断言其已完成且有错误。如果捕获到异常，则再次检查异常消息是否符合预期。


  FLAGS_torch_jit_enable_rethrow_caught_exception = original_flag_value;

恢复 `FLAGS_torch_jit_enable_rethrow_caught_exception` 到其原始值，以保持状态一致性。
}

} // namespace jit
} // namespace torch
```