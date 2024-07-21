# `.\pytorch\test\cpp\jit\test_utils.cpp`

```py
#include <gtest/gtest.h>  // 包含 Google Test 框架的头文件

#include <test/cpp/jit/test_utils.h>  // 包含测试工具函数的头文件
#include <torch/csrc/jit/jit_log.h>  // 包含 JIT 编译日志的头文件
#include <torch/csrc/jit/passes/clear_undefinedness.h>  // 包含清除未定义变量的头文件
#include <torch/csrc/jit/runtime/custom_operator.h>  // 包含自定义运算符的头文件

namespace torch {
namespace jit {

Stack createStack(std::vector<at::Tensor>&& list) {
  // 创建一个堆栈对象，使用移动迭代器初始化
  return Stack(
      std::make_move_iterator(list.begin()),
      std::make_move_iterator(list.end()));
}

void assertAllClose(const tensor_list& a, const tensor_list& b) {
  // 断言两个张量列表的大小相等
  ASSERT_EQ(a.size(), b.size());
  // 逐一比较对应位置的张量是否具有相同的大小和近似值
  for (size_t i = 0; i < a.size(); ++i) {
    ASSERT_TRUE(a[i].is_same_size(b[i]));
    ASSERT_TRUE(a[i].allclose(b[i]));
  }
}

std::vector<at::Tensor> run(
    InterpreterState& interp,
    const std::vector<at::Tensor>& inputs) {
  // 将输入张量列表转换为 IValue 类型的堆栈
  std::vector<IValue> stack(inputs.begin(), inputs.end());
  // 运行解释器状态 interp 的堆栈
  interp.run(stack);
  // 将堆栈中的每个 IValue 转换回张量，并返回张量列表
  return fmap(stack, [](const IValue& i) { return i.toTensor(); });
}

static void unpackReturnTuple(Stack& stack) {
  // 弹出堆栈顶部的元组并展开到堆栈中
  auto tuple = pop(stack).toTuple();
  stack.insert(stack.end(), tuple->elements().begin(), tuple->elements().end());
}

std::pair<tensor_list, tensor_list> runGradient(
    Gradient& grad_spec,
    tensor_list& tensors_in,
    tensor_list& tensor_grads_in) {
  // 定义 lambda 函数，将堆栈中的 IValue 转换为张量列表
  static const auto as_tensorlist = [](const Stack& stack) {
    return fmap(stack, [](const IValue& i) { return i.toTensor(); });
  };
  // 清除梯度规范中的未定义行为
  ClearUndefinedness(grad_spec.df);
  // 创建包含函数和梯度函数的代码对象
  Code f_code{grad_spec.f, ""}, df_code{grad_spec.df, ""};
  // 创建函数解释器和梯度函数解释器的状态
  InterpreterState f_interpreter{f_code}, df_interpreter{df_code};

  // 将输入张量列表转换为 IValue 的堆栈并运行函数解释器
  auto f_stack = fmap<IValue>(tensors_in);
  f_interpreter.run(f_stack);

  // 创建梯度函数堆栈，并将梯度输入张量添加到堆栈中
  Stack df_stack;
  df_stack.insert(
      df_stack.end(), tensor_grads_in.begin(), tensor_grads_in.end());
  // 根据梯度规范中捕获的输入索引，将相应的张量添加到堆栈中
  for (auto offset : grad_spec.df_input_captured_inputs)
    df_stack.push_back(tensors_in[offset]);
  // 根据梯度规范中捕获的输出索引，将函数堆栈中的值添加到堆栈中
  for (auto offset : grad_spec.df_input_captured_outputs)
    df_stack.push_back(f_stack[offset]);
  // 运行梯度函数解释器的堆栈
  df_interpreter.run(df_stack);
  // 展开返回的元组到堆栈中
  unpackReturnTuple(df_stack);
  // 删除函数堆栈中超出真实输出的部分
  f_stack.erase(f_stack.begin() + grad_spec.f_real_outputs, f_stack.end());
  // 返回运行结果的张量列表对
  return std::make_pair(as_tensorlist(f_stack), as_tensorlist(df_stack));
}

std::shared_ptr<Graph> build_lstm() {
  // 定义 LSTM 图的字符串表示
  const auto graph_string = R"IR(
    # 定义一个函数graph，接受5个Tensor作为输入参数
    graph(%0 : Tensor,
          %1 : Tensor,
          %2 : Tensor,
          %3 : Tensor,
          %4 : Tensor):
      # 执行矩阵乘法，计算 %0 和 %3 的乘积，结果存储在 %5 中
      %5 : Tensor = aten::mm(%0, %3)
      # 执行矩阵乘法，计算 %1 和 %4 的乘积，结果存储在 %6 中
      %6 : Tensor = aten::mm(%1, %4)
      # 创建一个整数常量 %7，其值为1
      %7 : int = prim::Constant[value=1]()
      # 执行张量加法，将 %5 和 %6 相加，加法结果存储在 %8 中
      %8 : Tensor = aten::add(%5, %6, %7)
      # 将 %8 张量在维度1上分割成4个块，结果存储在 %9, %10, %11, %12 中
      %9 : Tensor, %10 : Tensor, %11 : Tensor, %12 : Tensor = prim::ConstantChunk[chunks=4, dim=1](%8)
      # 对 %9 张量执行sigmoid操作，结果存储在 %13 中
      %13 : Tensor = aten::sigmoid(%9)
      # 对 %12 张量执行sigmoid操作，结果存储在 %14 中
      %14 : Tensor = aten::sigmoid(%12)
      # 对 %11 张量执行tanh操作，结果存储在 %15 中
      %15 : Tensor = aten::tanh(%11)
      # 对 %10 张量执行sigmoid操作，结果存储在 %16 中
      %16 : Tensor = aten::sigmoid(%10)
      # 执行张量乘法，计算 %16 和 %2 的乘积，结果存储在 %17 中
      %17 : Tensor = aten::mul(%16, %2)
      # 执行张量乘法，计算 %13 和 %15 的乘积，结果存储在 %18 中
      %18 : Tensor = aten::mul(%13, %15)
      # 创建一个整数常量 %19，其值为1
      %19 : int = prim::Constant[value=1]()
      # 执行张量加法，将 %17 和 %18 相加，加法结果存储在 %20 中
      %20 : Tensor = aten::add(%17, %18, %19)
      # 对 %20 张量执行tanh操作，结果存储在 %21 中
      %21 : Tensor = aten::tanh(%20)
      # 执行张量乘法，计算 %14 和 %21 的乘积，结果存储在 %22 中
      %22 : Tensor = aten::mul(%14, %21)
      # 返回两个张量 %22 和 %20 作为结果
      return (%22, %20))IR";
    # 创建一个空的图对象 g
    auto g = std::make_shared<Graph>();
    # 使用图形字符串 graph_string 解析并填充图对象 g
    torch::jit::parseIR(graph_string, g.get());
    # 对图 g 进行静态检查
    g->lint();
    # 返回填充并检查过的图对象 g
    return g;
std::shared_ptr<Graph> build_mobile_export_analysis_graph() {
  // 创建一个字符串，包含用于构建图形的IR表示
  const auto graph_string = R"IR(
    graph(%0 : Tensor):
      %1 : int = prim::Constant[value=1]()
      %2 : int = prim::Constant[value=2]()
      %20 : int = prim::Constant[value=0]()
      %21 : int = prim::Constant[value=9223372036854775807]()
      %22 : str = prim::Constant[value="value"]()
      %3 : Tensor  = aten::slice(%0, %1, %20, %2, %1)
      %4 : Tensor = aten::slice(%0, %2, %20, %21, %1)
      %5 : str = aten::slice(%22, %20, %21, %2)
      return (%3, %4, %5))IR";
  
  // 创建一个新的图形对象
  auto g = std::make_shared<Graph>();
  // 解析图形字符串并将其加载到新创建的图形对象中
  torch::jit::parseIR(graph_string, g.get());
  // 对图形进行Lint检查，确保其结构的正确性
  g->lint();
  // 返回创建的图形对象
  return g;
}

std::shared_ptr<Graph> build_mobile_export_with_out() {
  // 创建一个字符串，包含用于构建图形的IR表示
  const auto graph_string = R"IR(
    graph(%x.1 : Tensor,
          %y.1 : Tensor):
      %8 : NoneType = prim::Constant()
      %6 : int = prim::Constant[value=1]()
      %7 : Tensor = aten::add(%x.1, %y.1, %6, %y.1)
      return (%8))IR";
  
  // 创建一个新的图形对象
  auto g = std::make_shared<Graph>();
  // 解析图形字符串并将其加载到新创建的图形对象中
  torch::jit::parseIR(graph_string, g.get());
  // 对图形进行Lint检查，确保其结构的正确性
  g->lint();
  // 返回创建的图形对象
  return g;
}

std::shared_ptr<Graph> build_mobile_export_analysis_graph_nested() {
  // 创建一个字符串，包含用于构建图形的IR表示
  const auto graph_string = R"IR(
    graph(%0 : Tensor):
      %1 : int = prim::Constant[value=1]()
      %2 : int = prim::Constant[value=2]()
      %20 : int = prim::Constant[value=0]()
      %21 : int = prim::Constant[value=9223372036854775807]()
      %22 : str = prim::Constant[value="value"]()
      %3 : Tensor  = aten::slice(%0, %1, %20, %2, %1)
      %23 : bool = aten::Bool(%3)
      %c : Tensor = prim::If(%23)
        block0():
          %4 : Tensor = aten::slice(%0, %2, %20, %21, %1)
          %5 : str = aten::slice(%22, %20, %21, %2)
          %c.1 : Tensor = aten::slice(%0, %1, %20, %2, %1)
          -> (%c.1)
        block1():
          -> (%3)
      return (%3, %3))IR";
  
  // 创建一个新的图形对象
  auto g = std::make_shared<Graph>();
  // 解析图形字符串并将其加载到新创建的图形对象中
  torch::jit::parseIR(graph_string, g.get());
  // 对图形进行Lint检查，确保其结构的正确性
  g->lint();
  // 返回创建的图形对象
  return g;
}

std::shared_ptr<Graph> build_mobile_export_analysis_graph_with_vararg() {
  // 此函数未提供完整的IR表示字符串，因此不需要进一步注释
  const auto graph_string = R"IR(
    def graph(%0 : Tensor):
      %1 : int = prim::Constant[value=1]()  # 创建一个常量张量，值为1
      %2 : int = prim::Constant[value=2]()  # 创建一个常量张量，值为2
      %3 : int = prim::Constant[value=3]()  # 创建一个常量张量，值为3
      %4 : int[] = prim::tolist(%1, %2)     # 将张量 %1 和 %2 转换为整数数组 %4
      %5 : int[] = prim::tolist(%1, %2, %3)  # 将张量 %1, %2 和 %3 转换为整数数组 %5
      return (%4, %5)  # 返回数组 %4 和 %5
    
    
    
    auto g = std::make_shared<Graph>();  // 创建一个指向 Graph 对象的 shared_ptr g
    torch::jit::parseIR(graph_string, g.get());  // 解析给定的 IR 字符串到 Graph 对象 g
    g->lint();  // 对 Graph 对象 g 进行 lint 检查（语法检查）
    return g;  // 返回创建和检查后的 Graph 对象 g
}

// 构建一个移动导出分析图的非常量版本
std::shared_ptr<Graph> build_mobile_export_analysis_graph_non_const() {
  // 定义包含图形描述的字符串，使用R"IR( ... )IR"语法
  const auto graph_string = R"IR(
      graph(%input.1 : Tensor):
        %7 : int = prim::Constant[value=1]() # <string>:3:58
        %9 : int = prim::Constant[value=0]() # <string>:3:66
        %8 : int[] = prim::ListConstruct(%7, %7)
        %10 : int[] = prim::ListConstruct(%9, %9)
        %11 : int[] = prim::ListConstruct(%7, %7)
        %12 : Tensor = aten::conv2d(%input.1, %input.1, %input.1, %8, %10, %11, %7)
        return (%12))IR";
  
  // 创建一个新的图形对象
  auto g = std::make_shared<Graph>();
  // 解析图形描述字符串到图形对象中
  torch::jit::parseIR(graph_string, g.get());
  // 对图形进行静态分析
  g->lint();
  // 返回构建好的图形对象
  return g;
}

// 函数 t_use：返回输入张量 x
at::Tensor t_use(at::Tensor x) {
  return x;
}

// 函数 t_def：返回输入张量 x 的转置
at::Tensor t_def(at::Tensor x) {
  return x.t();
}

// 函数 checkRtol：检查两个张量之间的相对容差是否符合指定的阈值
bool checkRtol(const at::Tensor& diff, const std::vector<at::Tensor> inputs) {
  // 初始化最大值为 0
  double maxValue = 0.0;
  // 遍历输入张量列表，更新最大值
  for (auto& tensor : inputs) {
    maxValue = fmax(tensor.abs().max().item<float>(), maxValue);
  }
  // 检查差异张量相对值是否小于阈值的最大值乘以指定常数
  return diff.abs().max().item<float>() < 2e-6 * maxValue;
}

// 函数 almostEqual：检查两个张量是否几乎相等
bool almostEqual(const at::Tensor& a, const at::Tensor& b) {
  // 调用 checkRtol 函数检查相对容差
  return checkRtol(a - b, {a, b});
}

// 函数 exactlyEqual：检查两个张量是否完全相等
bool exactlyEqual(const at::Tensor& a, const at::Tensor& b) {
  // 检查两个张量之间的绝对差异是否为 0
  return (a - b).abs().max().item<float>() == 0.f;
}

// 函数 exactlyEqual：检查两个张量向量是否完全相等
bool exactlyEqual(
    const std::vector<at::Tensor>& a,
    const std::vector<at::Tensor>& b) {
  // 首先检查向量大小是否相同
  if (a.size() != b.size()) {
    return false;
  }
  // 遍历每个张量，调用 exactlyEqual 函数检查是否完全相等
  for (size_t i = 0; i < a.size(); ++i) {
    if (!exactlyEqual(a[i], b[i])) {
      return false;
    }
  }
  // 如果所有张量都相等，则返回 true
  return true;
}

// 函数 lstm：模拟 LSTM 神经网络操作
std::pair<at::Tensor, at::Tensor> lstm(
    at::Tensor input,
    at::Tensor hx,
    at::Tensor cx,
    at::Tensor w_ih,
    at::Tensor w_hh) {
  // 计算输入张量和权重的线性组合
  auto gates = input.mm(t_use(w_ih)) + hx.mm(t_use(w_hh));

  // 将线性组合的结果分割成不同门的结果
  auto chunked_gates = gates.chunk(4, 1);
  auto ingate = chunked_gates[0];
  auto forgetgate = chunked_gates[1];
  auto cellgate = chunked_gates[2];
  auto outgate = chunked_gates[3];

  // 应用激活函数
  ingate = ingate.sigmoid();
  outgate = outgate.sigmoid();
  cellgate = cellgate.tanh();
  forgetgate = forgetgate.sigmoid();

  // 计算新的细胞状态和隐藏状态
  auto cy = (forgetgate * cx) + (ingate * cellgate);
  auto hy = outgate * cy.tanh();

  // 返回隐藏状态和细胞状态
  return {hy, cy};
}

// 内联函数 aliasAnalysisFromSchema：从架构中获取别名分析类型
inline c10::AliasAnalysisKind aliasAnalysisFromSchema() {
  return c10::AliasAnalysisKind::FROM_SCHEMA;
}

// 匿名命名空间：注册运算符
namespace {
RegisterOperators reg({
    // 这个运算符用于 JIT 分析和转换的单元测试，生成空张量
    // 不应在执行图形时使用，因为它总是生成空张量
    Operator(
        "prim::MakeTestTensor() -> Tensor",
        [](Stack& stack) { push(stack, at::Tensor()); },
        aliasAnalysisFromSchema()),
});
} // namespace

// 函数 runGraph：运行给定的图形对象
std::vector<at::Tensor> runGraph(
    std::shared_ptr<Graph> graph,
    // 定义函数，接收一个输入的张量向量作为参数
    const std::vector<at::Tensor>& inputs) {
  // 将输入的张量向量转换为对应的 IValue 类型向量
  std::vector<IValue> stack = fmap<IValue>(inputs);
  // 使用给定的计算图和名称创建 Code 对象
  Code code(graph, "test");
  // 创建解释器状态，并运行解释器，传入堆栈数据
  InterpreterState(code).run(stack);
  // 断言堆栈不为空
  TORCH_INTERNAL_ASSERT(!stack.empty());
  // 如果堆栈顶部的元素是张量列表
  // 返回该列表作为结果
  // 期望的输出类型包括：
  //   * 一个张量列表。
  //   * 一个张量。
  if (stack.front().isTensorList()) {
    return stack.front().toTensorVector();
  }
  // 否则，断言堆栈顶部的元素是一个张量
  TORCH_INTERNAL_ASSERT(stack.front().isTensor());
  // 返回仅包含单个张量的向量作为结果
  return {stack.front().toTensor()};
}
}

// 结束 jit 命名空间
} // namespace jit
// 结束 torch 命名空间
} // namespace torch
```