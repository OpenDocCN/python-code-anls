# `.\pytorch\test\cpp\tensorexpr\test_dynamic_shapes.cpp`

```py
// 包含 Google Test 的头文件，用于测试框架
#include <gtest/gtest.h>

// 包含 ATen 的代码模板
#include <ATen/code_template.h>
// 包含 C10 核心库中的设备类型定义
#include <c10/core/DeviceType.h>
// 包含 TensorExpr 测试基础的头文件
#include <test/cpp/tensorexpr/test_base.h>
// 包含 Torch 的 IR（Intermediate Representation）定义
#include <torch/csrc/jit/ir/ir.h>
// 包含 Torch 的 IR 解析器
#include <torch/csrc/jit/ir/irparser.h>
// 包含 Torch 的符号形状运行时融合相关的 passes
#include <torch/csrc/jit/passes/symbolic_shape_runtime_fusion.h>
// 包含 Torch 的 TensorExpr 核心头文件
#include <torch/csrc/jit/tensorexpr/kernel.h>
// 包含 Torch 的文件检查工具头文件
#include <torch/csrc/jit/testing/file_check.h>
// 包含 Torch 的核心库
#include <torch/torch.h>

// 包含数学函数库
#include <cmath>
// 包含字符串流处理库
#include <sstream>
// 包含异常处理库
#include <stdexcept>
// 包含多线程支持库
#include <thread>

// Torch 命名空间下的 JIT 命名空间
namespace torch {
namespace jit {

// 使用 Torch 的索引命名空间
using namespace torch::indexing;
// 使用 Torch 的 TensorExpr 命名空间
using namespace torch::jit::tensorexpr;

// 定义 DynamicShapes 测试类，继承自 gtest 的测试类
TEST(DynamicShapes, SimpleGraph) {
#ifdef TORCH_ENABLE_LLVM
  // 创建一个共享的图对象
  std::shared_ptr<Graph> graph = std::make_shared<Graph>();
  // 定义包含 IR 的字符串
  const auto graph_string = R"IR(
      graph(%x : Tensor,
            %SS_2 : int,
            %SS_3 : int):
        %3 : Tensor = aten::tanh(%x)
        %4 : Tensor = aten::erf(%3)
        return (%4))IR";
  // 解析 IR 字符串并加载到图对象中
  torch::jit::parseIR(graph_string, graph.get());

  // 获取图的输入节点并设置符号形状类型
  auto x_inp = graph->inputs()[0];
  auto x_type = TensorType::create(at::rand({10, 5}));
  std::vector<ShapeSymbol> x_sym_dims(
      {c10::ShapeSymbol::newSymbol(), c10::ShapeSymbol::newSymbol()});
  auto x_sym_type = x_type->withSymbolicShapes(x_sym_dims);
  graph->inputs().at(0)->setType(x_sym_type);
  for (const auto n : graph->nodes()) {
    n->output()->setType(x_sym_type);
  }

  // 定义图的符号形状描述
  std::vector<torch::jit::StrideInput> input_desc = {
      torch::jit::StrideInput::TENSOR_CONT};
  std::unordered_map<
      const torch::jit::Value*,
      std::vector<torch::jit::StrideInput>>
      symbolic_strides;
  symbolic_strides[x_inp] = input_desc;
  symbolic_strides[graph->outputs().at(0)] = input_desc;
  // 获取符号形状的输入参数
  std::vector<int64_t> symbolic_shape_inputs = c10::fmap(
      x_sym_dims,
      [](const c10::ShapeSymbol& shapeSym) { return shapeSym.value(); });

  // 创建 TensorExprKernel 对象用于运行图
  TensorExprKernel kernel(
      graph, {}, symbolic_shape_inputs, false, symbolic_strides);

  // 使用相同静态维度运行图
  {
    auto a = at::rand({10, 5}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto ref = at::erf(at::tanh(a));

    // 创建输入值堆栈并运行内核
    std::vector<IValue> stack = fmap<IValue>(std::vector<at::Tensor>({a}));
    stack.push_back(10);
    stack.push_back(5);
    kernel.run(stack);

    // 获取输出并使用断言验证结果
    auto o = stack[0].toTensor();
    ASSERT_TRUE(at::allclose(o, ref));
  }

  // 使用不同维度的输入运行图
  {
    auto a = at::rand({50, 100}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto ref = at::erf(at::tanh(a));

    // 创建输入值堆栈并运行内核
    std::vector<IValue> stack = fmap<IValue>(std::vector<at::Tensor>({a}));
    stack.push_back(50);
    stack.push_back(100);
    kernel.run(stack);

    // 获取输出并使用断言验证结果
    auto o = stack[0].toTensor();
    ASSERT_TRUE(at::allclose(o, ref));
  }
#endif
}

// JIT 命名空间结束
} // namespace jit
// Torch 命名空间结束
} // namespace torch
#ifdef TORCH_ENABLE_LLVM
  // 如果 TORCH_ENABLE_LLVM 宏已定义，则编译以下代码块

  // 创建一个共享指针指向新建的图对象
  std::shared_ptr<Graph> graph = std::make_shared<Graph>();

  // 定义包含图结构的字符串
  const auto graph_string = R"IR(
      graph(%x : Tensor,
            %y : Tensor,
            %SS_2 : int,
            %SS_3 : int):
        %3 : Tensor = aten::tanh(%x)
        %4 : Tensor = aten::erf(%3)
        %5 : Tensor = aten::mul(%4, %y)
        return (%5))IR";

  // 解析图结构字符串并填充到图对象中
  torch::jit::parseIR(graph_string, graph.get());

  // 获取输入节点 x 和 y
  auto x_inp = graph->inputs()[0];
  auto y_inp = graph->inputs()[1];

  // 创建一个随机形状的张量类型作为 x 的类型
  auto x_type = TensorType::create(at::rand({10, 5}));

  // 创建一个包含两个形状符号的符号形状向量，作为 x 的符号形状类型
  std::vector<ShapeSymbol> x_sym_dims(
      {c10::ShapeSymbol::newSymbol(), c10::ShapeSymbol::newSymbol()});
  auto x_sym_type = x_type->withSymbolicShapes(x_sym_dims);

  // 设置图的输入节点 x 和 y 的类型为符号形状类型
  graph->inputs().at(0)->setType(x_sym_type);
  graph->inputs().at(1)->setType(x_sym_type);

  // 设置图中所有节点的输出类型为符号形状类型
  for (const auto n : graph->nodes()) {
    n->output()->setType(x_sym_type);
  }

  // 创建一个描述符向量，包含符号形状输入的值
  std::vector<int64_t> symbolic_shape_inputs = c10::fmap(
      x_sym_dims,
      [](const c10::ShapeSymbol& shapeSym) { return shapeSym.value(); });

  // 创建一个输入描述符向量，描述张量的布局
  std::vector<torch::jit::StrideInput> input_desc = {
      torch::jit::StrideInput::TENSOR_CONT};

  // 创建一个无序映射，将张量值映射到符号步幅描述符向量
  std::unordered_map<
      const torch::jit::Value*,
      std::vector<torch::jit::StrideInput>>
      symbolic_strides;

  // 将输入节点 x 和 y 以及图的输出节点映射到输入描述符向量
  symbolic_strides[x_inp] = input_desc;
  symbolic_strides[y_inp] = input_desc;
  symbolic_strides[graph->outputs().at(0)] = input_desc;

  // 创建一个张量表达式内核对象，用于运行包含符号形状的图
  TensorExprKernel kernel(
      graph, {}, symbolic_shape_inputs, false, symbolic_strides);

  // 使用与初始化图时相同的静态形状维度运行
  {
    // 创建两个大小相同的随机张量 a 和 b
    auto a = at::rand({10, 5}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto b = at::rand({10, 5}, at::TensorOptions(at::kCPU).dtype(at::kFloat));

    // 计算参考结果，对应于 erf(tanh(a)) * b
    auto ref = at::mul(at::erf(at::tanh(a)), b);

    // 创建包含 a、b 和静态形状维度的值向量
    std::vector<IValue> stack = fmap<IValue>(std::vector<at::Tensor>({a, b}));
    stack.push_back(10);  // 添加静态形状维度 10
    stack.push_back(5);   // 添加静态形状维度 5

    // 运行张量表达式内核
    kernel.run(stack);

    // 获取输出张量 o
    auto o = stack[0].toTensor();

    // 断言 o 和参考结果 ref 接近
    ASSERT_TRUE(at::allclose(o, ref));
  }

  // 使用不同形状维度的输入运行
  {
    // 创建两个大小不同的随机张量 a 和 b
    auto a = at::rand({50, 100}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto b = at::rand({50, 100}, at::TensorOptions(at::kCPU).dtype(at::kFloat));

    // 计算参考结果，对应于 erf(tanh(a)) * b
    auto ref = at::mul(at::erf(at::tanh(a)), b);

    // 创建包含 a、b 和不同形状维度的值向量
    std::vector<IValue> stack = fmap<IValue>(std::vector<at::Tensor>({a, b}));
    stack.push_back(50);  // 添加不同的形状维度 50
    stack.push_back(100); // 添加不同的形状维度 100

    // 运行张量表达式内核
    kernel.run(stack);

    // 获取输出张量 o
    auto o = stack[0].toTensor();

    // 断言 o 和参考结果 ref 接近
    ASSERT_TRUE(at::allclose(o, ref));
  }
#endif
TEST(DynamicShapes, GraphWith2InputsAndBroadcast) {
#ifdef TORCH_ENABLE_LLVM
  // 当图中的第二个输入具有尺寸为1的维度时，应在 at::mul 操作中进行广播。
  // 创建一个空的图对象，用于存储解析后的图结构
  std::shared_ptr<Graph> graph = std::make_shared<Graph>();
  // 定义包含图结构的字符串表示，使用 R"IR("...")IR" 语法
  const auto graph_string = R"IR(
      graph(%x : Float(10, 5, requires_grad=0, device=cpu),
            %y : Float(1, 5, requires_grad=0, device=cpu),
            %SS_2 : int,
            %SS_3 : int):
        %3 : Tensor = aten::tanh(%x)
        %4 : Tensor = aten::erf(%3)
        %5 : Tensor = aten::mul(%4, %y)
        return (%5))IR";
  // 解析字符串并填充到图对象中
  torch::jit::parseIR(graph_string, graph.get());

  // 获取图的输入节点
  auto x_inp = graph->inputs()[0];
  auto y_inp = graph->inputs()[1];
  // 创建具有静态形状的输入张量类型
  auto x_type = TensorType::create(at::rand({10, 5}));
  auto y_type = TensorType::create(at::rand({1, 5}));
  // 创建新的符号形状表示
  auto x_dim0_sym = c10::ShapeSymbol::newSymbol();
  auto x_dim1_sym = c10::ShapeSymbol::newSymbol();
  auto x_sym_type = x_type->withSymbolicShapes(
      std::vector<ShapeSymbol>({x_dim0_sym, x_dim1_sym}));
  auto y_sym_type = y_type->withSymbolicShapes(std::vector<ShapeSymbol>(
      {c10::ShapeSymbol::fromStaticSize(1), x_dim1_sym}));
  // 设置图的输入节点的类型
  graph->inputs().at(0)->setType(x_sym_type);
  graph->inputs().at(1)->setType(y_sym_type);
  // 设置图中所有节点的输出类型
  for (const auto n : graph->nodes()) {
    n->output()->setType(x_sym_type);
  }

  // 图的符号形状表示：
  //
  // graph(%x : Float(SS(-6), SS(-7)),
  //       %y : Float(1, SS(-7)),
  //       %SS_2 : int,
  //       %SS_3 : int):
  //   %4 : Float(SS(-6), SS(-7)) = aten::tanh(%x)
  //   %5 : Float(SS(-6), SS(-7)) = aten::erf(%4)
  //   %6 : Float(SS(-6), SS(-7)) = aten::mul(%5, %y)
  //   return (%6)

  // 创建包含符号形状输入维度的向量
  std::vector<int64_t> symbolic_shape_inputs(
      {x_dim0_sym.value(), x_dim1_sym.value()});

  // 定义输入描述，这里是一个包含 TENSOR_CONT 的输入描述向量
  std::vector<torch::jit::StrideInput> input_desc = {
      torch::jit::StrideInput::TENSOR_CONT};
  // 创建一个映射，将图中的值与其对应的符号步长输入描述关联起来
  std::unordered_map<
      const torch::jit::Value*,
      std::vector<torch::jit::StrideInput>>
      symbolic_strides;
  symbolic_strides[x_inp] = input_desc;
  symbolic_strides[y_inp] = input_desc;
  symbolic_strides[graph->outputs().at(0)] = input_desc;

  // 创建一个 TensorExprKernel 对象，用于表示和处理这个图的张量表达式
  TensorExprKernel kernel(
      graph, {}, symbolic_shape_inputs, false, symbolic_strides);

  // 使用与初始化图相同的静态维度运行
  {
    auto a = at::rand({10, 5}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto b = at::rand({1, 5}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto ref = at::mul(at::erf(at::tanh(a)), b);

    std::vector<IValue> stack = fmap<IValue>(std::vector<at::Tensor>({a, b}));
    stack.push_back(10);
    stack.push_back(5);
    kernel.run(stack);

    auto o = stack[0].toTensor();
    ASSERT_TRUE(at::allclose(o, ref));
  }

  // 使用不同维度的输入运行
  {
    auto a = at::rand({50, 100}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto b = at::rand({1, 100}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto ref = at::mul(at::erf(at::tanh(a)), b);
    # 创建一个空的 std::vector<IValue> 对象 stack，并初始化为包含 a 和 b 两个 at::Tensor 的 vector
    std::vector<IValue> stack = fmap<IValue>(std::vector<at::Tensor>({a, b}));
    # 向 stack 中添加整数 50
    stack.push_back(50);
    # 向 stack 中添加整数 100
    stack.push_back(100);
    # 调用 kernel 的 run 方法，传入 stack 作为参数执行计算
    kernel.run(stack);
    
    # 从 stack 中取出索引为 0 的元素，并将其转换为 at::Tensor 类型，赋值给变量 o
    auto o = stack[0].toTensor();
    # 使用 ASSERT_TRUE 断言函数，验证 o 与 ref 的值在误差允许范围内相等
    ASSERT_TRUE(at::allclose(o, ref));
TEST(DynamicShapes, GraphWithPartiallySymbolicOutput) {
#ifdef TORCH_ENABLE_LLVM
  // 如果 LLVM 支持被启用，则执行以下测试

  // 创建一个共享指针指向一个新的图形对象
  std::shared_ptr<Graph> graph = std::make_shared<Graph>();
  // 定义一个包含 IR 字符串的常量
  const auto graph_string = R"IR(
      graph(%x : Float(1, 5, requires_grad=0, device=cpu),
            %y : Float(1, 5, requires_grad=0, device=cpu),
            %SS_2 : int):
        %4 : Tensor = aten::tanh(%x)
        %5 : Tensor = aten::mul(%4, %y)
        return (%5))IR";
  // 解析 IR 字符串并将其加载到图形对象中
  torch::jit::parseIR(graph_string, graph.get());

  // 获取输入节点
  auto x_inp = graph->inputs()[0];
  auto y_inp = graph->inputs()[1];
  // 创建一个新的静态张量类型
  auto x_type = TensorType::create(at::rand({1, 5}));
  // 创建一个新的形状符号
  auto x_dim1_sym = c10::ShapeSymbol::newSymbol();
  // 创建具有符号形状的张量类型
  auto x_sym_type = x_type->withSymbolicShapes(std::vector<ShapeSymbol>(
      {c10::ShapeSymbol::fromStaticSize(1), x_dim1_sym}));
  // 设置图形输入节点的类型
  graph->inputs().at(0)->setType(x_sym_type);
  graph->inputs().at(1)->setType(x_sym_type);
  // 为图中的每个节点设置输出类型
  for (const auto n : graph->nodes()) {
    n->output()->setType(x_sym_type);
  }

  // 创建具有符号形状的图形：
  //
  // graph(%x : Float(1, SS(-2)),
  //       %y : Float(1, SS(-2)),
  //       %SS_2 : int):
  //   %3 : Float(1, SS(-2)) = aten::tanh(%x)
  //   %4 : Float(1, SS(-2)) = aten::mul(%3, %y)
  //   return (%4)

  // 创建一个包含符号形状的输入向量
  std::vector<int64_t> symbolic_shape_inputs({x_dim1_sym.value()});

  // 创建一个输入描述的向量
  std::vector<torch::jit::StrideInput> input_desc = {
      torch::jit::StrideInput::TENSOR_CONT};
  // 创建一个映射，将值指针映射到符号步幅的向量
  std::unordered_map<
      const torch::jit::Value*,
      std::vector<torch::jit::StrideInput>>
      symbolic_strides;
  symbolic_strides[x_inp] = input_desc;
  symbolic_strides[y_inp] = input_desc;
  symbolic_strides[graph->outputs().at(0)] = input_desc;

  // 创建一个 TensorExprKernel 对象
  TensorExprKernel kernel(
      graph, {}, symbolic_shape_inputs, false, symbolic_strides);

  // 使用与初始化图形时相同的静态维度运行
  {
    auto a = at::rand({1, 5}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto b = at::rand({1, 5}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto ref = at::mul(at::tanh(a), b);

    // 创建一个存储输入值的堆栈
    std::vector<IValue> stack = fmap<IValue>(std::vector<at::Tensor>({a, b}));
    stack.push_back(5);
    // 运行 kernel
    kernel.run(stack);

    // 获取输出张量
    auto o = stack[0].toTensor();
    // 断言输出与参考值接近
    ASSERT_TRUE(at::allclose(o, ref));
  }

  // 使用不同维度的输入运行
  {
    auto a = at::rand({1, 100}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto b = at::rand({1, 100}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto ref = at::mul(at::tanh(a), b);

    // 创建一个存储输入值的堆栈
    std::vector<IValue> stack = fmap<IValue>(std::vector<at::Tensor>({a, b}));
    stack.push_back(100);
    // 运行 kernel
    kernel.run(stack);

    // 获取输出张量
    auto o = stack[0].toTensor();
    // 断言输出与参考值接近
    ASSERT_TRUE(at::allclose(o, ref));
  }
#endif
}
  // 定义一个名为 graph 的函数，接受四个参数，其中前两个为 Float 类型的张量，后两个为整数
  graph(%0 : Float(SS(-2), SS(-3), requires_grad=0, device=cpu),
        %1 : Float(SS(-2), SS(-3), requires_grad=0, device=cpu),
        %SS_3 : int,
        %SS_2 : int):
    // 创建一个整数常量值为 1
    %15 : int = prim::Constant[value=1]()
    // 计算 %0 和 %1 张量的加法，结果存储在 %21 中
    %21 : Float(SS(-2), SS(-3), requires_grad=0, device=cpu) = aten::add(%0, %1, %15)
    // 计算 %21 和 %0 张量的乘法，结果存储在 %22 中
    %22 : Float(SS(-2), SS(-3), requires_grad=0, device=cpu) = aten::mul(%21, %0)
    // 返回 %22 作为函数结果
    return (%22))IR";

  // 解析图形字符串并存储到 graph 中
  parseIR(graph_string, &*graph);

  // 定义输入描述为张量步长的容器
  std::vector<torch::jit::StrideInput> input_desc = {
      torch::jit::StrideInput::S_AS_ARG, torch::jit::StrideInput::S_ONE};
  // 定义输出描述为连续张量的容器
  std::vector<torch::jit::StrideInput> output_desc = {
      torch::jit::StrideInput::TENSOR_CONT};
  // 创建一个无序映射，将图形输入张量与其步长描述关联起来
  std::unordered_map<
      const torch::jit::Value*,
      std::vector<torch::jit::StrideInput>>
      symbolic_strides;
  symbolic_strides[graph->inputs().at(0)] = input_desc;
  symbolic_strides[graph->inputs().at(1)] = input_desc;
  symbolic_strides[graph->outputs().at(0)] = output_desc;

  // 定义符号形状输入的容器，包含两个负数值
  std::vector<int64_t> symbolic_shape_inputs = {-3, -2};
  // 创建一个张量表达式内核对象
  TensorExprKernel k(graph, {}, symbolic_shape_inputs, false, symbolic_strides);

  // 第一个作用域，运行张量表达式内核 k，使用随机生成的 x0 和 x1 张量进行计算
  {
    auto x0 = at::rand({10, 32}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto x1 = at::rand({10, 32}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto ref = at::mul(at::add(x0, x1, 1), x0);

    // 创建输入张量的向量和 IValue 堆栈
    std::vector<at::Tensor> inputs = {x0, x1};
    std::vector<IValue> stack = at::fmap<at::IValue>(inputs);
    stack.push_back(32);  // 将整数值 32 推入堆栈
    stack.push_back(10);  // 将整数值 10 推入堆栈
    k.run(stack);  // 运行内核 k，并传递堆栈作为输入参数

    auto o = stack[0].toTensor();  // 从堆栈中获取输出张量 o
    ASSERT_TRUE(at::allclose(o, ref));  // 断言输出张量 o 与预期结果 ref 相近
  }

  // 第二个作用域，运行张量表达式内核 k，使用随机生成的 x0 和 x1 张量及预分配的 out 张量进行计算
  {
    auto x0 = at::rand({10, 32}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto x1 = at::rand({10, 32}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto out = at::rand({10, 32}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto ref = at::mul(at::add(x0, x1, 1), x0);

    // 创建输入张量的向量和 IValue 堆栈
    std::vector<at::Tensor> inputs = {out, x0, x1};
    std::vector<IValue> stack = at::fmap<at::IValue>(inputs);
    stack.push_back(32);  // 将整数值 32 推入堆栈
    stack.push_back(10);  // 将整数值 10 推入堆栈
    k.runWithAllocatedOutputs(stack);  // 运行内核 k，并传递堆栈作为输入参数

    ASSERT_TRUE(at::allclose(out, ref));  // 断言输出张量 out 与预期结果 ref 相近
  }
#ifdef TORCH_ENABLE_LLVM
  // 创建一个空的图形对象，使用指定的数据类型
  std::shared_ptr<Graph> graph = std::make_shared<Graph>();
  // 定义表示 IR 字符串的常量
  const auto graph_string = R"IR(
    graph(%0 : Float(SS(-2), SS(-3), requires_grad=0, device=cpu),
          %1 : Float(SS(-2), SS(-4), requires_grad=0, device=cpu),
          %2 : Float(SS(-2), SS(-5), requires_grad=0, device=cpu),
          %input.4 : Long(SS(-2), SS(-6), requires_grad=0, device=cpu),
          %4 : Float(SS(-7), requires_grad=0, device=cpu),
          %5 : Float(SS(-7), requires_grad=0, device=cpu),
          %SS_10 : int,
          %SS_9 : int,
          %SS_8 : int,
          %SS_7 : int,
          %SS_6 : int,
          %SS_5 : int,
          %SS_4 : int,
          %SS_3 : int,
          %SS_2 : int):
      %15 : int = prim::Constant[value=1]()
      %16 : bool = prim::Constant[value=0]()
      %17 : int = prim::Constant[value=6]()
      %18 : Float(SS(-2), SS(-6), strides=[139, 1], requires_grad=0, device=cpu) = aten::to(%input.4, %17, %16, %16)
      %19 : Tensor[] = prim::ListConstruct(%0, %1, %18, %2)
      %20 : Float(SS(-2), SS(-8), strides=[261, 1], requires_grad=0, device=cpu) = aten::cat(%19, %15)
      %21 : Float(SS(-2), SS(-9), strides=[261, 1], requires_grad=0, device=cpu) = aten::add(%20, %5, %15)
      %22 : Float(SS(-2), SS(-10), requires_grad=0, device=cpu) = aten::mul(%21, %4)
      return (%22))IR";
  // 解析 IR 字符串并填充图形对象
  parseIR(graph_string, &*graph);

  // 创建表示张量步幅的描述列表
  std::vector<torch::jit::StrideInput> input_desc = {
      torch::jit::StrideInput::TENSOR_CONT};
  // 创建表示符号步幅的映射
  std::unordered_map<
      const torch::jit::Value*,
      std::vector<torch::jit::StrideInput>>
      symbolic_strides;
  // 将输入张量与其步幅描述关联
  symbolic_strides[graph->inputs().at(0)] = input_desc;
  symbolic_strides[graph->inputs().at(1)] = input_desc;
  symbolic_strides[graph->inputs().at(2)] = input_desc;
  symbolic_strides[graph->inputs().at(3)] = input_desc;
  symbolic_strides[graph->inputs().at(4)] = input_desc;
  symbolic_strides[graph->inputs().at(5)] = input_desc;
  // 将输出张量与其步幅描述关联
  symbolic_strides[graph->outputs().at(0)] = input_desc;

  // 创建表示符号形状输入的列表
  std::vector<int64_t> symbolic_shape_inputs = {
      -10, -9, -8, -7, -6, -5, -4, -3, -2};
  // 创建张量表达式内核对象
  TensorExprKernel k(graph, {}, symbolic_shape_inputs, false, symbolic_strides);

  // 定义多个整数变量并初始化
  int64_t i2 = 10;
  int64_t i3 = 32;
  int64_t i4 = 19;
  int64_t i5 = 71;
  int64_t i6 = 139;
  int64_t i7 = 261;
  int64_t i8 = 261;
  int64_t i9 = 261;
  int64_t i10 = 261;
  // 使用指定形状和选项创建随机张量
  auto x0 = at::rand({i2, i3}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
  auto x1 = at::rand({i2, i4}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
  auto x2 = at::rand({i2, i5}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
  auto x3 = at::ones({i2, i6}, at::TensorOptions(at::kCPU).dtype(at::kLong));
  auto x4 = at::rand({i7}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
  auto x5 = at::rand({i8}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
  // 计算参考结果张量
  auto ref = at::mul(at::add(at::cat({x0, x1, x3, x2}, 1), x5), x4);

  {
#endif
}
    // 创建一个包含输入张量的向量，顺序为 x0, x1, x2, x3, x4, x5
    std::vector<at::Tensor> inputs = {x0, x1, x2, x3, x4, x5;
    
    // 将输入张量转换为 IValue 类型的向量 stack
    std::vector<IValue> stack = at::fmap<at::IValue>(inputs);

    // 将 i10 添加到 stack 中
    stack.emplace_back(i10);
    // 将 i9 添加到 stack 中
    stack.emplace_back(i9);
    // 将 i8 添加到 stack 中
    stack.emplace_back(i8);
    // 将 i7 添加到 stack 中
    stack.emplace_back(i7);
    // 将 i6 添加到 stack 中
    stack.emplace_back(i6);
    // 将 i5 添加到 stack 中
    stack.emplace_back(i5);
    // 将 i4 添加到 stack 中
    stack.emplace_back(i4);
    // 将 i3 添加到 stack 中
    stack.emplace_back(i3);
    // 将 i2 添加到 stack 中
    stack.emplace_back(i2);
    
    // 运行模型，传入 stack 中的数据
    k.run(stack);

    // 从 stack 中取出第一个元素，并将其转换为 Tensor 类型，存入 o
    auto o = stack[0].toTensor();
    // 使用断言确保 o 与参考张量 ref 具有相同的值
    ASSERT_TRUE(at::allclose(o, ref));
  }

  {
    // 创建一个形状为 [i2, i10] 的随机张量 out，数据类型为 float，存储于 CPU
    auto out = at::rand({i2, i10}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    
    // 创建一个包含 out 和输入张量的向量 inputs，顺序为 out, x0, x1, x2, x3, x4, x5
    std::vector<at::Tensor> inputs = {out, x0, x1, x2, x3, x4, x5};
    
    // 将输入张量转换为 IValue 类型的向量 stack
    std::vector<IValue> stack = at::fmap<at::IValue>(inputs);

    // 将 i10 添加到 stack 中
    stack.emplace_back(i10);
    // 将 i9 添加到 stack 中
    stack.emplace_back(i9);
    // 将 i8 添加到 stack 中
    stack.emplace_back(i8);
    // 将 i7 添加到 stack 中
    stack.emplace_back(i7);
    // 将 i6 添加到 stack 中
    stack.emplace_back(i6);
    // 将 i5 添加到 stack 中
    stack.emplace_back(i5);
    // 将 i4 添加到 stack 中
    stack.emplace_back(i4);
    // 将 i3 添加到 stack 中
    stack.emplace_back(i3);
    // 将 i2 添加到 stack 中
    stack.emplace_back(i2);
    
    // 运行模型，传入 stack 中的数据，其中 out 作为输出张量被分配
    k.runWithAllocatedOutputs(stack);

    // 使用断言确保 out 与参考张量 ref 具有相同的值
    ASSERT_TRUE(at::allclose(out, ref));
  }
TEST(DynamicShapes, MultiThreadedExecution) {
#ifdef TORCH_ENABLE_LLVM
  // 定义一个模板化的 IR 图形字符串，包含输入和操作，其中 ${device} 被替换为 "cpu" 或 "cuda:0"
  const auto graph_template = R"IR(
      graph(%x : Float(SS(-2), SS(-3), requires_grad=0, device=${device}),
            %y : Float(SS(-2), SS(-3), requires_grad=0, device=${device}),
            %SS_2 : int,
            %SS_3 : int):
        %3 : Float(SS(-2), SS(-3), requires_grad=0, device=${device}) = aten::tanh(%x)
        %4 : Float(SS(-2), SS(-3), requires_grad=0, device=${device}) = aten::erf(%3)
        %5 : Float(SS(-2), SS(-3), requires_grad=0, device=${device}) = aten::mul(%4, %y)
        return (%5))IR";
  
  // 针对使用 CUDA 和 CPU 的情况，循环运行测试
  for (bool use_cuda : {false, true}) {
    // 如果 CUDA 不可用且需要使用 CUDA，则跳过此次循环
    if (!torch::cuda::is_available() && use_cuda) {
      continue;
    }
    
    // 确定当前使用的设备（CUDA 或 CPU）
    auto device = use_cuda ? at::kCUDA : at::kCPU;
    // 创建一个模板环境对象
    at::jit::TemplateEnv env;
    // 根据当前设备设置环境变量
    env.s("device", use_cuda ? "cuda:0" : "cpu");
    // 根据模板和环境生成完整的 IR 图形字符串
    const auto graph_string = format(graph_template, env);
    // 创建一个共享指针指向图形对象
    std::shared_ptr<Graph> graph = std::make_shared<Graph>();
    // 解析 IR 字符串，填充图形对象
    torch::jit::parseIR(graph_string, graph.get());

    // 定义符号形状输入的向量
    std::vector<int64_t> symbolic_shape_inputs = {-2, -3};

    // 定义输入描述的向量
    std::vector<torch::jit::StrideInput> input_desc = {
        torch::jit::StrideInput::TENSOR_CONT};
    
    // 创建映射，将图形对象的输入和输出与输入描述关联
    std::unordered_map<
        const torch::jit::Value*,
        std::vector<torch::jit::StrideInput>>
        symbolic_strides;
    symbolic_strides[graph->inputs().at(0)] = input_desc;
    symbolic_strides[graph->inputs().at(1)] = input_desc;
    symbolic_strides[graph->outputs().at(0)] = input_desc;

    // 创建 TensorExprKernel 对象，用于运行优化后的图形计算
    TensorExprKernel kernel(
        graph, {}, symbolic_shape_inputs, false, symbolic_strides);

    // 定义并运行一个 lambda 函数，用于测试多线程执行的一致性
    auto run_kernel = [&](int dim1, int dim2) {
      // 创建随机张量 a 和 b，形状为 dim1 x dim2，数据类型为 float，设备为指定的 device
      auto a =
          at::rand({dim1, dim2}, at::TensorOptions(device).dtype(at::kFloat));
      auto b =
          at::rand({dim1, dim2}, at::TensorOptions(device).dtype(at::kFloat));

      // 计算参考结果 ref，等效于 at::mul(at::erf(at::tanh(a)), b)
      auto ref = at::mul(at::erf(at::tanh(a)), b);

      // 构建堆栈并调用 kernel 的 run 方法，执行优化后的计算图
      std::vector<IValue> stack = fmap<IValue>(std::vector<at::Tensor>({a, b}));
      stack.emplace_back(dim1);
      stack.emplace_back(dim2);
      kernel.run(stack);

      // 获取运行后的张量结果 o
      auto o = stack[0].toTensor();
      // 断言运行结果 o 与参考结果 ref 接近
      ASSERT_TRUE(at::allclose(o, ref));
    };

    // 以并行方式运行内核，以确保 TensorExprKernel 中的 run() 方法不会更改任何状态
    constexpr size_t kNumThreads = 4;
    std::vector<std::thread> threads;
    for (size_t id = 0; id < kNumThreads; ++id) {
      threads.emplace_back(run_kernel, id + 5, id + 20);
    }
    // 等待所有线程运行完成
    for (auto& t : threads) {
      t.join();
    }
  }
#endif
}
```