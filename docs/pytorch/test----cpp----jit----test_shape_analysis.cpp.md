# `.\pytorch\test\cpp\jit\test_shape_analysis.cpp`

```py
// 包含 Google Test 框架的头文件
#include <gtest/gtest.h>

// 包含 ATen 库的头文件
#include <ATen/ATen.h>

// 包含 ATen 核心的内部化字符串的头文件
#include <ATen/core/interned_strings.h>

// 包含 C10 库中的异常处理头文件
#include <c10/util/Exception.h>

// 包含 C10 库中的可选类型头文件
#include <c10/util/Optional.h>

// 包含 Torch 的 JIT 测试工具的头文件
#include <test/cpp/jit/test_utils.h>

// 包含 Torch 的 JIT 中间表示(IR)的头文件
#include <torch/csrc/jit/ir/ir.h>

// 包含 Torch 的 JIT IR 视图的头文件
#include <torch/csrc/jit/ir/ir_views.h>

// 包含 Torch 的 JIT IR 解析器的头文件
#include <torch/csrc/jit/ir/irparser.h>

// 包含 Torch 的 JIT 常量传播 passes 的头文件
#include <torch/csrc/jit/passes/constant_propagation.h>

// 包含 Torch 的 JIT 符号形状分析 passes 的头文件
#include <torch/csrc/jit/passes/symbolic_shape_analysis.h>

// 包含 Torch 的 JIT 符号形状缓存 passes 的头文件
#include <torch/csrc/jit/passes/symbolic_shape_cache.h>

// 包含 Torch 的 JIT 符号形状运行时融合 passes 的头文件
#include <torch/csrc/jit/passes/symbolic_shape_runtime_fusion.h>

// 包含 Torch 的 JIT 子图工具的头文件
#include <torch/csrc/jit/passes/utils/subgraph_utils.h>

// 包含 Torch 的 JIT 图遍历器的头文件
#include <torch/csrc/jit/runtime/graph_iterator.h>

// 包含 Torch 的 JIT 解释器的头文件
#include <torch/csrc/jit/runtime/interpreter.h>

// 包含 Torch 的 JIT 测试文件检查头文件
#include <torch/csrc/jit/testing/file_check.h>

// 包含 Torch 的 CUDA 头文件
#include <torch/cuda.h>

// 包含标准库的无序映射头文件
#include <unordered_map>

// Torch 的 JIT 命名空间开始
namespace torch {
namespace jit {

// Torch 的 JIT 匿名命名空间开始

// 在给定图中查找特定符号节点的函数定义
Node* findNode(std::shared_ptr<Graph>& g, Symbol k) {
  // 创建深度优先图节点迭代器
  DepthFirstGraphNodeIterator graph_it(g);
  // 遍历图中的每个节点
  for (auto node = graph_it.next(); node != nullptr; node = graph_it.next()) {
    // 如果节点的类型与指定的符号匹配，则返回该节点
    if (node->kind() == k) {
      return node;
    }
  }
  // 如果未找到符号对应的节点，则抛出内部断言错误
  TORCH_INTERNAL_ASSERT(false, "Couldn't find node");
}

// Torch 的 JIT 匿名命名空间结束

} // namespace jit
} // namespace torch
TEST(ShapeAnalysisTest, DynamicShapesFusion) {
  // Test Generalizing shapes to symbolic dimensions, guarding those symbolic
  // dimensions and passing in runtime computed symbolic dimensions via inlined
  // shape functions

  // 创建一个新的图形对象，用于测试
  std::shared_ptr<Graph> subgraph = std::make_shared<Graph>();

  // 定义一个包含图形的字符串，表示一个简单的计算图
  const auto graph_string = R"IR(
      graph(%x.1 : Tensor, %y.1 : Tensor, %z: Tensor):
        %11 : int = prim::Constant[value=0]()
        %3 : Tensor = aten::tanh(%x.1)
        %out1.1 : Tensor = aten::erf(%3)
        %out2.1 : Tensor = aten::relu(%y.1)
        %10 : Tensor[] = prim::ListConstruct(%out1.1, %out2.1)
        %25 : Tensor = aten::cat(%10, %11)
        %28 : Tensor = aten::hardswish(%25)
        %29 : Tensor = aten::mul(%28, %z)
        return (%28))IR";
  
  // 解析上述字符串表示的计算图，将其添加到子图中
  torch::jit::parseIR(graph_string, subgraph.get());

  /*
  set up fused TensorExprGroup
  */

  // 创建另一个新的图形对象
  std::shared_ptr<Graph> g = std::make_shared<Graph>();

  // 添加输入节点到图形对象中，并设置其类型
  auto x_inp = g->addInput("x_inp");
  auto y_inp = g->addInput("y_inp");
  auto z_inp = g->addInput("z_inp");
  auto x_type = TensorType::create(at::rand({10, 5}));
  auto y_type = TensorType::create(at::rand({4, 5}));
  auto z_type = TensorType::create(at::rand({1, 1}));
  x_inp->setType(x_type);
  y_inp->setType(y_type);
  z_inp->setType(z_type);

  // 将子图的输入类型与图形对象中对应的输入类型关联起来
  subgraph->inputs().at(0)->setType(x_type);
  subgraph->inputs().at(1)->setType(y_type);
  subgraph->inputs().at(2)->setType(z_type);

  // 设置子图的输出类型
  subgraph->outputs().at(0)->setType(TensorType::create(at::rand({14, 5})));

  // 在图形对象中创建一个 TensorExprGroup 节点，并获取其输出
  auto output = g->insertNode(g->create(prim::TensorExprGroup))->output();

  // 将子图与输出节点关联
  output->node()->addInput(x_inp);
  output->node()->addInput(y_inp);
  output->node()->addInput(z_inp);
  output->node()->g_(attr::Subgraph, subgraph);

  // 生成并验证符号维度保护
  auto success = GenerateGuard(output->node());
  TORCH_INTERNAL_ASSERT(success);

  // 使用 FileCheck 进行进一步的验证
  testing::FileCheck()
      .check("TensorExprDynamicGuard")
      ->check_next("prim::If")
      ->check("aten::add")
      ->check("TensorExprGroup")
      ->check_same("symbolic_shape_inputs")
      ->check("block1")
      ->check("aten::cat")
      ->run(*g);

  // clang-format off
  /* Graph Should Look Something like: (note: strides not yet handled)
  graph(%x_inp : Float(10, 5, strides=[5, 1], requires_grad=0, device=cpu),
      %y_inp : Float(4, 5, strides=[5, 1], requires_grad=0, device=cpu),
      %z_inp : Float(1, 1, strides=[1, 1], requires_grad=0, device=cpu)):
  %4 : bool = prim::TensorExprDynamicGuard[types=[Float(SS(-2), SS(-3), strides=[5, 1], requires_grad=0, device=cpu), Float(SS(-4), SS(-3), strides=[5, 1], requires_grad=0, device=cpu), Float(1, 1, strides=[1, 1], requires_grad=0, device=cpu)]](%x_inp, %y_inp, %z_inp)
  %5 : Tensor = prim::If(%4)

  */

  // clang-format on
}
    block0():
      # 获取输入张量 x_inp 的尺寸
      %15 : int[] = aten::size(%x_inp)
      # 获取输入张量 y_inp 的尺寸
      %16 : int[] = aten::size(%y_inp)
      # 创建整数常量 1
      %17 : int = prim::Constant[value=1]()
      # 创建整数常量 0
      %18 : int = prim::Constant[value=0]()
      # 从 x_inp 的尺寸数组中取出索引为 0 的元素
      %elem.3 : int = aten::__getitem__(%15, %18) # <string>:40:10
      # 从 x_inp 的尺寸数组中取出索引为 1 的元素
      %elem.5 : int = aten::__getitem__(%15, %17) # <string>:40:10
      # 从 y_inp 的尺寸数组中取出索引为 0 的元素
      %elem.11 : int = aten::__getitem__(%16, %18) # <string>:40:10
      # 计算沿着指定维度（通常是通道维度）的拼接后的大小
      %cat_dim_size.48 : int = aten::add(%elem.3, %elem.11) # <string>:321:29
      # 使用 TensorExprGroup_0 执行张量表达式组的计算
      %3 : Tensor = prim::TensorExprGroup_0[symbolic_shape_inputs=[-5, -4, -3, -2]](%x_inp, %y_inp, %z_inp, %cat_dim_size.48, %elem.11, %elem.5, %elem.3)
      -> (%3)
    
    block1():
      // FallbackGraph is inlined
      # 使用 FallbackGraph_1 备用图来计算
      %14 : Tensor = prim::FallbackGraph_1(%x_inp, %y_inp, %z_inp)
      -> (%14)
  
  return ()
  # 返回空值

  with prim::TensorExprGroup_0 = graph(%x.1 : Float(SS(-2), SS(-3), strides=[5, 1], requires_grad=0, device=cpu),
        %y.1 : Float(SS(-4), SS(-3), strides=[5, 1], requires_grad=0, device=cpu),
        %z : Float(1, 1, strides=[1, 1], requires_grad=0, device=cpu),
        %SS_5 : int,
        %SS_4 : int,
        %SS_3 : int,
        %SS_2 : int):
    # 创建整数常量 0
    %3 : int = prim::Constant[value=0]()
    # 对 x.1 张量进行双曲正切操作
    %4 : Tensor(SS(-2), SS(-3)) = aten::tanh(%x.1)
    # 对上一步结果张量执行误差函数操作
    %5 : Tensor(SS(-2), SS(-3)) = aten::erf(%4)
    # 对 y.1 张量执行修正线性单元操作
    %6 : Tensor(SS(-4), SS(-3)) = aten::relu(%y.1)
    # 创建张量列表，包含 %5 和 %6 两个张量
    %7 : Tensor[] = prim::ListConstruct(%5, %6)
    # 沿着第一个维度拼接张量列表中的张量
    %8 : Tensor(SS(-5), SS(-3)) = aten::cat(%7, %3)
    # 对输入张量执行 hard swish 激活函数操作
    %9 : Tensor(SS(-5), SS(-3)) = aten::hardswish(%8)
    # 将 %9 与标量张量 z 执行逐元素乘法
    %10 : Tensor(SS(-5), SS(-3)) = aten::mul(%9, %z)
    return (%9)
  */
  // clang-format on

  // 创建一个深度优先图节点迭代器，用于遍历图 g
  DepthFirstGraphNodeIterator graph_it(g);
  // 在图 g 中查找并获取名为 prim::TensorExprGroup 的节点
  Node* te_group = findNode(g, prim::TensorExprGroup);

  /*
  测试内核输入 (10, 5), (4, 5), (1, 1) 是否正确推广为符号维度，并且输出 (10 + 4, 5)
  是否正确保留非连接维作为符号形状，连接维作为新的符号形状
  */

  // 获取 tensor expression 图
  auto tensorexpr_graph = te_group->g(attr::Subgraph);
  // 获取第一个、第二个和第三个输入张量的类型，并期望其为 TensorType
  auto inp1 = tensorexpr_graph->inputs().at(0)->type()->expect<TensorType>();
  auto inp2 = tensorexpr_graph->inputs().at(1)->type()->expect<TensorType>();
  auto inp3 = tensorexpr_graph->inputs().at(2)->type()->expect<TensorType>();
  // 获取输出张量的类型，并期望其为 TensorType
  auto out = tensorexpr_graph->outputs().at(0)->type()->expect<TensorType>();

  // 确保维度为 1 的尺寸被保留
  auto inp3_sizes = inp3->sizes().concrete_sizes();
  TORCH_INTERNAL_ASSERT(inp3_sizes);
  TORCH_INTERNAL_ASSERT(
      inp3_sizes->size() == 2 && inp3_sizes->at(0) == 1 &&
      inp3_sizes->at(1) == 1);

  // 确保维度为 5 被转换为符号形状
  ASSERT_EQ(
      inp1->symbolic_sizes()[1].value(), inp2->symbolic_sizes()[1].value());
  ASSERT_EQ(
      out->symbolic_sizes()[1].value(), inp2->symbolic_sizes()[1].value());

  // 确保维度为 4, 10, 14 有不同的符号形状
  ASSERT_NE(
      inp1->symbolic_sizes()[0].value(), inp2->symbolic_sizes()[0].value());
  ASSERT_NE(
      out->symbolic_sizes()[0].value(), inp1->symbolic_sizes()[0].value());
  ASSERT_NE(
      out->symbolic_sizes()[0].value(), inp2->symbolic_sizes()[0].value());

  /*
    测试保护程序在运行时的正确行为和符号形状的计算是否正确。
    由于我们不支持动态形状的 TE 内核，我们将在保护成功时返回所有计算出的运行时符号维度作为图的输出，
    在保护失败时返回 None。
  */

  // 设置保护程序以在保护成功时返回符号形状，并在失败时返回 None
  Node* if_node = findNode(g, prim::If);
  IfView if_v(if_node);
  if_node->eraseOutput(0);
  if_v.thenBlock()->eraseOutput(0);
  if_v.elseBlock()->eraseOutput(0);
  WithInsertPoint guard(if_node);
  auto none_val = g->insertConstant(IValue());

  // 检索 te_group 的符号形状输入
  auto sym_shapes = te_group->is(Symbol::attr("symbolic_shape_inputs"));
  auto offset = te_group->inputs().size() - sym_shapes.size();
  for (size_t i = 0; i < sym_shapes.size(); ++i) {
    // 在 then 分支插入符号形状输入
    if_v.thenBlock()->insertOutput(i, te_group->inputs().at(offset + i));
    // 在 else 分支插入 None
    if_v.elseBlock()->insertOutput(i, none_val);
  if_node->insertOutput(i)->setType(OptionalType::create(IntType::get()));



  // 在 if 节点中的第 i 个输出位置插入一个可选类型的整数类型，并设置其类型
  if_node->insertOutput(i)->setType(OptionalType::create(IntType::get()));



  }



  // 结束 for 循环
  }



  auto new_outputs = g->createTuple(if_node->outputs())->insertAfter(if_node);



  // 在图 g 中创建一个元组，其中包含 if 节点的输出，并在 if 节点之后插入这个新创建的元组
  auto new_outputs = g->createTuple(if_node->outputs())->insertAfter(if_node);



  g->registerOutput(new_outputs->output());



  // 在图 g 中注册新输出，这个输出是 new_outputs 的输出
  g->registerOutput(new_outputs->output());



  te_group->destroy();



  // 销毁 te_group 对象
  te_group->destroy();



  findNode(g, prim::FallbackGraph)->destroy();



  // 查找图 g 中的名为 prim::FallbackGraph 的节点，并销毁它
  findNode(g, prim::FallbackGraph)->destroy();



  // Testing bad inputs



  // 测试不良输入



  auto first_inp = at::rand({2, 5});



  // 创建一个形状为 [2, 5] 的随机张量 first_inp
  auto first_inp = at::rand({2, 5});



  std::vector<std::vector<at::Tensor>> second_inps = {



  // 创建一个嵌套的张量向量，second_inps，包含多组不同的测试输入
  std::vector<std::vector<at::Tensor>> second_inps = {



      {at::rand({3, 4}), at::rand({1, 1})}, // sym shape mismatch



      // 第一组测试输入：形状不匹配
      {at::rand({3, 4}), at::rand({1, 1})},



      {at::rand({5, 2}).transpose(0, 1), at::rand({1, 1})}, // discontiguous



      // 第二组测试输入：不连续张量
      {at::rand({5, 2}).transpose(0, 1), at::rand({1, 1})},



      {at::zeros({2, 5}).to(at::ScalarType::Int),
       at::rand({1, 1})}, // wrong dtype



      // 第三组测试输入：错误的数据类型
      {at::zeros({2, 5}).to(at::ScalarType::Int), at::rand({1, 1})},



      {at::rand({2, 5, 1}), at::rand({1, 1})}, // wrong # dims



      // 第四组测试输入：维度数量错误
      {at::rand({2, 5, 1}), at::rand({1, 1})},



      {at::rand({2, 5}).requires_grad_(true),
       at::rand({1, 1})}, // requires grad



      // 第五组测试输入：梯度跟踪标志设置为 true
      {at::rand({2, 5}).requires_grad_(true), at::rand({1, 1})},



      {at::rand({2, 5}), at::rand({1, 12})}, // concrete dim mismatch (1)



      // 第六组测试输入：具体维度不匹配
      {at::rand({2, 5}), at::rand({1, 12})},



  };



  // 结束 second_inps 的初始化
  };



  if (torch::cuda::is_available()) {
    second_inps.push_back({at::rand({2, 5}).cuda(), at::rand({1, 1})});
  }



  // 如果 CUDA 可用，则添加一组 CUDA 张量到 second_inps 中
  if (torch::cuda::is_available()) {
    second_inps.push_back({at::rand({2, 5}).cuda(), at::rand({1, 1})});
  }



  for (const auto& last_inps : second_inps) {



  // 对 second_inps 中的每一组测试输入执行以下循环
  for (const auto& last_inps : second_inps) {



    // todo - reusing interpreter across iters gave error



    // todo - 跨迭代重用解释器导致错误



    Code code(g, "");



    // 使用图 g 创建代码对象 code
    Code code(g, "");



    InterpreterState interp(code);



    // 使用 code 创建解释器状态 interp
    InterpreterState interp(code);



    auto stack = createStack({at::rand({2, 5}), last_inps[0], last_inps[1]});



    // 创建一个堆栈 stack，包含 first_inp、last_inps[0] 和 last_inps[1]
    auto stack = createStack({at::rand({2, 5}), last_inps[0], last_inps[1]});



    interp.run(stack);



    // 运行解释器 interp，传入堆栈 stack
    interp.run(stack);



    TORCH_INTERNAL_ASSERT(pop(stack).toTuple()->elements().at(0).isNone());



    // 断言：从堆栈中弹出的第一个元素是空值
    TORCH_INTERNAL_ASSERT(pop(stack).toTuple()->elements().at(0).isNone());



  }



  // 结束对 second_inps 的测试循环
  }



  // Test good inputs



  // 测试良好输入



  Code code(g, "");



  // 使用图 g 创建代码对象 code
  Code code(g, "");



  InterpreterState interp(code);



  // 使用 code 创建解释器状态 interp
  InterpreterState interp(code);



  std::vector<at::Tensor> inps = {
      at::rand({2, 5}), at::rand({4, 5}), at::rand({1, 1})};



  // 创建一个张量向量 inps，包含三个不同的张量
  std::vector<at::Tensor> inps = {
      at::rand({2, 5}), at::rand({4, 5}), at::rand({1, 1})};



  Stack stack(inps.begin(), inps.end());



  // 创建一个堆栈 stack，使用向量 inps 的所有元素
  Stack stack(inps.begin(), inps.end());



  interp.run(stack);



  // 运行解释器 interp，传入堆栈 stack
  interp.run(stack);



  auto tuple = pop(stack).toTuple();



  // 从堆栈中弹出一个元组，并将其赋给变量 tuple
  auto tuple = pop(stack).toTuple();



  TORCH_INTERNAL_ASSERT(tuple->elements().at(0).isInt());



  // 断言：元组的第一个元素是整数类型
  TORCH_INTERNAL_ASSERT(tuple->elements().at(0).isInt());



  // Testing that the sym shape calculation was correct



  // 测试符号形状计算是否正确



  for (size_t i = 0; i < sym_shapes.size(); ++i) {



  // 遍历 sym_shapes 的每一个元素
  for (size_t i = 0; i < sym_shapes.size(); ++i) {



    auto sym_shape = sym_shapes[i];



    // 将 sym_shapes 的第 i 个元素赋给变量 sym_shape
    auto sym_shape = sym_shapes[i];



    auto computed_value = tuple->elements
}

// 定义一个测试用例，测试将常量移出融合组
TEST(ShapeAnalysisTest, MovingConstantOutOfFusionGroups) {
  // 创建一个共享指针指向新的图对象
  std::shared_ptr<Graph> subgraph = std::make_shared<Graph>();
  // 定义包含图形结构的字符串
  const auto graph_string = R"IR(
      graph(%x.1 : Tensor):
        %none : NoneType = prim::Constant()
        %size1 : int = prim::Constant[value=1]()
        %size10 : int = prim::Constant[value=10]()
        %sizes : int[] = prim::ListConstruct(%size10, %size1)
        %device : Device = prim::Constant[value="cpu"]()
        %10 : Tensor = aten::ones(%sizes, %none, %none, %device, %none)
        %3 : Tensor = aten::tanh(%x.1)
        %29 : Tensor = aten::mul(%3, %10)
        return (%29))IR";
  // 解析图形字符串并将其添加到子图中
  torch::jit::parseIR(graph_string, subgraph.get());
  // 对子图进行常量传播优化
  ConstantPropagation(subgraph);

  // 创建一个新的图对象
  std::shared_ptr<Graph> g = std::make_shared<Graph>();
  // 添加一个输入节点并设置其类型
  auto x_inp = g->addInput("x_inp");
  auto x_type = TensorType::create(at::rand({10, 5}));
  x_inp->setType(x_type);
  // 设置子图的输入和输出类型
  subgraph->inputs().at(0)->setType(x_type);
  subgraph->outputs().at(0)->setType(x_type);
  // 插入一个节点并将子图作为其属性
  auto output = g->insertNode(g->create(prim::TensorExprGroup))->output();
  output->node()->addInput(x_inp);
  output->node()->g_(attr::Subgraph, subgraph);

  // 生成保护条件
  auto success = GenerateGuard(output->node());
  // 内部断言成功
  TORCH_INTERNAL_ASSERT(success);

  // 检查常量是否已从融合图中移出
  // 这应该导致除了检查TensorExprDynamicGuard结果之外不会有其他条件语句
  testing::FileCheck()
      .check("TensorExprDynamicGuard")
      ->check_next("prim::If")
      ->check_not("prim::If") // no other IFs due to constants.
      ->check("TensorExprGroup")
      ->check("block1")
      ->check("FallbackGraph")
      ->run(*g);
}

// 匿名命名空间开始

// 定义一个带有可选维度的符号形状的断言函数
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
void assertShapeEqual(c10::SymbolicShape& a, c10::SymbolicShape& e) {
  // 规范化符号形状a和e
  auto a_canonical = CanonicalizedSymbolicShape(a);
  auto e_canonical = CanonicalizedSymbolicShape(e);
  // 使用断言验证规范化后的形状是否相等
  EXPECT_EQ(a_canonical, e_canonical);
}

// 定义一个断言函数，比较实际和预期的符号形状向量
void assertShapeEqual(
    std::optional<std::vector<c10::SymbolicShape>>& actual,
    std::vector<std::optional<int64_t>> expected) {
  // 断言实际值已有值
  ASSERT_TRUE(actual.has_value());
  // 断言实际值向量大小为1
  ASSERT_EQ(actual->size(), 1);

  // 创建符号形状的预期值对象
  auto symb_expected = c10::SymbolicShape(expected);
  // 使用前面的断言函数验证实际值与预期值的符号形状是否相等
  assertShapeEqual(actual->at(0), symb_expected);
}

// 获取给定名称操作符的函数模式
const FunctionSchema* getSchema(const char* name) {
  // 返回名称对应操作符的函数模式
  return &(getOperatorForLiteral(name)->schema());
}
// 匿名命名空间结束
TEST(ShapeAnalysisTest, SymbolicShapeAPI) {
  // Figure out how to fetch a function schema
  // 确定如何获取函数模式(schema)

  // Ask someone else how to create a function schema / operator in C++
  // 询问其他人如何在 C++ 中创建函数模式/操作符
  auto schema = getSchema(
      "aten::sub.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor");

  c10::IValue const_size_1 = std::vector<int64_t>{64, 56, 56};
  c10::IValue const_size_2 = std::vector<int64_t>{1, 56, 56};

  // Check vector initializer list syntax
  // 检查向量初始化列表的语法
  c10::SymbolicShape ss_concrete =
      std::vector<std::optional<int64_t>>{1, 56, 56};
  c10::SymbolicShape ss1 = std::vector<std::optional<int64_t>>{sym_dim, 56, 56};
  c10::SymbolicShape ss2 =
      std::vector<std::optional<int64_t>>{64, sym_dim, sym_dim};
  c10::SymbolicShape ss3 =
      std::vector<std::optional<int64_t>>{sym_dim, sym_dim, sym_dim, sym_dim};

  auto res = calculateSymbolicShapesOnOp(
      schema, std::vector<SSAInput>{const_size_1, const_size_1});
  assertShapeEqual(res, {64, 56, 56});

  res = calculateSymbolicShapesOnOp(
      schema, std::vector<SSAInput>{const_size_1, const_size_2});
  assertShapeEqual(res, {64, 56, 56});

  res = calculateSymbolicShapesOnOp(
      schema, std::vector<SSAInput>{const_size_1, ss1});
  assertShapeEqual(res, {64, 56, 56});

  res = calculateSymbolicShapesOnOp(
      schema, std::vector<SSAInput>{const_size_2, ss1});
  assertShapeEqual(res, {sym_dim, 56, 56});

  res = calculateSymbolicShapesOnOp(
      schema, std::vector<SSAInput>{ss_concrete, ss2});
  assertShapeEqual(res, {64, 56, 56});

  res = calculateSymbolicShapesOnOp(schema, std::vector<SSAInput>{ss2, ss3});
  assertShapeEqual(res, {sym_dim, 64, sym_dim, sym_dim});
}

TEST(ShapeAnalysisTest, BoundedSymbolicShapes) {
  auto schema = getSchema("aten::nonzero(Tensor self) -> (Tensor)");

  // Test that we generate symbolic shapes for the output of a nonzero op
  // 测试我们是否为非零操作的输出生成了符号形状
  c10::IValue const_size_1 = std::vector<int64_t>{5, 10};
  auto res =
      calculateSymbolicShapesOnOp(schema, std::vector<SSAInput>{const_size_1});
  assertShapeEqual(res, {sym_dim, 2});

  // Test that nonzero can also create concrete shapes
  // 测试非零操作也可以创建具体形状
  c10::IValue const_size_2 = std::vector<int64_t>({1, 0});
  res =
      calculateSymbolicShapesOnOp(schema, std::vector<SSAInput>{const_size_2});
  assertShapeEqual(res, {0, 2});
}
// 定义测试用例 ShapeAnalysisTest 中的 SymbolicShapeCaching 函数
TEST(ShapeAnalysisTest, SymbolicShapeCaching) {
  // 清空形状缓存
  clear_shape_cache();
  // 获取操作的模式字符串表示
  auto schema = getSchema("aten::mm(Tensor self, Tensor mat2) -> Tensor");

  // 定义三个常量形状
  c10::IValue const_size_1 = std::vector<int64_t>{64, 56};
  c10::IValue const_size_2 = std::vector<int64_t>{64, 56};
  c10::IValue const_size_3 = std::vector<int64_t>{64, 20};

  // 创建三个符号形状对象，其中 sym_dim 是一个符号维度
  c10::SymbolicShape ss1 = c10::SymbolicShape({sym_dim, 64});
  c10::SymbolicShape ss2 = c10::SymbolicShape({sym_dim, 64});
  c10::SymbolicShape ss3 = c10::SymbolicShape({sym_dim, sym_dim});

  // 计算第一个操作的符号形状结果，并验证其形状
  auto res = calculateSymbolicShapesOnOp(schema, {ss1, const_size_1});
  assertShapeEqual(res, {sym_dim, 56});
  auto res1_val = res->at(0);

  // 使用相同的参数再次计算操作，验证结果应该与之前相同
  res = calculateSymbolicShapesOnOp(schema, {ss1, const_size_1});
  auto res2_val = res->at(0);
  EXPECT_EQ(res1_val, res2_val);
  EXPECT_EQ(get_shape_cache_size(), 1);

  // 使用相同形状但不同符号计算操作，结果形状应该相同但符号不同
  res = calculateSymbolicShapesOnOp(schema, {ss2, const_size_2});
  auto res3_val = res->at(0);

  assertShapeEqual(res3_val, res2_val);
  EXPECT_NE(res3_val, res2_val);  // 验证符号不同
  EXPECT_EQ(get_shape_cache_size(), 1);

  // 使用不同具体形状进行操作，应该被单独缓存
  res = calculateSymbolicShapesOnOp(schema, {ss1, const_size_3});
  assertShapeEqual(res, {sym_dim, 20});
  EXPECT_EQ(get_shape_cache_size(), 2);

  // 使用符号形状 ss3 和具体形状 const_size_3 进行操作，结果应该与之前相同
  res = calculateSymbolicShapesOnOp(schema, {ss3, const_size_3});
  assertShapeEqual(res, {sym_dim, 20});
  EXPECT_EQ(get_shape_cache_size(), 3);

  // 使用相同的符号形状 ss3 进行操作，结果应该是相同的符号形状
  res = calculateSymbolicShapesOnOp(schema, {ss3, ss3});
  assertShapeEqual(res, {sym_dim, sym_dim});
  EXPECT_EQ(get_shape_cache_size(), 4);
}
TEST(ShapeAnalysisTest, ShapeCacheMultipleFns) {
  // 清空形状缓存
  clear_shape_cache();

  // 获取操作的架构定义
  auto squeeze_op =
      getSchema("aten::squeeze.dim(Tensor(a) self, int dim) -> Tensor(a)");
  auto mul_tensor =
      getSchema("aten::mul.Tensor(Tensor self, Tensor other) -> Tensor");
  auto mul_scalar =
      getSchema("aten::mul.Scalar(Tensor self, Scalar other) -> Tensor");
  auto div_tensor =
      getSchema("aten::div.Tensor(Tensor self, Tensor other) -> Tensor");
  auto matmul = getSchema("aten::mm(Tensor self, Tensor mat2) -> Tensor");

  // 定义常量整数值
  c10::IValue const_int = 1;

  // 定义符号形状对象 ss1
  c10::SymbolicShape ss1 = c10::SymbolicShape({sym_dim, 64});

  // 计算 squeeze_op 操作的符号形状
  auto res = calculateSymbolicShapesOnOp(squeeze_op, {ss1, const_int});
  // 断言结果形状与期望形状一致
  assertShapeEqual(res, {sym_dim, 64});

  // 展示缓存能处理多个函数的情况
  res = calculateSymbolicShapesOnOp(mul_scalar, {ss1, const_int});
  assertShapeEqual(res, {sym_dim, 64});
  // 断言形状缓存的大小为 2
  EXPECT_EQ(get_shape_cache_size(), 2);

  // 继续测试，计算 mul_tensor 操作的符号形状
  res = calculateSymbolicShapesOnOp(mul_tensor, {ss1, ss1});
  assertShapeEqual(res, {sym_dim, 64});
  // 断言形状缓存的大小为 3
  EXPECT_EQ(get_shape_cache_size(), 3);

  // 即使期望结果相同，div_tensor 操作也不应该冲突
  res = calculateSymbolicShapesOnOp(div_tensor, {ss1, ss1});
  assertShapeEqual(res, {sym_dim, 64});
  // 断言形状缓存的大小为 4
  EXPECT_EQ(get_shape_cache_size(), 4);

  // 不应该丢失缓存的对象
  res = calculateSymbolicShapesOnOp(mul_scalar, {ss1, const_int});
  assertShapeEqual(res, {sym_dim, 64});
  // 断言形状缓存的大小为 4（确认未增加）
  EXPECT_EQ(get_shape_cache_size(), 4);

  // 再次计算 matmul 操作的符号形状
  res = calculateSymbolicShapesOnOp(matmul, {ss1, ss1});
  // SSA 可以推断出 sym_dim 是 64，因为两个张量使用相同的 sym_dim
  assertShapeEqual(res, {64, 64});
  // 断言形状缓存的大小为 5
  EXPECT_EQ(get_shape_cache_size(), 5);
}

TEST(ShapeAnalysisTest, TestShapeMultipleReturns) {
  // 清空形状缓存
  clear_shape_cache();

  // 获取 max_dim_op 操作的架构定义
  auto max_dim_op = getSchema(
      "aten::max.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)");
  // 定义常量整数值和布尔值
  c10::IValue const_int = 1;
  c10::IValue false_ival = false;

  // 定义符号形状对象 ss1 和 ss2
  c10::SymbolicShape ss1 = c10::SymbolicShape({sym_dim, 64});
  c10::SymbolicShape ss2 = c10::SymbolicShape({sym_dim, 64});

  // 计算 max_dim_op 操作的符号形状
  auto res =
      calculateSymbolicShapesOnOp(max_dim_op, {ss1, const_int, false_ival});
  // 定义期望的结果形状
  c10::SymbolicShape expected_res =
      c10::SymbolicShape(std::vector<std::optional<int64_t>>{sym_dim});
  // 断言第一个返回结果的形状与期望形状一致
  assertShapeEqual(res->at(0), expected_res);
  // res0 和 res1 应该共享相同的符号符号
  EXPECT_EQ(res->at(0), res->at(1));

  // 进一步测试形状缓存是否返回一致的结果形状
  res = calculateSymbolicShapesOnOp(max_dim_op, {ss2, const_int, false_ival});
  assertShapeEqual(res->at(0), expected_res);
  // 断言形状缓存的大小为 1
  EXPECT_EQ(get_shape_cache_size(), 1);
}
} // namespace jit
} // namespace torch
```