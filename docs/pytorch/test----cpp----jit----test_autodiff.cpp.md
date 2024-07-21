# `.\pytorch\test\cpp\jit\test_autodiff.cpp`

```
// 包含 Google Test 框架的头文件
#include <gtest/gtest.h>

// 包含用于测试的自定义实用工具的头文件
#include "test/cpp/jit/test_utils.h"

// 包含 Torch 前端追踪器的头文件
#include "torch/csrc/jit/frontend/tracer.h"

// 包含 Torch JIT passes 中的常见子表达式消除的头文件
#include "torch/csrc/jit/passes/common_subexpression_elimination.h"

// 包含 Torch JIT passes 中的常量传播的头文件
#include "torch/csrc/jit/passes/constant_propagation.h"

// 包含 Torch JIT passes 中的自动微分子图创建的头文件
#include "torch/csrc/jit/passes/create_autodiff_subgraphs.h"

// 包含 Torch JIT passes 中的死代码消除的头文件
#include "torch/csrc/jit/passes/dead_code_elimination.h"

// 包含 Torch JIT passes 中的图融合的头文件
#include "torch/csrc/jit/passes/graph_fuser.h"

// 包含 Torch JIT passes 中的梯度下降的头文件
#include "torch/csrc/jit/passes/lower_grad_of.h"

// 包含 Torch JIT passes 中的梯度分析的头文件
#include "torch/csrc/jit/passes/requires_grad_analysis.h"

// 包含 Torch JIT passes 中的形状分析的头文件
#include "torch/csrc/jit/passes/shape_analysis.h"

// 包含 Torch JIT passes 中的子图工具的头文件
#include "torch/csrc/jit/passes/utils/subgraph_utils.h"

// 包含 Torch JIT 运行时参数规范的头文件
#include "torch/csrc/jit/runtime/argument_spec.h"

// 包含 Torch JIT 自动微分的头文件
#include "torch/csrc/jit/runtime/autodiff.h"

// 包含 Torch JIT 图迭代器的头文件
#include "torch/csrc/jit/runtime/graph_iterator.h"

// 包含 Torch JIT 性能图执行器实现的头文件
#include "torch/csrc/jit/runtime/profiling_graph_executor_impl.h"

// 包含 Torch 主库的头文件
#include "torch/torch.h"

// 包含 ATen 的头文件
#include <ATen/ATen.h>

// 包含 Torch 自动求导引擎的头文件
#include "torch/csrc/autograd/engine.h"

// 包含 Torch 自动生成的变量工厂的头文件
#include "torch/csrc/autograd/generated/variable_factories.h"

// 包含 Torch 自动求导变量的头文件
#include "torch/csrc/autograd/variable.h"

// 命名空间声明：torch::jit
namespace torch {
namespace jit {

// 使用 torch::autograd 命名空间
using namespace torch::autograd;

// 定义变量元类型为 int64_t 的向量
using var_meta_type = std::vector<int64_t>;

// 定义变量元类型向量的向量
using var_meta_list = std::vector<var_meta_type>;

// 定义测试函数类型为接受变量列表并返回变量列表的函数
using test_fn_type = std::function<variable_list(const variable_list&)>;

// ADTestSpec 结构体：用于描述自动微分测试规范
struct ADTestSpec {
  // 构造函数：初始化 ADTestSpec 对象
  ADTestSpec(
      const char* name,
      var_meta_list input_meta, // 输入的变量元信息列表
      test_fn_type test_fn,     // 测试函数
      float clampMax = -1.0f)   // 最大截断值，默认为 -1.0
      : name(name),
        input_meta(input_meta),
        test_fn(test_fn),
        clampMax(clampMax) {}

  // 操作符重载：调用 ADTestSpec 对象，执行测试函数并返回结果
  variable_list operator()(const variable_list& inputs) const {
    return test_fn(inputs);
  };

  // 创建变量函数：根据输入元信息创建变量列表
  std::vector<Variable> make_vars() const {
    std::vector<Variable> out;
    for (const auto& m : input_meta) {
      if (clampMax > 0.0f) {
        // 若最大截断值大于 0，则创建在指定范围内截断的随机变量
        out.push_back(torch::randn(m, at::requires_grad(true))
                          .clamp(-clampMax, clampMax));
        continue;
      }
      // 创建指定元信息的随机变量
      out.push_back(torch::randn(m, at::requires_grad(true)));
    }
    return out;
  }

  const char* name;      // 测试规范的名称
  var_meta_list input_meta; // 输入变量的元信息列表
  test_fn_type test_fn;  // 测试函数
  float clampMax;        // 截断的最大值
};

// get_grad_outputs 函数：获取变量列表的梯度输出
variable_list get_grad_outputs(const variable_list& vars) {
  return fmap(vars, [](const Variable& v) -> Variable {
    return at::randn(v.sizes(), v.options());
  });
}

// grad 函数：计算梯度
variable_list grad(
    const variable_list& outputs,
    const variable_list& inputs,
    const variable_list& grad_outputs) {
  const auto get_edge = [](const Variable& v) {
    return torch::autograd::impl::gradient_edge(v);
  };
  auto& engine = torch::autograd::Engine::get_default_engine();
  return engine.execute(
      fmap(outputs, get_edge),
      grad_outputs,
      true,
      false,
      false,
      fmap(inputs, get_edge));
}

// AutodiffTest 测试套件：ADFormulas 测试用例
TEST(AutodiffTest, ADFormulas) {
  // cast 函数：将变量转换为特定类型
  const auto cast = [](const Variable& v) {
    // 将变量 v 强制转换为 at::Tensor 类型，并返回结果
    return static_cast<at::Tensor>(v);
    };
    
    // 定义变量别名 VL 为 variable_list 类型的别名
    using VL = variable_list;
    
    // 定义二元逐点操作的变量元信息列表
    const var_meta_list binary_pointwise = {{2, 3, 4, 5}, {2, 3, 4, 5}};
    
    // 定义一元逐点操作的变量元信息列表
    const var_meta_list unary_pointwise = {{2, 3, 4, 5}};
    
    // 定义二维一元逐点操作的变量元信息列表
    const var_meta_list unary_pointwise_2d = {{2, 3}};
    
    // 定义自动微分测试用例的列表 ad_tests
    const std::vector<ADTestSpec> ad_tests = {
        // "add" 操作的测试，对应的操作函数为 v[0] + v[1]
        {"add",
         binary_pointwise,
         [](const VL& v) -> VL { return {v[0] + v[1]}; }},
        // "sub" 操作的测试，对应的操作函数为 v[0] - v[1]
        {"sub",
         binary_pointwise,
         [](const VL& v) -> VL { return {v[0] - v[1]}; }},
        // "mul" 操作的测试，对应的操作函数为 v[0] * v[1]
        {"mul",
         binary_pointwise,
         [](const VL& v) -> VL { return {v[0] * v[1]}; }},
        // "sigmoid" 操作的测试，对应的操作函数为 v[0].sigmoid()
        {"sigmoid",
         unary_pointwise,
         [](const VL& v) -> VL { return {v[0].sigmoid()}; }},
        // "tanh" 操作的测试，对应的操作函数为 v[0].tanh()，并限制输入张量值在 [-3, 3] 内
        {"tanh",
         unary_pointwise,
         [](const VL& v) -> VL { return {v[0].tanh()}; },
         3.0f},
        // "t" 操作的测试，对应的操作函数为 v[0].t()
        {"t", unary_pointwise_2d, [](const VL& v) -> VL { return {v[0].t()}; }},
        // "view" 操作的测试，对应的操作函数为 v[0].view({3, 2})
        {"view",
         unary_pointwise_2d,
         [](const VL& v) -> VL {
           return {v[0].view({3, 2})};
         }},
        // "expand" 操作的测试，对应的操作函数为 v[0].expand({2, 3})
        {"expand",
         {{2, 1}},
         [](const VL& v) -> VL {
           return {v[0].expand({2, 3})};
         }},
        // "mm" 操作的测试，对应的操作函数为 v[0].mm(v[1])
        {"mm",
         {{10, 12}, {12, 15}},
         [](const VL& v) -> VL { return {v[0].mm(v[1])}; }},
    };
    
    // 遍历自动微分测试用例 ad_tests
    for (const auto& test : ad_tests) {
      // 获取自动微分测试用例的输入变量
      auto vars_in = test.make_vars();
      // 调用测试用例函数计算输出变量
      auto vars_out = test(vars_in);
      // 获取输出变量的梯度输入
      auto var_grads_in = get_grad_outputs(vars_out);
      // 计算输出变量相对于输入变量的梯度
      auto var_grads_out = grad(vars_out, vars_in, var_grads_in);
    
      // 跟踪和微分操作
      auto graph = tracer::trace(
                       fmap<IValue>(vars_in),
                       [&test](Stack in) -> Stack {
                         // 将输入栈中的 IValue 转换为 Variable 类型
                         auto ivalue_inps = fmap(in, [](const IValue& v) {
                           return Variable(v.toTensor());
                         });
                         // 对测试函数进行调用，并返回其结果
                         return fmap<IValue>(test(ivalue_inps));
                       },
                       [](const Variable& var) { return ""; })
                       .first->graph;
      EliminateDeadCode(graph); // 执行死代码消除优化
      ConstantPropagation(graph); // 执行常量传播优化
      auto grad_spec = differentiate(graph); // 计算梯度规范
      LowerGradOf(*grad_spec.df); // 降低梯度的规范
      // 从解释器获取输出张量
      auto tensors_in = fmap(vars_in, cast);
      // 从解释器获取张量的梯度输入
      auto tensor_grads_in = fmap(var_grads_in, cast);
    //`
    // 声明 tensor_list 类型的变量 tensors_out 和 tensor_grads_out
    tensor_list tensors_out, tensor_grads_out;
    // 调用 runGradient 函数，传入 grad_spec, tensors_in 和 tensor_grads_in，返回结果赋值给 tensors_out 和 tensor_grads_out
    std::tie(tensors_out, tensor_grads_out) =
        runGradient(grad_spec, tensors_in, tensor_grads_in);

    // 将 vars_out 中的每个元素通过 cast 函数进行转换，并赋值给 expected_tensors_out
    auto expected_tensors_out = fmap(vars_out, cast);
    // 将 var_grads_out 中的每个元素通过 cast 函数进行转换，并赋值给 expected_tensor_grads_out
    auto expected_tensor_grads_out = fmap(var_grads_out, cast);
    // 使用 assertAllClose 函数验证 tensors_out 和 expected_tensors_out 是否接近，若不接近会抛出异常
    assertAllClose(tensors_out, expected_tensors_out);
    // 使用 assertAllClose 函数验证 tensor_grads_out 和 expected_tensor_grads_out 是否接近，若不接近会抛出异常
    assertAllClose(tensor_grads_out, expected_tensor_grads_out);
  }
TEST(AutodiffTest, Differentiate) {
  // 注意：由于问题＃23989，无法在此测试中使用IRParser
  auto graph = std::make_shared<Graph>();
  // 定义张量的尺寸和步长
  std::vector<int64_t> sizes{2, 3, 4};
  std::vector<int64_t> strides{12, 4, 1};
  // 创建张量类型对象
  const auto type = TensorType::create(
      at::ScalarType::Float,
      at::kCPU,
      c10::VaryingShape<int64_t>{sizes},
      c10::VaryingShape<int64_t>{strides},
      true);

  // 构建计算图：a * b * a + b
  auto* a = graph->addInput()->setType(type);  // 添加输入节点 a
  auto* b = graph->addInput()->setType(type);  // 添加输入节点 b
  auto* cOne = graph->insertConstant(1);  // 插入常量节点 1

  auto* ab = graph->insertNode(graph->create(aten::mul, /*num_outputs =*/1));  // 插入乘法节点 ab
  ab->addInput(a);
  ab->addInput(b);

  auto* aba = graph->insertNode(graph->create(aten::mul, /*num_outputs =*/1));  // 插入乘法节点 aba
  aba->addInput(ab->output());
  aba->addInput(a);

  auto* abaplusb =
      graph->insertNode(graph->create(aten::add, /*num_outputs =*/1));  // 插入加法节点 ab + b
  abaplusb->addInput(aba->output());
  abaplusb->addInput(b);
  abaplusb->addInput(cOne);

  graph->registerOutput(abaplusb->output());  // 注册输出节点

  // 对计算图进行自动微分
  auto grad_spec = differentiate(graph);
  // 预期的捕获输入节点索引
  std::vector<size_t> expected_captured_inputs = {0, 1};
  // 预期的捕获输出节点索引
  std::vector<size_t> expected_captured_outputs = {1, 2, 3, 4, 5, 6, 7};
  // 预期的输入梯度节点索引
  std::vector<size_t> expected_input_vjps = {0, 1};
  // 预期的输出梯度节点索引
  std::vector<size_t> expected_output_vjps = {0, 1};
  // 断言各项微分特性
  ASSERT_EQ(grad_spec.f_real_outputs, 1);
  ASSERT_EQ(grad_spec.df_input_captured_inputs, expected_captured_inputs);
  ASSERT_EQ(grad_spec.df_input_captured_outputs, expected_captured_outputs);
  ASSERT_EQ(grad_spec.df_input_vjps, expected_input_vjps);
  ASSERT_EQ(grad_spec.df_output_vjps, expected_output_vjps);
  // 运行文件检查以验证计算图结构
  testing::FileCheck()
      .check_count("aten::mul", 2)
      ->check("aten::size")
      ->check("aten::add")
      ->run(*grad_spec.f);
  // 运行文件检查以验证微分计算图结构
  testing::FileCheck()
      .check("prim::GradOf[name=\"aten::add\"]")
      ->check_count("prim::GradOf[name=\"aten::mul\"]", 2)
      ->check_count("AutogradAdd", 2)
      ->run(*grad_spec.df);
}

TEST(AutodiffTest, DifferentiateWithRequiresGrad) {
  const auto graph_string = R"IR(
    // 定义一个名为 graph 的函数，接受两个 Tensor 类型的参数 %0 和 %1
    graph(%0 : Tensor,
          %1 : Tensor):
      // 创建一个整数常量 %2 值为 1
      %2 : int = prim::Constant[value=1]()
      // 计算 %1 与自身的乘积，结果存储在 %3 中
      %3 : Tensor = aten::mul(%1, %1)
      // 计算 %3 与 %1 的和，加上 %2，结果存储在 %4 中
      %4 : Tensor = aten::add(%3, %1, %2)
      // 计算 %4 与 %0 的和，加上 %2，结果存储在 %5 中
      %5 : Tensor = aten::add(%4, %0, %2)
      // 计算 %5 与 %0 的乘积，结果存储在 %6 中
      %6 : Tensor = aten::mul(%5, %0)
      // 计算 %6 与 %1 的和，加上 %2，结果存储在 %7 中
      %7 : Tensor = aten::add(%6, %1, %2)
      // 返回 %4 和 %7 作为结果
      return (%4, %7))IR";
    // 创建一个共享指针 g，指向 Graph 类对象
    auto g = std::make_shared<Graph>();
    // 解析 graph_string 中的 IR 到 g
    torch::jit::parseIR(graph_string, g.get());
    
    // 创建一个自动求导的变量 a_var
    auto a_var = autograd::make_variable(
        at::empty_strided(2, 2, at::CPU(at::kFloat).options()), true);
    // 创建一个非自动求导的变量 b_var
    auto b_var = autograd::make_variable(
        at::empty_strided(2, 2, at::CPU(at::kFloat).options()), false);
    
    // 创建 ArgumentSpecCreator 对象 asc，用于分析和生成函数的参数规范
    ArgumentSpecCreator asc(*g);
    // 根据变量 a_var 和 b_var 生成特定类型的函数参数规范
    asc.specializeTypes(*g, asc.create(true, {a_var, b_var}));
    
    // 传播输入的形状信息到 Graph 对象 g
    PropagateInputShapes(g);
    // 传播需要梯度的信息到 Graph 对象 g
    PropagateRequiresGrad(g);
    
    // 对 Graph 对象 g 进行自动求导，得到梯度规范 grad_spec
    auto grad_spec = differentiate(g);
    // 预期的输入变量 JVPs 的索引，对应 {1, 2}，即 e 和 %4 = (d + a)
    std::vector<size_t> expected_input_vjps = {1, 2};
    // 预期的输出变量 JVPs 的索引，只有 a 需要梯度，对应 {0}
    std::vector<size_t> expected_output_vjps = {0};
    // 断言梯度规范中的真实输出数为 2
    ASSERT_EQ(grad_spec.f_real_outputs, 2);
    // 断言梯度规范中 df_input_captured_inputs 的内容为 {0}
    ASSERT_EQ(grad_spec.df_input_captured_inputs, std::vector<size_t>({0}));
    // 断言梯度规范中 df_input_captured_outputs 的内容为 {2, 3, 4, 5, 6}
    ASSERT_EQ(
        grad_spec.df_input_captured_outputs,
        std::vector<size_t>({2, 3, 4, 5, 6}));
    // 断言梯度规范中 df_input_vjps 的内容与 expected_input_vjps 相等
    ASSERT_EQ(grad_spec.df_input_vjps, expected_input_vjps);
    // 断言梯度规范中 df_output_vjps 的内容与 expected_output_vjps 相等
    ASSERT_EQ(grad_spec.df_output_vjps, expected_output_vjps);
    
    // 运行 FileCheck 对 grad_spec.f 进行多个匹配
    testing::FileCheck()
        .check("aten::mul")
        ->check_count("aten::add", 2)
        ->check("aten::mul")
        ->check("aten::size")
        ->check("aten::add")
        ->run(*grad_spec.f);
    
    // 运行 FileCheck 对 grad_spec.df 进行匹配，确保 "aten::mul" 的数量为 1
    testing::FileCheck()
        .check_count("prim::GradOf[name=\"aten::mul\"]", 1, /*exactly*/ true)
        ->run(*grad_spec.df);
} // 结束 AutodiffRemoveUnusedGradientsTest 类定义

class AutodiffRemoveUnusedGradientsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // 保存并设置先前的执行模式，并设置为执行模式
    prev_exec = getExecutorMode();
    getExecutorMode() = true;
    // 保存并设置先前的自动微分子图内联选项，并关闭自动微分子图内联
    prev_inline_autodiff = getAutodiffSubgraphInlining();
    debugSetAutodiffSubgraphInlining(false);
  }
  void TearDown() override {
    // 恢复先前的执行模式
    getExecutorMode() = prev_exec;
    // 恢复先前的自动微分子图内联选项
    debugSetAutodiffSubgraphInlining(prev_inline_autodiff);
  }

  bool prev_exec; // 先前的执行模式
  bool prev_profiling; // 先前的性能分析
  bool prev_inline_autodiff; // 先前的自动微分子图内联选项
};

TEST_F(AutodiffRemoveUnusedGradientsTest, Linear) {
  auto graph = std::make_shared<Graph>();
  const std::string input =
      R"IR(
graph(%inp.1 : Tensor,
      %weight.1 : Tensor,
      %bias.1 : Tensor):
  %6 : Tensor = aten::linear(%inp.1, %weight.1, %bias.1)
  return (%6))IR";
  parseIR(input, graph.get());

  auto inp = torch::randn({10, 10}).requires_grad_(false); // 创建随机输入张量，不需要梯度
  auto weight = torch::randn({10, 10}).requires_grad_(true); // 创建随机权重张量，需要梯度
  auto bias = torch::randn({1, 10}).requires_grad_(true); // 创建随机偏置张量，需要梯度
  auto stack = createStack({inp, weight, bias}); // 创建输入栈

  ProfilingGraphExecutorImpl executor(graph, "linear"); // 使用图和名称创建性能分析图执行器

  // 初次运行以进行性能分析并获取 requires_grad 信息
  auto plan = executor.getPlanFor(stack, 20);
  InterpreterState is{plan.code};
  is.run(stack);

  auto optimized_plan = executor.getPlanFor(stack, 20);
  DepthFirstGraphNodeIterator it(optimized_plan.graph);
  Node* diff_graph_node = nullptr;

  // 遍历优化后的执行计划，找到第一个微分图节点
  while ((diff_graph_node = it.next()) != nullptr) {
    if (diff_graph_node->kind() == prim::DifferentiableGraph) {
      break;
    }
  }
  ASSERT_NE(nullptr, diff_graph_node); // 断言确保找到微分图节点

  auto backward_graph = diff_graph_node->g(attr::ReverseSubgraph);

  // 预期计算 grad_weight（需要一个 matmul），但不预期计算 grad_input。
  // 因此，预期只有一个 matmul 操作。
  // 注意：这可能会改变，例如如果使用 mm 而不是 matmul。
  testing::FileCheck().check_count("matmul", 1, true)->run(*backward_graph);
}

} // 结束命名空间 jit
} // 结束命名空间 torch
```