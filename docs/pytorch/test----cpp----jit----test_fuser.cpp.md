# `.\pytorch\test\cpp\jit\test_fuser.cpp`

```py
#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <ATen/core/interned_strings.h>
#include <ATen/core/ivalue.h>
#include <c10/util/irange.h>

#include <torch/csrc/autograd/engine.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/codegen/cuda/interface.h>
#include <torch/csrc/jit/codegen/fuser/interface.h>
#include <torch/csrc/jit/frontend/ir_emitter.h>
#include <torch/csrc/jit/frontend/tracer.h>
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/ir/attributes.h>
#include <torch/csrc/jit/ir/irparser.h>
#include <torch/csrc/jit/passes/canonicalize.h>
#include <torch/csrc/jit/passes/common_subexpression_elimination.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/create_autodiff_subgraphs.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/graph_fuser.h>
#include <torch/csrc/jit/passes/lower_grad_of.h>
#include <torch/csrc/jit/passes/lower_tuples.h>
#include <torch/csrc/jit/passes/requires_grad_analysis.h>
#include <torch/csrc/jit/passes/shape_analysis.h>
#include <torch/csrc/jit/passes/utils/subgraph_utils.h>
#include <torch/csrc/jit/runtime/argument_spec.h>
#include <torch/csrc/jit/runtime/autodiff.h>
#include <torch/csrc/jit/runtime/custom_operator.h>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <torch/csrc/jit/runtime/interpreter.h>
#include <torch/csrc/jit/runtime/symbolic_script.h>
#include <torch/csrc/jit/serialization/import.h>
#include <torch/csrc/jit/testing/file_check.h>

#include <onnx/onnx_pb.h>

#include <c10/util/Exception.h>

#include <algorithm>
#include <cstddef>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

// 命名空间 torch::jit 内定义了一系列与 JIT 相关的类和函数
namespace torch {
namespace jit {

// 定义 FuserTest 类，继承自测试框架的 ::testing::Test 类
class FuserTest : public ::testing::Test {
  // 在每个测试用例执行之前调用的设置方法
  void SetUp() override {
    // 设置 CUDA 的 fuser 为禁用，并保存原始的设置值
    old_nvfuser_value_ = fuser::cuda::setEnabled(false);
  }
  // 在每个测试用例执行之后调用的清理方法
  void TearDown() override {
    // 恢复 CUDA 的 fuser 设置为之前保存的值
    fuser::cuda::setEnabled(old_nvfuser_value_);
  }

 private:
  bool old_nvfuser_value_; // 保存旧的 CUDA fuser 设置值
};

// 测试用例 FuserTest.TestSimple_CUDA，测试简单的 CUDA 图执行
TEST_F(FuserTest, TestSimple_CUDA) {
#if defined(FBCODE_CAFFE2)
  return; // 如果在 Facebook 的代码中，跳过测试
#endif
  // 定义一个表示计算图的字符串
  const auto graph_string = R"IR(
      graph(%0 : Tensor,
            %1 : Tensor):
        %2 : Tensor = aten::mul(%0, %1)
        return (%2))IR";
  Graph graph;
  // 解析上面定义的计算图字符串到 Graph 对象中
  torch::jit::parseIR(graph_string, &graph);

  // 创建 CUDA 设备上的张量 a、b、o
  auto a = at::rand({3, 4}, at::kCUDA);
  auto b = at::rand({4, 3}, at::kCUDA).transpose(0, 1);
  auto o = at::zeros({3, 4}, at::kCUDA);
  
  // 调用 debugLaunchGraph 函数执行计算图，并获取输出张量
  auto outputs = debugLaunchGraph(graph, {a, b});
  ASSERT_EQ(outputs.size(), 1); // 断言输出张量的数量为1

  // 在 CPU 上计算期望的输出张量 o2
  auto o2 = a * b;
  // 计算实际输出张量与期望输出张量的最大差异
  float max_diff = (o2 - outputs[0]).abs().max().item<double>();
  // 断言最大差异为0
  ASSERT_EQ(max_diff, 0);
}

// 测试用例 FuserTest.TestOne_CUDA，测试单个 CUDA 图的执行
TEST_F(FuserTest, TestOne_CUDA) {
#if defined(FBCODE_CAFFE2)
  return; // 如果在 Facebook 的代码中，跳过测试
#endif
#endif
auto testOne = [&](int ti, int tj) {
  // 定义包含 IR 的字符串，描述一个特定的计算图
  const auto graph_string = R"IR(
    graph(%0 : Tensor,
          %1 : Tensor,
          %2 : Tensor,
          %3 : Tensor,
          %4 : Tensor):
      %5 : Tensor = aten::sigmoid(%4)
      %6 : Tensor = aten::sigmoid(%3)
      %7 : Tensor = aten::tanh(%2)
      %8 : Tensor = aten::sigmoid(%1)
      %9 : Tensor = aten::mul(%6, %0)
      %10 : Tensor = aten::mul(%5, %7)
      %11 : int = prim::Constant[value=1]()
      %12 : Tensor = aten::add(%9, %10, %11)
      %13 : Tensor = aten::tanh(%12)
      %14 : Tensor = aten::mul(%8, %13)
      return (%14, %12))IR";
  
  // 创建一个空白的计算图对象
  Graph graph;
  // 从字符串中解析 IR，填充计算图对象
  torch::jit::parseIR(graph_string, &graph);

  // 对计算图进行静态分析和检查
  graph.lint();

  // 准备输入张量向量
  std::vector<at::Tensor> inputs;
  
  // 生成与计算图输入张量相匹配的随机张量
  std::generate_n(
      std::back_inserter(inputs), graph.inputs().size(), [ti, tj] {
        // 定义张量的维度，通过维度交换生成不同内部步长的张量视图
        std::array<int64_t, 3> dims = {128, 128, 32};
        std::swap(dims[ti], dims[tj]);
        // 创建具有指定维度的 CUDA 张量，然后对维度进行交换
        return at::rand(dims, at::kCUDA).transpose(ti, tj);
      });

  // 计算输入张量的一系列操作，生成输出张量 out0
  auto t22 = inputs[4].sigmoid();
  auto t20 = inputs[3].sigmoid();
  auto t18 = inputs[2].tanh();
  auto t16 = inputs[1].sigmoid();
  auto t14 = t20 * inputs[0];
  auto t11 = t22 * t18;
  auto out1 = t14 + t11;
  auto t5 = out1.tanh();
  auto out0 = t16 * t5;

  // 执行调试版本的计算图，获取输出结果
  auto outputs = debugLaunchGraph(graph, inputs);
  
  // 断言：输出张量的数量与计算图输出的数量相等
  ASSERT_EQ(outputs.size(), graph.outputs().size());
  // 断言：out0 与 debugLaunchGraph 返回的第一个输出张量大小相同
  ASSERT_TRUE(out0.is_same_size(outputs.front()));
  // 计算输出张量与 out0 之间的最大差异，检查误差是否在可接受范围内
  float max_diff = (outputs.front() - out0).abs().max().item<double>();
  ASSERT_TRUE(max_diff < 1e-6);
};
// 使用 testOne 函数执行一系列测试
testOne(0, 0);
testOne(0, 1);
testOne(1, 2);
testOne(0, 2);
}

TEST_F(FuserTest, FusedConcat_CUDA) {
#if defined(FBCODE_CAFFE2)
return;
#endif
const auto graph_string0 = R"IR(
graph(%0 : Tensor,
      %1 : Tensor):
  %2 : Tensor = aten::mul(%0, %1)
  %3 : Tensor = prim::FusedConcat[dim=0](%0, %2)
  return (%2, %3))IR";
const auto graph_string1 = R"IR(
graph(%0 : Tensor,
      %1 : Tensor):
  %2 : Tensor = aten::mul(%0, %1)
  %3 : Tensor = prim::FusedConcat[dim=1](%0, %2)
  return (%2, %3))IR";
const auto graph_string2 = R"IR(
graph(%0 : Tensor,
      %1 : Tensor):
  %2 : Tensor = aten::mul(%0, %1)
  %3 : Tensor = prim::FusedConcat[dim=2](%0, %2)
  return (%2, %3))IR";

auto a = at::rand({3, 4, 5}, at::kCUDA);
auto b = at::rand({4, 3, 5}, at::kCUDA).transpose(0, 1);
const auto o_r = a * b;

// 将计算图字符串存储在向量中
std::vector<std::string> graph_strings{
  graph_string0, graph_string1, graph_string2};
// 遍历每个计算图字符串
for (const auto i : c10::irange(graph_strings.size())) {
  // 创建空白计算图对象
  Graph g;
  // 从字符串中解析 IR，填充计算图对象
  torch::jit::parseIR(graph_strings[i], &g);

  // 执行调试版本的计算图，传入张量 a 和 b 作为输入
  auto outputs = debugLaunchGraph(g, {a, b});
    # 断言输出的张量大小为2
    ASSERT_EQ(outputs.size(), 2);
    
    # 计算第一个输出张量 o_r 与 outputs[0] 的绝对差的最大值，转换为双精度浮点数
    float max_diff = (o_r - outputs[0]).abs().max().item<double>();
    # 断言最大差值为0
    ASSERT_EQ(max_diff, 0);
    
    # 将张量 a 和 o_r 沿指定维度 i 进行连接，得到 o2_r
    const auto o2_r = at::cat({a, o_r}, i);
    # 计算 o2_r 与 outputs[1] 的绝对差的最大值，转换为双精度浮点数
    float max_diff2 = (o2_r - outputs[1]).abs().max().item<double>();
    # 断言最大差值为0
    ASSERT_EQ(max_diff2, 0);
}

TEST_F(FuserTest, FusionAliasing) {
#if defined(FBCODE_CAFFE2)
  return;
#endif

  // 定义一个包含IR的字符串，描述计算图的结构和操作
  const auto graph_string = R"IR(
    graph(%0 : Tensor,
          %1 : Tensor):
      %12 : int = prim::Constant[value=1]()
      %2.1 : Tensor = aten::mul(%0, %1)
      %2 : Tensor = aten::mul(%2.1, %1)
      %3 : Tensor = aten::add_(%2, %1, %12)
      %4 : Tensor = aten::mul(%2, %1)
      %5 : Tensor = aten::add(%2, %4, %12)
      return (%5))IR";

  // 创建一个空的图形对象
  auto g = std::make_shared<Graph>();
  // 解析IR字符串并将其加载到图形对象中
  torch::jit::parseIR(graph_string, g.get());

  // 对图进行静态分析
  g->lint();
  // 对图进行融合操作
  FuseGraph(g);

  // 进行断言检查，验证是否成功进行了融合
  // 检查是否出现了FusionGroup_0节点、aten::add_节点和FusionGroup_1节点
  testing::FileCheck()
      .check("prim::FusionGroup_0")
      ->check("aten::add_")
      ->check("prim::FusionGroup_1")
      ->run(*g);
}

TEST_F(FuserTest, KernelCaching) {
#if defined(FBCODE_CAFFE2)
  return;
#endif

  // 构建两个功能等价的计算图
  const auto graph0_string = R"IR(
    graph(%0 : Float(2, 3, 4),
          %1 : Float(2, 3, 4)):
      %c0 : Float(2, 3, 4) = aten::mul(%0, %1)
      %d0 : Float(2, 3, 4) = aten::mul(%c0, %0)
      return (%d0))IR";
  auto g0 = std::make_shared<Graph>();
  torch::jit::parseIR(graph0_string, g0.get());

  const auto graph1_string = R"IR(
    graph(%0 : Float(2, 3, 4),
          %1 : Float(2, 3, 4)):
      %c1 : Float(2, 3, 4) = aten::mul(%0, %1)
      %d1 : Float(2, 3, 4) = aten::mul(%c1, %0)
      return (%d1))IR";
  auto g1 = std::make_shared<Graph>();
  torch::jit::parseIR(graph1_string, g1.get());

  // 定义一个函数，用于获取图中的融合组
  auto getFusionGroup = [](const std::shared_ptr<Graph>& graph) {
    const auto& nodes = graph->nodes();
    auto maybe_fusion_group =
        std::find_if(nodes.begin(), nodes.end(), [](const Node* node) {
          return node->kind() == prim::FusionGroup;
        });
    TORCH_CHECK(
        maybe_fusion_group != nodes.end(),
        "testRegisterFusionCachesKernel: could not create FusionGroup");
    return *maybe_fusion_group;
  };

  // 允许在CPU上进行融合
  torch::jit::overrideCanFuseOnCPU(true);
  // 对两个图进行融合操作
  FuseGraph(g0);
  FuseGraph(g1);
  // 禁止在CPU上进行融合
  torch::jit::overrideCanFuseOnCPU(false);
  // 获取两个图中的融合组
  auto fg0 = getFusionGroup(g0);
  auto fg1 = getFusionGroup(g1);

  // 使用融合编译器注册两个融合组
  auto expected_key = registerFusion(fg0);
  auto second_key = registerFusion(fg1);

  // 由于图形是α等价的，它们应返回相同的键，
  // 因此共享一个KernelSpec以共享特化的内核
  ASSERT_EQ(second_key, expected_key);
}
} // namespace jit
} // namespace torch
```