# `.\pytorch\torch\csrc\jit\passes\fuse_linear.cpp`

```py
// 引入 Torch 库中的头文件，用于线性层融合优化
#include <torch/csrc/jit/passes/fuse_linear.h>
#include <torch/csrc/jit/passes/graph_rewrite_helper.h>
#include <torch/csrc/jit/passes/quantization/helper.h>
#include <torch/csrc/jit/passes/subgraph_rewrite.h>

// 定义 Torch 命名空间内的命名空间 jit 中的函数 FuseLinear
namespace torch {
namespace jit {

// 定义函数 FuseLinear，接受一个指向图的 shared_ptr
void FuseLinear(std::shared_ptr<Graph>& graph) {
  // 定义字符串 addmm_pattern，描述了要替换的 addmm 模式的图形 IR
  std::string addmm_pattern = R"IR(
    graph(%input, %weight_t, %bias, %beta, %alpha):
        %res = aten::addmm(%bias, %input, %weight_t, %beta, %alpha)
        return (%res))IR";

  // 定义字符串 fused_linear_addmm，描述了替换后的 linear 模式的图形 IR
  std::string fused_linear_addmm = R"IR(
    graph(%input, %weight_t, %bias, %beta, %alpha):
        %weight = aten::t(%weight_t)
        %res = aten::linear(%input, %weight, %bias)
        return (%res))IR";

  // lambda 函数 beta_is_one，检查 beta 是否为整数常量 1
  auto beta_is_one = [](const Match& match,
                        const std::unordered_map<std::string, Value*>& vmap) {
    return is_int_constant(match, vmap, "beta", 1);
  };

  // lambda 函数 weight_transposed，检查 weight_t 是否由 aten::t 生成，以确保可以转换模式为 aten::linear
  auto weight_transposed =
      [](const Match& match,
         const std::unordered_map<std::string, Value*>& vmap) {
        const auto& match_vmap = match.values_map;
        auto v = match_vmap.at(vmap.at("weight_t"));
        return v->node()->kind() == Symbol::aten("t");
      };

  // 创建 SubgraphRewriter 对象 addmm_to_linear，用于替换 addmm 模式为 linear 模式
  SubgraphRewriter addmm_to_linear;
  std::vector<std::pair<std::string, std::string>> value_mappings(
      {{"weight", "res"}, {"res", "res"}});
  addmm_to_linear.RegisterRewritePattern(
      addmm_pattern, fused_linear_addmm, value_mappings);
  // 在图上运行 SubgraphRewriter 对象 addmm_to_linear，应用注册的转换规则
  addmm_to_linear.runOnGraph(
      graph, {aten_add_alpha_is_one, beta_is_one, weight_transposed});

  // 定义字符串 matmul_add_pattern，描述了 matmul + add 模式的图形 IR
  std::string matmul_add_pattern = R"IR(
    graph(%input, %weight_t, %bias, %alpha):
        %output = aten::matmul(%input, %weight_t)
        %res = aten::add_(%output, %bias, %alpha)
        return (%res))IR";

  // 定义字符串 fused_linear_matmul，描述了替换后的 linear 模式的图形 IR
  std::string fused_linear_matmul = R"IR(
    graph(%input, %weight_t, %bias, %alpha):
        %weight = aten::t(%weight_t)
        %res = aten::linear(%input, %weight, %bias)
        return (%res))IR";

  // 更新 value_mappings 用于 matmul + add 模式的替换
  value_mappings = {{"weight", "output"}, {"res", "output"}};

  // 创建 SubgraphRewriter 对象 matmuladd_to_linear，用于替换 matmul + add 模式为 linear 模式
  SubgraphRewriter matmuladd_to_linear;
  matmuladd_to_linear.RegisterRewritePattern(
      matmul_add_pattern, fused_linear_matmul, value_mappings);
  // 在图上运行 SubgraphRewriter 对象 matmuladd_to_linear，应用注册的转换规则
  matmuladd_to_linear.runOnGraph(
      graph, {aten_add_alpha_is_one, weight_transposed});

  // 定义字符串 matmul_pattern，描述了 matmul 模式的图形 IR
  std::string matmul_pattern = R"IR(
    graph(%input, %weight_t):
        %output = aten::matmul(%input, %weight_t)
        return (%output))IR";

  // 定义字符串 fused_linear_bias_none，描述了替换后的 linear 模式的图形 IR
  std::string fused_linear_bias_none = R"IR(
    graph(%input, %weight_t):
        %weight = aten::t(%weight_t)
        %res = aten::linear(%input, %weight)
        return (%res))IR";
  // 定义一个图表达式（IR），包含输入和转置后的权重
  graph(%input, %weight_t):
      // 将权重矩阵转置
      %weight = aten::t(%weight_t)
      // 创建一个空的常量偏置
      %bias: Tensor? = prim::Constant()
      // 使用输入、转置后的权重和空的偏置进行线性变换操作
      %res = aten::linear(%input, %weight, %bias)
      // 返回线性变换的结果
      return (%res))IR";

// 用线性变换模式替换包含matmul且偏置为None的模式
SubgraphRewriter matmul_to_linear;
matmul_to_linear.RegisterRewritePattern(
    matmul_pattern, fused_linear_bias_none, value_mappings);
matmul_to_linear.runOnGraph(graph, weight_transposed);

// 清理掉aten::linear操作中多余的权重转置操作
std::string linear_weight_extra_transpose = R"IR(
  graph(%input, %weight, %bias):
      // 对权重进行两次转置操作
      %weight_t1 = aten::t(%weight)
      %weight_t2 = aten::t(%weight_t1)
      // 使用输入、最终转置后的权重和偏置进行线性变换操作
      %res = aten::linear(%input, %weight_t2, %bias)
      // 返回线性变换的结果
      return (%res))IR";

std::string linear_weight_no_transpose = R"IR(
  graph(%input, %weight, %bias):
      // 直接使用输入、不转置的权重和偏置进行线性变换操作
      %res = aten::linear(%input, %weight, %bias)
      // 返回线性变换的结果
      return (%res))IR";

// 设置值映射用于注册重写模式
value_mappings = {{"res", "res"}};
// 注册并运行子图重写器，将多余的权重转置操作清理掉
SubgraphRewriter cleanup;
cleanup.RegisterRewritePattern(
    linear_weight_extra_transpose,
    linear_weight_no_transpose,
    value_mappings);
cleanup.runOnGraph(graph);

// 对图中的FunctionalLinear进行交换操作
SwapFunctionalLinear(graph);
} // 结束 SwapFunctionalLinear 函数定义

void SwapFunctionalLinear(Module& module) {
  // 遍历模块中的每个方法
  for (auto& method : module.get_methods()) {
    // 获取当前方法的计算图
    std::shared_ptr<Graph> g = method.graph();
    // 调用下面定义的 SwapFunctionalLinear 函数处理当前计算图
    SwapFunctionalLinear(g);
  }
  // 遍历模块中的每个子模块
  for (Module m : module.children()) {
    // 递归调用 SwapFunctionalLinear 函数处理子模块
    SwapFunctionalLinear(m);
  }
}

void SwapFunctionalLinear(std::shared_ptr<Graph>& graph) {
  // 定义转换前的 functional_linear 图形式
  std::string functional_linear = R"(
graph(%linear, %input, %weight, %bias):
  %r = prim::CallFunction(%linear, %input, %weight, %bias)
  return (%r) )";
  // 定义转换后的 aten_linear 图形式
  std::string aten_linear = R"(
graph(%linear, %input, %weight, %bias):
  %r = aten::linear(%input, %weight, %bias)
  return (%r) )";

  // 定义用于筛选匹配的 lambda 函数 filter
  auto filter = [](const Match& match,
                   const std::unordered_map<std::string, Value*>& vmap) {
    const auto& match_vmap = match.values_map;
    // 获取匹配中的 linear 值
    auto linear = graph_rewrite_helper::getValue("linear", match_vmap, vmap);
    // 获取 linear 函数的名称
    auto func_name = graph_rewrite_helper::getFuncName(linear);
    // 返回是否匹配 linear 函数名为 "linear"
    return func_name == "linear";
  };
  
  // 创建子图重写器对象
  SubgraphRewriter rewriter;
  // 注册重写模式，将 functional_linear 转换为 aten_linear
  rewriter.RegisterRewritePattern(functional_linear, aten_linear);
  // 在给定的图上运行重写器，使用 filter 进行匹配过滤
  rewriter.runOnGraph(graph, filter);
}

} // namespace jit
} // namespace torch
```