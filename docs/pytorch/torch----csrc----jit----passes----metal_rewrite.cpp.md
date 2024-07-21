# `.\pytorch\torch\csrc\jit\passes\metal_rewrite.cpp`

```py
// 包含 ATen 库的头文件，定义了 JIT 类型
#include <ATen/core/jit_type.h>
// 包含 c10 实用工具中的整数范围处理
#include <c10/util/irange.h>

// 包含 Torch JIT 中的 IR 相关头文件
#include <torch/csrc/jit/ir/ir.h>
// 包含 Torch JIT 中的子图匹配器
#include <torch/csrc/jit/ir/subgraph_matcher.h>
// 包含 Torch JIT 中的常量池优化 pass
#include <torch/csrc/jit/passes/constant_pooling.h>
// 包含 Torch JIT 中的卷积与批归一化融合 pass
#include <torch/csrc/jit/passes/fold_conv_bn.h>
// 包含 Torch JIT 中的模块冻结 pass
#include <torch/csrc/jit/passes/freeze_module.h>
// 包含 Torch JIT 中的线性层融合 pass
#include <torch/csrc/jit/passes/fuse_linear.h>
// 包含 Torch JIT 中的图重写辅助函数
#include <torch/csrc/jit/passes/graph_rewrite_helper.h>
// 包含 Torch JIT 中的 Metal 重写 pass
#include <torch/csrc/jit/passes/metal_rewrite.h>
// 包含 Torch JIT 中的预打包折叠 pass
#include <torch/csrc/jit/passes/prepack_folding.h>
// 包含 Torch JIT 中的移除 dropout pass
#include <torch/csrc/jit/passes/remove_dropout.h>
// 包含 Torch JIT 中的移除变异 pass
#include <torch/csrc/jit/passes/remove_mutation.h>
// 包含 Torch JIT 中的子图重写 pass
#include <torch/csrc/jit/passes/subgraph_rewrite.h>
// 包含 Torch JIT 中的图执行器实现
#include <torch/csrc/jit/runtime/graph_executor_impl.h>

// Torch 的命名空间
namespace torch {
// Torch JIT 的命名空间
namespace jit {

// 匿名命名空间，定义了一些局部函数
namespace {

// 在图中插入预打包线性操作
void insertPrePackedLinearOp(std::shared_ptr<Graph>& graph) {
  // 调用函数将分解的线性层融合成 aten::linear
  FuseLinear(graph);

  // 定义用于匹配的线性层模式和预打包操作模式的字符串表示
  std::string linear_pattern = R"(
    graph(%input, %weight, %bias):
        %r = aten::linear(%input, %weight, %bias)
        return (%r))";
  std::string prepacked_ops_pattern = R"(
    graph(%input, %weight, %bias):
        %output_min_max : None = prim::Constant()
        %packed_weight_bias = metal_prepack::linear_prepack(
            %weight, %bias, %output_min_max, %output_min_max)
        %res = metal_prepack::linear_run(%input, %packed_weight_bias)
        return (%res))";

  // 创建子图重写器对象，注册线性层模式重写规则
  SubgraphRewriter linear_rewriter;
  linear_rewriter.RegisterRewritePattern(linear_pattern, prepacked_ops_pattern);
  // 在图上运行子图重写器
  linear_rewriter.runOnGraph(graph);
}

// 在图中插入预打包的二维卷积操作
void insertPrePackedConv2dOp(std::shared_ptr<Graph>& graph) {
  // 使用帮助函数将卷积替换为 aten::conv2d
  graph_rewrite_helper::replaceConvolutionWithAtenConv(graph);

  // 定义用于匹配的卷积模式和预打包卷积操作模式的字符串表示
  std::string conv_2d_pattern = R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[], %dilation:int[], %groups:int):
        %r = aten::conv2d(%input, %weight, %bias, %stride, %padding, %dilation, %groups)
        return (%r) )";

  std::string prepacked_ops_conv2d_pattern = R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[],
          %dilation:int[], %groups:int):
        %output_min_max : None = prim::Constant()
        %packed_weight_bias = metal_prepack::conv2d_prepack(
            %weight, %bias, %stride, %padding, %dilation, %groups,
            %output_min_max, %output_min_max)
        %r = metal_prepack::conv2d_run(%input, %packed_weight_bias)
        return (%r) )";

  // 创建子图重写器对象，注册卷积模式重写规则
  SubgraphRewriter rewriter;
  rewriter.RegisterRewritePattern(
      conv_2d_pattern, prepacked_ops_conv2d_pattern);
  // 在图上运行子图重写器
  rewriter.runOnGraph(graph);
}

// 融合带有预打包操作的 ReLU 操作
void fuseReluWithPackedOps(std::shared_ptr<Graph>& graph) {
  // 创建子图重写器对象
  SubgraphRewriter rewriter;

  // 定义用于匹配的融合后的线性层与预打包操作的模式字符串表示
  std::string linear_prepack_run_relu_fused = R"(
    graph(%input, %weight, %bias):
        %output_min_max : None = prim::Constant()
        %packed_weight_bias = metal_prepack::linear_prepack(
            %weight, %bias, %output_min_max, %output_min_max)
        %r = metal_prepack::linear_run(%input, %packed_weight_bias)
        %res = aten::relu(%r)
        return (%res))";
  graph(%input, %weight, %bias, %dummy_min_max):
      %output_min: float = prim::Constant[value=0.0]()
      %output_max: None = prim::Constant()
      # 使用 Metal 预打包操作将线性层的权重和偏置打包为一个上下文对象
      %packed_weight_bias : __torch__.torch.classes.metal.LinearOpContext = metal_prepack::linear_prepack(
          %weight, %bias, %output_min, %output_max)
      # 使用 Metal 执行线性层的计算
      %res = metal_prepack::linear_run(%input, %packed_weight_bias)
      return (%res))";

rewriter.RegisterRewritePattern(
    linear_prepack_run_relu, linear_prepack_run_relu_fused);

std::string conv2d_prepack_run_relu = R"(
  graph(%input, %weight, %bias, %stride:int[], %padding:int[],
        %dilation:int[], %groups:int, %dummy_min_max):
      # 使用 Metal 预打包操作将卷积层的权重和偏置打包为一个上下文对象
      %packed_weight_bias = metal_prepack::conv2d_prepack(
          %weight, %bias, %stride, %padding, %dilation, %groups,
          %dummy_min_max, %dummy_min_max)
      # 使用 Metal 执行卷积层的计算
      %r = metal_prepack::conv2d_run(%input, %packed_weight_bias)
      # 对卷积结果应用 ReLU 激活函数
      %r = aten::relu(%r)
      return (%r) )";

std::string conv2d_prepack_run_relu_fused = R"(
graph(%input, %weight, %bias, %stride:int[], %padding:int[],
      %dilation:int[], %groups:int, %dummy_min_max):
    %output_min: float = prim::Constant[value=0.0]()
    %output_max: None = prim::Constant()
    # 使用 Metal 预打包操作将卷积层的权重和偏置打包为一个上下文对象
    %packed_weight_bias: __torch__.torch.classes.metal.Conv2dOpContext = metal_prepack::conv2d_prepack(
        %weight, %bias, %stride, %padding, %dilation, %groups,
        %output_min, %output_max)
    # 使用 Metal 执行卷积层的计算
    %r = metal_prepack::conv2d_run(%input, %packed_weight_bias)
    return (%r) )";

rewriter.RegisterRewritePattern(
    conv2d_prepack_run_relu, conv2d_prepack_run_relu_fused);

std::string linear_prepack_run_relu_inplace = R"(
    // 定义一个名为 `linear_prepack_run_relu_inplace` 的字符串，其包含了一个 TorchScript 图形（graph）的描述。
    std::string linear_prepack_run_relu_inplace = R"(
      graph(%input, %weight, %bias, %dummy_min_max):
          // 调用 Metal 预打包函数 `linear_prepack`，用给定的权重、偏置和虚拟的最小最大值进行预打包
          %packed_weight_bias = metal_prepack::linear_prepack(
              %weight, %bias, %dummy_min_max, %dummy_min_max)
          // 调用 Metal 运行函数 `linear_run` 执行线性运算，并将结果保存在 `%linear_res` 中
          %linear_res = metal_prepack::linear_run(%input, %packed_weight_bias)
          // 调用 PyTorch 的原地 ReLU 激活函数 `aten::relu_`，并将结果保存在 `%res` 中
          %res = aten::relu_(%linear_res)
          // 返回计算得到的结果 `%res`
          return (%res))";
    
    // 定义一个名为 `conv2d_prepack_run_relu_inplace` 的字符串，其包含了一个 TorchScript 图形（graph）的描述。
    std::string conv2d_prepack_run_relu_inplace = R"(
      graph(%input, %weight, %bias, %stride:int[], %padding:int[],
            %dilation:int[], %groups:int, %dummy_min_max):
          // 调用 Metal 预打包函数 `conv2d_prepack`，用给定的权重、偏置、步长、填充、膨胀、组数和虚拟的最小最大值进行预打包
          %packed_weight_bias = metal_prepack::conv2d_prepack(
              %weight, %bias, %stride, %padding, %dilation, %groups,
              %dummy_min_max, %dummy_min_max)
          // 调用 Metal 运行函数 `conv2d_run` 执行卷积运算，并将结果保存在 `%r` 中
          %r = metal_prepack::conv2d_run(%input, %packed_weight_bias)
          // 调用 PyTorch 的原地 ReLU 激活函数 `aten::relu_`，并将结果保存在 `%r` 中（覆盖之前的 `%r`）
          %r = aten::relu_(%r)
          // 返回计算得到的结果 `%r`
          return (%r) )";
    
    // 将 `linear_prepack_run_relu_inplace` 的图形模式注册到重写器中，指定它将被重写为 `linear_prepack_run_relu_fused`
    rewriter.RegisterRewritePattern(
        linear_prepack_run_relu_inplace, linear_prepack_run_relu_fused);
    
    // 将 `conv2d_prepack_run_relu_inplace` 的图形模式注册到重写器中，指定它将被重写为 `conv2d_prepack_run_relu_fused`
    rewriter.RegisterRewritePattern(
        conv2d_prepack_run_relu_inplace, conv2d_prepack_run_relu_fused);
    
    // 运行重写器，对给定的图形 `graph` 进行重写，使用 Torch 的图形重写辅助函数 `torch::jit::graph_rewrite_helper::isClampFusable` 进行判断是否可以融合
    rewriter.runOnGraph(graph, torch::jit::graph_rewrite_helper::isClampFusable);
void fuseHardtanhWithPackedOps(std::shared_ptr<Graph>& graph) {
  SubgraphRewriter rewriter;

  // 定义线性层预打包运行与硬切线融合的图表达式字符串
  std::string linear_prepack_run_hardtanh_fused = R"(
    graph(%input, %weight, %bias, %output_min, %output_max, %dummy_min_max):
        %packed_weight_bias : __torch__.torch.classes.metal.LinearOpContext = metal_prepack::linear_prepack(%weight, %bias, %output_min, %output_max)
        %res = metal_prepack::linear_run(%input, %packed_weight_bias)
        return (%res))";

  // 定义线性层预打包运行与硬切线的图表达式字符串
  std::string linear_prepack_run_hardtanh = R"(
    graph(%input, %weight, %bias, %output_min, %output_max, %dummy_min_max):
        %packed_weight_bias = metal_prepack::linear_prepack(
            %weight, %bias, %dummy_min_max, %dummy_min_max)
        %linear_res = metal_prepack::linear_run(%input, %packed_weight_bias)
        %res = aten::hardtanh(%linear_res, %output_min, %output_max)
        return (%res))";

  // 注册重写模式，将线性层预打包运行与硬切线的图表达式替换为融合版本
  rewriter.RegisterRewritePattern(
      linear_prepack_run_hardtanh, linear_prepack_run_hardtanh_fused);

  // 定义二维卷积层预打包运行与硬切线融合的图表达式字符串
  std::string conv2d_prepack_run_hardtanh_fused = R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[],
          %dilation:int[], %groups:int, %output_min, %output_max, %dummy_min_max):
        %packed_weight_bias: __torch__.torch.classes.metal.Conv2dOpContext = metal_prepack::conv2d_prepack(
            %weight, %bias, %stride, %padding, %dilation, %groups,
            %output_min, %output_max)
        %r = metal_prepack::conv2d_run(%input, %packed_weight_bias)
        return (%r) )";

  // 定义二维卷积层预打包运行与硬切线的图表达式字符串
  std::string conv2d_prepack_run_hardtanh = R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[],
          %dilation:int[], %groups:int, %output_min, %output_max, %dummy_min_max):
        %packed_weight_bias = metal_prepack::conv2d_prepack(
            %weight, %bias, %stride, %padding, %dilation, %groups,
            %dummy_min_max, %dummy_min_max)
        %r = metal_prepack::conv2d_run(%input, %packed_weight_bias)
        %r = aten::hardtanh(%r, %output_min, %output_max)
        return (%r) )";

  // 注册重写模式，将二维卷积层预打包运行与硬切线的图表达式替换为融合版本
  rewriter.RegisterRewritePattern(
      conv2d_prepack_run_hardtanh, conv2d_prepack_run_hardtanh_fused);

  // 定义带就地操作的二维卷积层预打包运行与硬切线的图表达式字符串
  std::string conv2d_prepack_run_hardtanh_inplace = R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[],
          %dilation:int[], %groups:int, %output_min, %output_max, %dummy_min_max):
        %packed_weight_bias = metal_prepack::conv2d_prepack(
            %weight, %bias, %stride, %padding, %dilation, %groups,
            %dummy_min_max, %dummy_min_max)
        %r = metal_prepack::conv2d_run(%input, %packed_weight_bias)
        %r = aten::hardtanh_(%r, %output_min, %output_max)
        return (%r) )";

  // 定义带就地操作的线性层预打包运行与硬切线的图表达式字符串
  std::string linear_prepack_run_hardtanh_inplace = R"(
    // 定义一个名为 graph 的函数，接受多个参数：输入数据、权重、偏置、输出最小值、输出最大值以及虚拟的最小和最大值
    graph(%input, %weight, %bias, %output_min, %output_max, %dummy_min_max):
        // 使用 metal_prepack::linear_prepack 对权重和偏置进行预打包，传入虚拟的最小和最大值
        %packed_weight_bias = metal_prepack::linear_prepack(
            %weight, %bias, %dummy_min_max, %dummy_min_max)
        // 使用 metal_prepack::linear_run 执行线性计算，传入输入数据和预打包的权重和偏置
        %linear_res = metal_prepack::linear_run(%input, %packed_weight_bias)
        // 使用 aten::hardtanh_ 对线性计算结果进行硬性限幅，传入输出最小值和最大值
        %res = aten::hardtanh_(%linear_res, %output_min, %output_max)
        // 返回硬性限幅后的结果
        return (%res));

  // 注册重写模式，将 linear_prepack_run_hardtanh_inplace 重写为 linear_prepack_run_hardtanh_fused
  rewriter.RegisterRewritePattern(
      linear_prepack_run_hardtanh_inplace, linear_prepack_run_hardtanh_fused);

  // 注册重写模式，将 conv2d_prepack_run_hardtanh_inplace 重写为 conv2d_prepack_run_hardtanh_fused
  rewriter.RegisterRewritePattern(
      conv2d_prepack_run_hardtanh_inplace, conv2d_prepack_run_hardtanh_fused);

  // 运行重写器，对 graph 进行重写操作，使用 torch::jit::graph_rewrite_helper::isClampFusable 作为条件判断
  rewriter.runOnGraph(graph, torch::jit::graph_rewrite_helper::isClampFusable);
} // 结束命名空间 torch

} // 结束命名空间 jit

// 向图中插入预打包操作
void metalInsertPrePackedOps(std::shared_ptr<Graph>& graph) {
  // 插入预打包的线性操作
  insertPrePackedLinearOp(graph);
  // 插入预打包的二维卷积操作
  insertPrePackedConv2dOp(graph);
}

// 向模块中插入预打包操作
void metalInsertPrePackedOps(script::Module& module) {
  // 遍历模块中的每个方法
  for (auto& method : module.get_methods()) {
    auto graph = method.graph();
    // 向方法的图中插入预打包操作
    metalInsertPrePackedOps(graph);
  }
  // 遍历模块的每个子模块
  for (script::Module m : module.children()) {
    // 递归地向子模块中插入预打包操作
    metalInsertPrePackedOps(m);
  }
}

// 折叠预打包操作
void metalFoldPrePackingOps(script::Module& m) {
  // 定义过滤函数，过滤出特定类型的节点
  PrePackingOpsFilterFn filter_fn = [](const Node* n) -> bool {
    return (
        (n->kind() ==
         Symbol::fromQualString("metal_prepack::conv2d_prepack")) ||
        (n->kind() == Symbol::fromQualString("metal_prepack::linear_prepack")));
  };
  // 对模块应用预打包操作的折叠
  PrePackingOpsFolder(m, filter_fn, "prepack_folding");
}

// 将预打包的卷积操作与 Clamp 融合
void metalFusePrePackedConvWithClamp(script::Module& module) {
  // 获取模块中 forward 方法的图
  auto graph = module.get_method("forward").graph();
  // 将 Relu 与预打包操作融合
  fuseReluWithPackedOps(graph);
  // 将 Hardtanh 与预打包操作融合
  fuseHardtanhWithPackedOps(graph);
}

// 移除变异操作
static void metalRemoveMutation(script::Module& module) {
  // 获取模块中 forward 方法的图
  auto graph = module.get_method("forward").graph();
  // 移除张量变异操作
  RemoveTensorMutation(graph);
}

// 运行标准优化操作
static void metalRunCanonicalOptimizations(script::Module& module) {
  // 获取模块中 forward 方法的图
  auto graph = module.get_method("forward").graph();
  // 运行优化，不进行循环展开
  runOptimization(graph, false /* no loop unrolling */);
}

// 为移动端优化模块
script::Module metalOptimizeForMobile(
    const script::Module& m,
    const std::vector<std::string>& preserved_methods) {
  // 克隆输入模块
  auto cloned_module = m.clone();
  // 设置克隆模块为评估模式
  cloned_module.eval();
  // 对模块进行卷积批归一化折叠
  cloned_module = FoldConvBatchNorm(cloned_module);
  // 向模块中插入预打包操作
  metalInsertPrePackedOps(cloned_module);
  // 冻结模块，并保留指定方法
  cloned_module = freeze_module(cloned_module, preserved_methods);
  // 将预打包的卷积操作与 Clamp 融合
  metalFusePrePackedConvWithClamp(cloned_module);
  // 折叠预打包操作
  metalFoldPrePackingOps(cloned_module);
  // 移除 Dropout 操作
  removeDropout(cloned_module);
  // 移除重复的常量
  metalRunCanonicalOptimizations(cloned_module);
  // 注册优化后的属性到模块
  cloned_module.register_attribute(
      "optimized_for_metal", BoolType::get(), true);
  // 返回优化后的模块
  return cloned_module;
}
```