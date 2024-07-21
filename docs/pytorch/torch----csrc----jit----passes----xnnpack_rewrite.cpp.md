# `.\pytorch\torch\csrc\jit\passes\xnnpack_rewrite.cpp`

```
// 包含头文件：ATen库的jit_type.h和native/xnnpack/OpContext.h
#include <ATen/core/jit_type.h>
#include <ATen/native/xnnpack/OpContext.h>

// 包含头文件：Torch库的ir.h、passes中的几个优化pass和runtime中的graph_executor_impl.h
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/fold_conv_bn.h>
#include <torch/csrc/jit/passes/freeze_module.h>
#include <torch/csrc/jit/passes/fuse_linear.h>
#include <torch/csrc/jit/passes/fuse_relu.h>
#include <torch/csrc/jit/passes/graph_rewrite_helper.h>
#include <torch/csrc/jit/passes/hoist_conv_packed_params.h>
#include <torch/csrc/jit/passes/mobile_optimizer_type.h>
#include <torch/csrc/jit/passes/prepack_folding.h>
#include <torch/csrc/jit/passes/remove_dropout.h>
#include <torch/csrc/jit/passes/subgraph_rewrite.h>
#include <torch/csrc/jit/passes/xnnpack_rewrite.h>
#include <torch/csrc/jit/runtime/graph_executor_impl.h>

// 命名空间声明：torch::jit
namespace torch {
namespace jit {

// 匿名命名空间，用于定义局部函数和变量
namespace {

// 将conv1d替换为conv2d的函数实现
void replaceConv1dWithConv2d(std::shared_ptr<Graph>& graph) {
  // 定义conv1d模式字符串
  std::string conv_1d_pattern = R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[], %dilation:int[], %groups:int):
        %res = aten::conv1d(%input, %weight, %bias, %stride, %padding, %dilation, %groups)
        return (%res) )";

  // 定义conv2d模式字符串
  std::string conv_2d_pattern = R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[], %dilation:int[], %groups:int):
        %zero : int = prim::Constant[value=0]()
        %one : int = prim::Constant[value=1]()
        %stride_w : int = prim::ListUnpack(%stride)
        %stride_2d : int[] = prim::ListConstruct(%one, %stride_w)
        %padding_w : int = prim::ListUnpack(%padding)
        %padding_2d : int[] = prim::ListConstruct(%zero, %padding_w)
        %dilation_w : int = prim::ListUnpack(%dilation)
        %dilation_2d : int[] = prim::ListConstruct(%one, %dilation_w)
        %two : int = prim::Constant[value=2]()
        %input_2d : Tensor = aten::unsqueeze(%input, %two)
        %weight_2d : Tensor = aten::unsqueeze(%weight, %two)
        %output_2d = aten::conv2d(
            %input_2d, %weight_2d, %bias, %stride_2d, %padding_2d, %dilation_2d, %groups)
        %output : Tensor = aten::squeeze(%output_2d, %two)
        return (%output) )";

  // 定义值映射，用于将conv1d模式中的符号映射到conv2d模式中的符号
  std::vector<std::pair<std::string, std::string>> value_mappings(
      {{"zero", "res"},
       {"one", "res"},
       {"stride_w", "res"},
       {"stride_2d", "res"},
       {"padding_w", "res"},
       {"padding_2d", "res"},
       {"dilation_w", "res"},
       {"dilation_2d", "res"},
       {"two", "res"},
       {"input_2d", "res"},
       {"weight_2d", "res"},
       {"output_2d", "res"},
       {"output", "res"}});

  // 创建子图重写器对象
  SubgraphRewriter rewriter;
  // 注册重写模式，用conv1d模式替换为conv2d模式，并使用值映射
  rewriter.RegisterRewritePattern(
      conv_1d_pattern, conv_2d_pattern, value_mappings);
  // 在图上运行重写器
  rewriter.runOnGraph(graph);
}

} // namespace

// 将conv1d转换为conv2d的函数，对输入的图进行操作
void transformConv1dToConv2d(std::shared_ptr<Graph>& graph) {
  // 使用helper函数将_convolution替换为aten::conv
  graph_rewrite_helper::replaceConvolutionWithAtenConv(graph);
  // 调用上面定义的函数，将conv1d替换为conv2d
  replaceConv1dWithConv2d(graph);
}

} // namespace jit
} // namespace torch
// 将一维卷积转换为二维卷积，遍历模块中的每个方法
void transformConv1dToConv2d(script::Module& module) {
  // 遍历模块中的每个方法
  for (auto& method : module.get_methods()) {
    // 获取方法的计算图
    auto graph = method.graph();
    // 在计算图中执行一维到二维卷积的转换
    transformConv1dToConv2d(graph);
  }
  // 遍历模块的子模块
  for (script::Module m : module.children()) {
    // 递归调用，将子模块中的一维卷积转换为二维卷积
    transformConv1dToConv2d(m);
  }
}

// 如果使用了 XNNPACK 库，则在匿名命名空间中定义以下函数
#ifdef USE_XNNPACK
namespace {

// 在计算图中插入预打包的线性操作
void insertPrePackedLinearOp(std::shared_ptr<Graph>& graph) {
  // 将分解的线性操作融合成 aten::linear
  FuseLinear(graph);

  // 定义线性模式和预打包操作模式的字符串表示
  std::string linear_pattern = R"(
    graph(%input, %weight, %bias):
        %res = aten::linear(%input, %weight, %bias)
        return (%res))";
  std::string prepacked_ops_pattern = R"(
    graph(%input, %weight, %bias):
        %output_min_max : None = prim::Constant()
        %packed_weight_bias = prepacked::linear_clamp_prepack(
            %weight, %bias, %output_min_max, %output_min_max)
        %res = prepacked::linear_clamp_run(%input, %packed_weight_bias)
        return (%res))";

  // 值映射关系，将输出映射到对应的变量
  std::vector<std::pair<std::string, std::string>> value_mappings(
      {{"output_min_max", "res"},
       {"packed_weight_bias", "res"},
       {"res", "res"}});

  // 创建子图重写器对象
  SubgraphRewriter linear_rewriter;
  // 注册线性模式重写规则
  linear_rewriter.RegisterRewritePattern(
      linear_pattern, prepacked_ops_pattern, value_mappings);
  // 在计算图上执行重写
  linear_rewriter.runOnGraph(graph);
}

// 在计算图中插入预打包的二维卷积操作
void insertPrePackedConv2dOp(std::shared_ptr<Graph>& graph) {
  // 替换 _convolution 为 conv2d
  graph_rewrite_helper::replaceConvolutionWithAtenConv(graph);

  // 定义二维卷积模式和预打包操作模式的字符串表示
  std::string conv_2d_pattern = R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[], %dilation:int[], %groups:int):
        %res = aten::conv2d(%input, %weight, %bias, %stride, %padding, %dilation, %groups)
        return (%res) )";

  std::string prepacked_ops_conv2d_pattern = R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[], %dilation:int[], %groups:int):
        %output_min_max : None = prim::Constant()
        %packed_weight_bias = prepacked::conv2d_clamp_prepack(
            %weight, %bias, %stride, %padding, %dilation, %groups,
            %output_min_max, %output_min_max)
        %res = prepacked::conv2d_clamp_run(%input, %packed_weight_bias)
        return (%res) )";

  // 值映射关系，将输出映射到对应的变量
  std::vector<std::pair<std::string, std::string>> value_mappings(
      {{"output_min_max", "res"},
       {"packed_weight_bias", "res"},
       {"res", "res"}});

  // 创建子图重写器对象
  SubgraphRewriter rewriter;
  // 注册二维卷积模式重写规则
  rewriter.RegisterRewritePattern(
      conv_2d_pattern, prepacked_ops_conv2d_pattern, value_mappings);
  // 在计算图上执行重写
  rewriter.runOnGraph(graph);

  // 定义转置二维卷积模式和预打包操作模式的字符串表示
  std::string conv_2d_transpose_pattern = R"(
      graph(%input, %weight, %bias, %stride:int[], %padding:int[], %dilation:int[],
          %output_padding:int[], %groups:int):
        %res = aten::conv_transpose2d(%input, %weight, %bias, %stride, %padding, %output_padding, %groups, %dilation)
        return (%res) )";

  std::string prepacked_ops_conv2d_transpose_pattern = R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[], %dilation:int[], %output_padding:int[], %groups:int):
        %output_min_max : None = prim::Constant()
        %packed_weight_bias = prepacked::conv2d_transpose_clamp_prepack(
            %weight, %bias, %stride, %padding, %output_padding, %dilation, %groups,
            %output_min_max, %output_min_max)
        %res = prepacked::conv2d_transpose_clamp_run(%input, %packed_weight_bias)
        return (%res) )";


# 定义一个名为 graph 的函数，参数包括输入、权重、偏置、步长、填充、膨胀、输出填充和分组数
graph(%input, %weight, %bias, %stride:int[], %padding:int[], %dilation:int[], %output_padding:int[], %groups:int):
    # 创建一个空的常量节点 output_min_max
    %output_min_max : None = prim::Constant()
    # 使用预装包的函数 prepacked::conv2d_transpose_clamp_prepack 对权重和偏置进行预装包处理
    %packed_weight_bias = prepacked::conv2d_transpose_clamp_prepack(
        %weight, %bias, %stride, %padding, %output_padding, %dilation, %groups,
        %output_min_max, %output_min_max)
    # 使用预装包的函数 prepacked::conv2d_transpose_clamp_run 运行卷积转置操作
    %res = prepacked::conv2d_transpose_clamp_run(%input, %packed_weight_bias)
    # 返回结果 %res
    return (%res) )";



  value_mappings = {
      {"output_min_max", "res"}, {"packed_weight_bias", "res"}, {"res", "res"}};


# 创建一个名为 value_mappings 的字典，用于映射替换规则
value_mappings = {
    {"output_min_max", "res"}, {"packed_weight_bias", "res"}, {"res", "res"}};



  SubgraphRewriter transpose_rewriter;
  transpose_rewriter.RegisterRewritePattern(
      conv_2d_transpose_pattern,
      prepacked_ops_conv2d_transpose_pattern,
      value_mappings);
  transpose_rewriter.runOnGraph(graph);


# 创建一个名为 transpose_rewriter 的子图重写器对象
SubgraphRewriter transpose_rewriter;
# 在 transpose_rewriter 上注册重写模式，将 conv_2d_transpose_pattern 替换为 prepacked_ops_conv2d_transpose_pattern，并使用 value_mappings 进行映射
transpose_rewriter.RegisterRewritePattern(
    conv_2d_transpose_pattern,
    prepacked_ops_conv2d_transpose_pattern,
    value_mappings);
# 在图 graph 上运行 transpose_rewriter 进行重写操作
transpose_rewriter.runOnGraph(graph);
void fuseHardtanhWithPackedOps(std::shared_ptr<Graph>& graph) {
  // 创建一个子图重写器对象
  SubgraphRewriter rewriter;

  // 定义线性层预打包运行与硬切线性整流（Hardtanh）融合的字符串表示
  std::string linear_prepack_run_hardtanh_fused = R"(
    graph(%input, %weight, %bias, %output_min, %output_max, %dummy_min_max):
        %packed_weight_bias : __torch__.torch.classes.xnnpack.LinearOpContext = prepacked::linear_clamp_prepack(
            %weight, %bias, %output_min, %output_max)
        %res = prepacked::linear_clamp_run(%input, %packed_weight_bias)
        return (%res))";

  // 定义二维卷积层预打包运行与硬切线性整流（Hardtanh）融合的字符串表示
  std::string conv2d_prepack_run_hardtanh_fused = R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[],
          %dilation:int[], %groups:int, %output_min, %output_max, %dummy_min_max):
        %packed_weight_bias : __torch__.torch.classes.xnnpack.Conv2dOpContext = prepacked::conv2d_clamp_prepack(
            %weight, %bias, %stride, %padding, %dilation, %groups,
            %output_min, %output_max)
        %res = prepacked::conv2d_clamp_run(%input, %packed_weight_bias)
        return (%res) )";

  // 定义线性层预打包运行与硬切线性整流（Hardtanh）融合（原位操作）的字符串表示
  std::string linear_prepack_run_hardtanh = R"(
    graph(%input, %weight, %bias, %output_min, %output_max, %dummy_min_max):
        %packed_weight_bias = prepacked::linear_clamp_prepack(
            %weight, %bias, %dummy_min_max, %dummy_min_max)
        %linear_res = prepacked::linear_clamp_run(%input, %packed_weight_bias)
        %res = aten::hardtanh(%linear_res, %output_min, %output_max)
        return (%res))";

  // 创建一个值映射的向量，用于注册线性层预打包运行与硬切线性整流（Hardtanh）融合的重写模式
  std::vector<std::pair<std::string, std::string>> value_mappings(
      {{"packed_weight_bias", "packed_weight_bias"}, {"res", "res"}});

  // 注册线性层预打包运行与硬切线性整流（Hardtanh）融合的重写模式
  rewriter.RegisterRewritePattern(
      linear_prepack_run_hardtanh,
      linear_prepack_run_hardtanh_fused,
      value_mappings);

  // 定义二维卷积层预打包运行与硬切线性整流（Hardtanh）融合（原位操作）的字符串表示
  std::string conv2d_prepack_run_hardtanh = R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[],
          %dilation:int[], %groups:int, %output_min, %output_max, %dummy_min_max):
        %packed_weight_bias = prepacked::conv2d_clamp_prepack(
            %weight, %bias, %stride, %padding, %dilation, %groups,
            %dummy_min_max, %dummy_min_max)
        %conv2d_res = prepacked::conv2d_clamp_run(%input, %packed_weight_bias)
        %res = aten::hardtanh(%conv2d_res, %output_min, %output_max)
        return (%res) )";

  // 更新值映射的向量，用于注册二维卷积层预打包运行与硬切线性整流（Hardtanh）融合的重写模式
  value_mappings = {
      {"packed_weight_bias", "packed_weight_bias"}, {"res", "res"}};

  // 注册二维卷积层预打包运行与硬切线性整流（Hardtanh）融合的重写模式
  rewriter.RegisterRewritePattern(
      conv2d_prepack_run_hardtanh,
      conv2d_prepack_run_hardtanh_fused,
      value_mappings);

  // 定义线性层预打包运行与硬切线性整流（Hardtanh）原位操作的字符串表示
  std::string linear_prepack_run_hardtanh_inplace = R"(
    graph(%input, %weight, %bias, %output_min, %output_max, %dummy_min_max):
        %packed_weight_bias = prepacked::linear_clamp_prepack(
            %weight, %bias, %dummy_min_max, %dummy_min_max)
        %linear_res = prepacked::linear_clamp_run(%input, %packed_weight_bias)
        %res = aten::hardtanh_(%linear_res, %output_min, %output_max)
        return (%res))";

  // 定义二维卷积层预打包运行与硬切线性整流（Hardtanh）原位操作的字符串表示
  std::string conv2d_prepack_run_hardtanh_inplace = R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[],
          %dilation:int[], %groups:int, %output_min, %output_max, %dummy_min_max):
        %packed_weight_bias = prepacked::conv2d_clamp_prepack(
            %weight, %bias, %stride, %padding, %dilation, %groups,
            %dummy_min_max, %dummy_min_max)
        %conv2d_res = prepacked::conv2d_clamp_run(%input, %packed_weight_bias)
        %res = aten::hardtanh_(%conv2d_res, %output_min, %output_max)
        return (%res) )";

  // }
    def graph(%input, %weight, %bias, %stride:int[], %padding:int[],
              %dilation:int[], %groups:int, %output_min, %output_max, %dummy_min_max):
        # 调用预打包函数，将权重、偏置等参数预打包成一个对象
        %packed_weight_bias = prepacked::conv2d_clamp_prepack(
            %weight, %bias, %stride, %padding, %dilation, %groups,
            %dummy_min_max, %dummy_min_max)
        # 执行预打包后的卷积操作，返回卷积结果
        %conv2d_res = prepacked::conv2d_clamp_run(%input, %packed_weight_bias)
        # 对卷积结果应用硬切函数，限制在指定的输出最小和最大值范围内
        %res = aten::hardtanh_(%conv2d_res, %output_min, %output_max)
        # 返回处理后的结果
        return (%res) )";
    
    # 创建值映射字典，用于重写模式注册
    value_mappings = {
        {"packed_weight_bias", "packed_weight_bias"}, {"res", "res"}};
    
    # 注册重写模式，将特定模式的图模式从 linear_prepack_run_hardtanh_inplace 重写到 linear_prepack_run_hardtanh_fused
    rewriter.RegisterRewritePattern(
        linear_prepack_run_hardtanh_inplace,
        linear_prepack_run_hardtanh_fused,
        value_mappings);
    
    # 重新创建值映射字典，用于另一个重写模式的注册
    value_mappings = {
        {"packed_weight_bias", "packed_weight_bias"}, {"res", "res"}};
    
    # 注册重写模式，将特定模式的图模式从 conv2d_prepack_run_hardtanh_inplace 重写到 conv2d_prepack_run_hardtanh_fused
    rewriter.RegisterRewritePattern(
        conv2d_prepack_run_hardtanh_inplace,
        conv2d_prepack_run_hardtanh_fused,
        value_mappings);
    
    # 运行重写器，应用于给定的图，使用 Torch 的辅助函数来判断是否可以融合 Clamp 操作
    rewriter.runOnGraph(graph, torch::jit::graph_rewrite_helper::isClampFusable);
// 定义一个函数，用于将线性操作与ReLU激活函数融合
void fuseReluWithPackedOps(std::shared_ptr<Graph>& graph) {
  // 创建子图重写器对象
  SubgraphRewriter rewriter;

  // 定义线性预打包运行并与ReLU融合的图表达式
  std::string linear_prepack_run_relu_fused = R"(
    graph(%input, %weight, %bias, %dummy_min_max):
        %output_min: float = prim::Constant[value=0.0]()
        %output_max: None = prim::Constant()
        %packed_weight_bias : __torch__.torch.classes.xnnpack.LinearOpContext = prepacked::linear_clamp_prepack(
            %weight, %bias, %output_min, %output_max)
        %res = prepacked::linear_clamp_run(%input, %packed_weight_bias)
        return (%res))";

  // 定义卷积预打包运行并与ReLU融合的图表达式
  std::string conv2d_prepack_run_relu_fused = R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[],
          %dilation:int[], %groups:int, %dummy_min_max):
        %output_min: float = prim::Constant[value=0.0]()
        %output_max: None = prim::Constant()
        %packed_weight_bias : __torch__.torch.classes.xnnpack.Conv2dOpContext = prepacked::conv2d_clamp_prepack(
            %weight, %bias, %stride, %padding, %dilation, %groups,
            %output_min, %output_max)
        %res = prepacked::conv2d_clamp_run(%input, %packed_weight_bias)
        return (%res) )";

  // 定义线性预打包运行并在原地应用ReLU的图表达式
  std::string linear_prepack_run_relu = R"(
    graph(%input, %weight, %bias, %dummy_min_max):
        %packed_weight_bias = prepacked::linear_clamp_prepack(
            %weight, %bias, %dummy_min_max, %dummy_min_max)
        %linear_res = prepacked::linear_clamp_run(%input, %packed_weight_bias)
        %res = aten::relu(%linear_res)
        return (%res))";

  // 创建值映射关系，将变量名映射到替换的变量名
  std::vector<std::pair<std::string, std::string>> value_mappings(
      {{"output_min", "packed_weight_bias"},
       {"output_max", "packed_weight_bias"},
       {"packed_weight_bias", "packed_weight_bias"},
       {"res", "res"}});

  // 注册线性预打包运行并与ReLU融合的重写模式
  rewriter.RegisterRewritePattern(
      linear_prepack_run_relu, linear_prepack_run_relu_fused, value_mappings);

  // 定义卷积预打包运行并在原地应用ReLU的图表达式
  std::string conv2d_prepack_run_relu = R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[],
          %dilation:int[], %groups:int, %dummy_min_max):
        %packed_weight_bias = prepacked::conv2d_clamp_prepack(
            %weight, %bias, %stride, %padding, %dilation, %groups,
            %dummy_min_max, %dummy_min_max)
        %conv2d_res = prepacked::conv2d_clamp_run(%input, %packed_weight_bias)
        %res = aten::relu(%conv2d_res)
        return (%res) )";

  // 更新值映射关系，将变量名映射到替换的变量名
  value_mappings = {
      {"output_min", "packed_weight_bias"},
      {"output_max", "packed_weight_bias"},
      {"packed_weight_bias", "packed_weight_bias"},
      {"res", "res"}};

  // 注册卷积预打包运行并与ReLU融合的重写模式
  rewriter.RegisterRewritePattern(
      conv2d_prepack_run_relu, conv2d_prepack_run_relu_fused, value_mappings);

  // 定义线性预打包运行并在原地应用ReLU的图表达式（原地指不使用额外内存）
  std::string linear_prepack_run_relu_inplace = R"(
  graph(%input, %weight, %bias, %dummy_min_max):
      %packed_weight_bias = prepacked::linear_clamp_prepack(
          %weight, %bias, %dummy_min_max, %dummy_min_max)
      %linear_res = prepacked::linear_clamp_run(%input, %packed_weight_bias)
      %res = aten::relu_(%linear_res)
      return (%res))";


# 定义了一个函数图(graph)，接受输入(%input)，权重(%weight)，偏置(%bias)，以及一个虚拟参数(%dummy_min_max)
graph(%input, %weight, %bias, %dummy_min_max):
    # 调用预打包函数 prepacked::linear_clamp_prepack，将权重和偏置预打包成一个对象
    %packed_weight_bias = prepacked::linear_clamp_prepack(
        %weight, %bias, %dummy_min_max, %dummy_min_max)
    # 使用预打包的权重和偏置进行线性计算
    %linear_res = prepacked::linear_clamp_run(%input, %packed_weight_bias)
    # 对线性计算结果应用 ReLU 激活函数
    %res = aten::relu_(%linear_res)
    # 返回 ReLU 处理后的结果
    return (%res))";



  std::string conv2d_prepack_run_relu_inplace = R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[],
          %dilation:int[], %groups:int, %dummy_min_max):
        %packed_weight_bias = prepacked::conv2d_clamp_prepack(
            %weight, %bias, %stride, %padding, %dilation, %groups,
            %dummy_min_max, %dummy_min_max)
        %conv2d_res = prepacked::conv2d_clamp_run(%input, %packed_weight_bias)
        %res = aten::relu_(%conv2d_res)
        return (%res) )";


# 定义了一个函数图(graph)，接受输入(%input)，权重(%weight)，偏置(%bias)，步幅(%stride)，填充(%padding)，膨胀(%dilation)，分组(%groups)，以及一个虚拟参数(%dummy_min_max)
graph(%input, %weight, %bias, %stride:int[], %padding:int[],
      %dilation:int[], %groups:int, %dummy_min_max):
    # 调用预打包函数 prepacked::conv2d_clamp_prepack，将权重、偏置及卷积参数预打包成一个对象
    %packed_weight_bias = prepacked::conv2d_clamp_prepack(
        %weight, %bias, %stride, %padding, %dilation, %groups,
        %dummy_min_max, %dummy_min_max)
    # 使用预打包的权重、偏置及卷积参数进行卷积计算
    %conv2d_res = prepacked::conv2d_clamp_run(%input, %packed_weight_bias)
    # 对卷积计算结果应用 ReLU 激活函数
    %res = aten::relu_(%conv2d_res)
    # 返回 ReLU 处理后的结果
    return (%res) )";



  value_mappings = {
      {"output_min", "packed_weight_bias"},
      {"output_max", "packed_weight_bias"},
      {"packed_weight_bias", "packed_weight_bias"},
      {"res", "res"}};


# 创建了一个值映射字典，将不同的键映射到相同的值
value_mappings = {
    {"output_min", "packed_weight_bias"},
    {"output_max", "packed_weight_bias"},
    {"packed_weight_bias", "packed_weight_bias"},
    {"res", "res"}};



  rewriter.RegisterRewritePattern(
      linear_prepack_run_relu_inplace,
      linear_prepack_run_relu_fused,
      value_mappings);


# 在重写器(rewriter)中注册重写模式，将 linear_prepack_run_relu_inplace 替换为 linear_prepack_run_relu_fused，并应用值映射
rewriter.RegisterRewritePattern(
    linear_prepack_run_relu_inplace,
    linear_prepack_run_relu_fused,
    value_mappings);



  rewriter.RegisterRewritePattern(
      conv2d_prepack_run_relu_inplace,
      conv2d_prepack_run_relu_fused,
      value_mappings);


# 在重写器(rewriter)中注册重写模式，将 conv2d_prepack_run_relu_inplace 替换为 conv2d_prepack_run_relu_fused，并应用值映射
rewriter.RegisterRewritePattern(
    conv2d_prepack_run_relu_inplace,
    conv2d_prepack_run_relu_fused,
    value_mappings);



  rewriter.runOnGraph(graph, torch::jit::graph_rewrite_helper::isClampFusable);


# 在给定图(graph)上运行重写器(rewriter)，使用 torch::jit::graph_rewrite_helper::isClampFusable 函数进行 Clamp 操作的融合
rewriter.runOnGraph(graph, torch::jit::graph_rewrite_helper::isClampFusable);
}

// 运行规范化优化，对给定模块中的每个方法的计算图进行优化处理
void runCanonicalOptimizations(script::Module& module) {
  // 遍历模块中的每个方法
  for (const auto& method : module.get_methods()) {
    auto graph = method.graph();
    // 检查是否有在移动设备上运行的模型需要展开循环，例如语言/语音模型，这里设置为不展开循环
    runOptimization(graph, false /* no loop unrolling */);
  }
}

// 命名空间结束

// 向图中插入预打包操作
void insertPrePackedOps(std::shared_ptr<Graph>& graph) {
  // 插入预打包的线性操作
  insertPrePackedLinearOp(graph);
  // 插入预打包的二维卷积操作
  insertPrePackedConv2dOp(graph);
}

// 向模块中插入预打包操作
void insertPrePackedOps(script::Module& module) {
  // 遍历模块中的每个方法
  for (auto& method : module.get_methods()) {
    auto graph = method.graph();
    // 向计算图中插入预打包操作
    insertPrePackedOps(graph);
  }
  // 遍历模块的子模块
  for (script::Module m : module.children()) {
    // 向子模块中插入预打包操作
    insertPrePackedOps(m);
  }
}

// 合并预打包的线性和卷积操作与 Clamp 操作
void fusePrePackedLinearConvWithClamp(script::Module& module) {
  // 遍历模块中的每个方法
  for (auto& method : module.get_methods()) {
    auto graph = method.graph();
    // 合并带有 ReLU 的预打包操作
    fuseReluWithPackedOps(graph);
    // 合并带有 Hardtanh 的预打包操作
    fuseHardtanhWithPackedOps(graph);

    // 忽略用户定义的类以便后续处理
    ConstantPropagation(graph, true);
  }
}

// 折叠预打包操作
void FoldPrePackingOps(script::Module& m) {
  // 定义预打包操作的过滤函数
  PrePackingOpsFilterFn filter_fn = [](const Node* n) -> bool {
    return (
        (n->kind() ==
         Symbol::fromQualString("prepacked::linear_clamp_prepack")) ||
        n->kind() ==
            Symbol::fromQualString("prepacked::conv2d_clamp_prepack") ||
        n->kind() ==
            Symbol::fromQualString(
                "prepacked::conv2d_transpose_clamp_prepack"));
  };
  // 折叠满足条件的预打包操作
  PrePackingOpsFolder(m, filter_fn, "prepack_folding");
  // 遍历模块中的每个方法
  for (auto& method : m.get_methods()) {
    auto graph = method.graph();
    // 对用户定义的类进行常量传播以支持折叠操作
    ConstantPropagation(graph, false);
  }
}

// 为移动设备优化模块
script::Module optimizeForMobile(
    const script::Module& m,
    const std::set<MobileOptimizerType>& optimization_blocklist,
    const std::vector<std::string>& preserved_methods) {
  // 克隆模块并进入评估模式
  auto cloned_module = m.clone();
  cloned_module.eval();

  // 如果不在优化阻止列表中，则转换 Conv1D 到 Conv2D
  if (!optimization_blocklist.count(MobileOptimizerType::CONV_1D_TO_2D)) {
    transformConv1dToConv2d(cloned_module);
  }

  // 如果不在优化阻止列表中，则折叠 ConvBatchNorm
  if (!optimization_blocklist.count(MobileOptimizerType::CONV_BN_FUSION)) {
    cloned_module = FoldConvBatchNorm(cloned_module);
  }

  // 许多优化需要一个冻结的模块，但 ConvBatchNorm 需要一个未冻结的模块
  cloned_module = freeze_module(cloned_module, preserved_methods);

  // 如果不在优化阻止列表中，则插入预打包操作并进行后续优化
  if (!optimization_blocklist.count(
          MobileOptimizerType::INSERT_FOLD_PREPACK_OPS)) {
    // 插入预打包操作
    insertPrePackedOps(cloned_module);
    // 再次冻结模块
    cloned_module = freeze_module(cloned_module, preserved_methods);
    // 合并预打包的线性和卷积操作与 Clamp 操作
    fusePrePackedLinearConvWithClamp(cloned_module);
    // 折叠预打包操作
    FoldPrePackingOps(cloned_module);
  }

  // 如果不在优化阻止列表中，则优化 Conv 层的参数提升
  if (!optimization_blocklist.count(
          MobileOptimizerType::HOIST_CONV_PACKED_PARAMS) &&
      cloned_module.find_method("forward")) {
  // 再次冻结模块，以防它在之前的可选步骤中未被冻结
  cloned_module = freeze_module(cloned_module, preserved_methods);
  // 提升压缩参数
  HoistConvPackedParams(cloned_module);
  // 再次冻结以移除空的 QuantizedConv 模块
  cloned_module = freeze_module(cloned_module, preserved_methods);
}

// 运行规范优化的步骤，此时图形已内联，因此不需要显式调用内联传递
runCanonicalOptimizations(cloned_module);

// 如果未在优化阻止列表中，则移除图中的 Dropout 操作
if (!optimization_blocklist.count(MobileOptimizerType::REMOVE_DROPOUT)) {
  for (const auto& method : cloned_module.get_methods()) {
    auto graph = method.graph();
    removeDropout(graph);
  }
}

// 如果未在优化阻止列表中，则在图中融合 Add 和 ReLU 操作
if (!optimization_blocklist.count(MobileOptimizerType::FUSE_ADD_RELU)) {
  for (const auto& method : cloned_module.get_methods()) {
    auto graph = method.graph();
    FuseAddRelu(graph);
  }
}

// 注册属性 "mobile_optimized" 为 true
cloned_module.register_attribute("mobile_optimized", BoolType::get(), true);
// 返回优化后的克隆模块
return cloned_module;
}
// 结束 #ifdef 和 #else 分支之外的代码块

#else

// 向图中插入预打包操作，用于 XNNPACK 未启用时的错误断言
void insertPrePackedOps(std::shared_ptr<Graph>& graph) {
  TORCH_INTERNAL_ASSERT(
      false, "XNNPACK is not enabled. Please build with USE_XNNPACK=1");
}

// 向模块中插入预打包操作，用于 XNNPACK 未启用时的错误断言
void insertPrePackedOps(script::Module& module) {
  TORCH_INTERNAL_ASSERT(
      false, "XNNPACK is not enabled. Please build with USE_XNNPACK=1");
}

// 将预打包的线性卷积与 Clamp 融合，用于 XNNPACK 未启用时的错误断言
void fusePrePackedLinearConvWithClamp(script::Module& module) {
  TORCH_INTERNAL_ASSERT(
      false, "XNNPACK is not enabled. Please build with USE_XNNPACK=1");
}

// 折叠预打包操作，用于 XNNPACK 未启用时的错误断言
void FoldPrePackingOps(script::Module& m) {
  TORCH_INTERNAL_ASSERT(
      false, "XNNPACK is not enabled. Please build with USE_XNNPACK=1");
}

// 优化模块以适应移动设备，用于 XNNPACK 未启用时的错误断言，并返回原始模块
script::Module optimizeForMobile(
    const script::Module& module,
    const std::set<MobileOptimizerType>& blocklist,
    const std::vector<std::string>& preserved_methods) {
  TORCH_INTERNAL_ASSERT(
      false,
      "Mobile optimization only available with XNNPACK at the moment. "
      "XNNPACK is not enabled. Please build with USE_XNNPACK=1");
  return module;
}

#endif
} // namespace jit
} // namespace torch
```