# `.\pytorch\torch\csrc\jit\passes\vulkan_rewrite.cpp`

```
// 引入 ATen 库中的头文件和 Torch JIT 的相关组件
#include <ATen/core/jit_type.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/subgraph_matcher.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/fold_conv_bn.h>
#include <torch/csrc/jit/passes/freeze_module.h>
#include <torch/csrc/jit/passes/fuse_linear.h>
#include <torch/csrc/jit/passes/graph_rewrite_helper.h>
#include <torch/csrc/jit/passes/prepack_folding.h>
#include <torch/csrc/jit/passes/remove_dropout.h>
#include <torch/csrc/jit/passes/remove_mutation.h>
#include <torch/csrc/jit/passes/subgraph_rewrite.h>
#include <torch/csrc/jit/passes/vulkan_rewrite.h>
#include <torch/csrc/jit/runtime/graph_executor_impl.h>

// Torch 的命名空间
namespace torch {
namespace jit {

// 匿名命名空间，用于定义插入预打包操作的函数
namespace {

// 函数：在图中插入预打包的批归一化操作
void insertPrePackedBatchNormOp(std::shared_ptr<Graph>& graph) {
  // 定义批归一化的原始模式字符串
  std::string batchnorm_pattern = R"(
    graph(%input, %weight, %bias, %mean, %var, %training, %momentum, %eps, %cudnn_enable):
        %r = aten::batch_norm(%input, %weight, %bias, %mean, %var, %training, %momentum, %eps, %cudnn_enable)
        return (%r))";
  // 定义预打包操作的模式字符串
  std::string prepacked_ops_pattern = R"(
    graph(%input, %weight, %bias, %mean, %var, %training, %momentum, %eps, %cudnn_enable):
        %op_context : __torch__.torch.classes.vulkan.BatchNormPackedContext = vulkan_prepack::create_batchnorm_context(
            %weight, %bias, %mean, %var, %training, %momentum, %eps, %cudnn_enable)
        %res = vulkan_prepack::run_batchnorm_context(%input, %op_context)
        return (%res))";

  // 创建子图重写器对象
  SubgraphRewriter batchnorm_rewriter;
  // 注册重写模式
  batchnorm_rewriter.RegisterRewritePattern(
      batchnorm_pattern, prepacked_ops_pattern);
  // 在图上运行重写器
  batchnorm_rewriter.runOnGraph(graph);
}

// 函数：在图中插入预打包的线性操作
void insertPrePackedLinearOp(std::shared_ptr<Graph>& graph) {
  // 融合分解后的线性操作为 aten::linear
  FuseLinear(graph);

  // 定义线性操作的原始模式字符串
  std::string linear_pattern = R"(
    graph(%input, %weight, %bias):
        %r = aten::linear(%input, %weight, %bias)
        return (%r))";
  // 定义预打包操作的模式字符串
  std::string prepacked_ops_pattern = R"(
    graph(%input, %weight, %bias):
        %weight_t = aten::t(%weight)
        %packed_weight_bias = vulkan_prepack::create_linear_context(
            %weight_t, %bias)
        %res = vulkan_prepack::run_linear_context(%input, %packed_weight_bias)
        return (%res))";

  // 创建子图重写器对象
  SubgraphRewriter linear_rewriter;
  // 注册重写模式
  linear_rewriter.RegisterRewritePattern(linear_pattern, prepacked_ops_pattern);
  // 在图上运行重写器
  linear_rewriter.runOnGraph(graph);
}

// 函数：在图中插入预打包的层归一化操作
void insertPrePackedLayernormOp(std::shared_ptr<Graph>& graph) {
  // 定义层归一化的原始模式字符串
  std::string layernorm_pattern = R"(
    graph(%input, %normalized_shape, %weight, %bias, %eps, %cudnn_enable):
        %r = aten::layer_norm(%input, %normalized_shape, %weight, %bias, %eps, %cudnn_enable)
        return (%r))";
  // 定义预打包操作的模式字符串，这里留待下文继续添加注释
  std::string prepacked_ops_pattern = R"(
    // 定义一个名为 graph 的函数，接受输入参数 %input, %normalized_shape, %weight, %bias, %eps, %cudnn_enable
    graph(%input, %normalized_shape, %weight, %bias, %eps, %cudnn_enable):
        // 使用 Vulkan 的函数 vulkan_prepack::create_layernorm_context 创建一个 LayernormPackedContext 对象 %op_context，
        // 该对象封装了权重 %weight, 偏置 %bias 和 epsilon 值 %eps
        %op_context : __torch__.torch.classes.vulkan.LayernormPackedContext = vulkan_prepack::create_layernorm_context(
            %weight, %bias, %eps)
        // 使用 Vulkan 的函数 vulkan_prepack::run_layernorm_context 执行 LayernormPackedContext 对象 %op_context，
        // 传入输入 %input 和规范化的形状 %normalized_shape，并将结果保存在 %res 中
        %res = vulkan_prepack::run_layernorm_context(%input, %normalized_shape, %op_context)
        // 返回 %res 变量，该变量包含了经过层归一化处理后的结果
        return (%res))";

  // 创建 SubgraphRewriter 类型的对象 layernorm_rewriter
  SubgraphRewriter layernorm_rewriter;
  // 使用 layernorm_rewriter 对象的 RegisterRewritePattern 方法注册重写模式，
  // 将 layernorm_pattern 替换为 prepacked_ops_pattern
  layernorm_rewriter.RegisterRewritePattern(
      layernorm_pattern, prepacked_ops_pattern);
  // 使用 layernorm_rewriter 对象的 runOnGraph 方法运行重写器，对 graph 进行重写操作
  layernorm_rewriter.runOnGraph(graph);
void insertPrePackedConv2dOp(std::shared_ptr<Graph>& graph) {
  // 替换图中的卷积操作为基于ATen的卷积操作
  graph_rewrite_helper::replaceConvolutionWithAtenConv(graph);

  // 定义原始的conv2d图模式字符串
  std::string conv_2d_pattern = R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[], %dilation:int[], %groups:int):
        %r = aten::conv2d(%input, %weight, %bias, %stride, %padding, %dilation, %groups)
        return (%r) )";

  // 定义预打包的conv2d图模式字符串
  std::string prepacked_ops_conv2d_pattern = R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[], %dilation:int[], %groups:int):
        %output_min_max : None = prim::Constant()
        %packed_weight_bias = vulkan_prepack::create_conv2d_context(
            %weight, %bias, %stride, %padding, %dilation, %groups,
            %output_min_max, %output_min_max)
        %r = vulkan_prepack::run_conv2d_context(%input, %packed_weight_bias)
        return (%r) )";

  // 创建子图重写器对象
  SubgraphRewriter rewriter;
  // 注册卷积操作重写模式
  rewriter.RegisterRewritePattern(
      conv_2d_pattern, prepacked_ops_conv2d_pattern);
  // 在图上运行重写器
  rewriter.runOnGraph(graph);

  // 定义原始的conv_transpose2d图模式字符串
  std::string conv_2d_transpose_pattern = R"(
      graph(%input, %weight, %bias, %stride:int[], %padding:int[], %dilation:int[],
          %output_padding:int[], %groups:int):
        %res = aten::conv_transpose2d(%input, %weight, %bias, %stride, %padding, %output_padding, %groups, %dilation)
        return (%res) )";

  // 定义预打包的conv_transpose2d图模式字符串
  std::string prepacked_ops_conv2d_transpose_pattern = R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[], %dilation:int[], %output_padding:int[], %groups:int):
        %output_min_max : None = prim::Constant()
        %packed_weight_bias = vulkan_prepack::create_tconv2d_context(
            %weight, %bias, %stride, %padding, %output_padding, %dilation, %groups,
            %output_min_max, %output_min_max)
        %res = vulkan_prepack::run_tconv2d_context(%input, %packed_weight_bias)
        return (%res) )";

  // 创建卷积转置操作的子图重写器对象
  SubgraphRewriter transpose_rewriter;
  // 注册卷积转置操作重写模式
  transpose_rewriter.RegisterRewritePattern(
      conv_2d_transpose_pattern, prepacked_ops_conv2d_transpose_pattern);
  // 在图上运行重写器
  transpose_rewriter.runOnGraph(graph);
}

void insertPrePackedConv1dOp(std::shared_ptr<Graph>& graph) {
  // 替换图中的卷积操作为基于ATen的卷积操作
  graph_rewrite_helper::replaceConvolutionWithAtenConv(graph);

  // 定义原始的conv1d图模式字符串
  std::string conv_1d_pattern = R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[], %dilation:int[], %groups:int):
        %r = aten::conv1d(%input, %weight, %bias, %stride, %padding, %dilation, %groups)
        return (%r) )";

  // 定义预打包的conv1d图模式字符串
  std::string prepacked_ops_conv1d_pattern = R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[], %dilation:int[], %groups:int):
        %packed_weight_bias = vulkan_prepack::create_conv1d_context(
            %weight, %bias, %stride, %padding, %dilation, %groups)
        %r = vulkan_prepack::run_conv1d_context(%input, %packed_weight_bias)
        return (%r) )";

  // 创建子图重写器对象
  SubgraphRewriter rewriter;
  // 注册conv1d操作重写模式
  rewriter.RegisterRewritePattern(
      conv_1d_pattern, prepacked_ops_conv1d_pattern);
  // 在图上运行重写器
  rewriter.runOnGraph(graph);
}
void transferInputOutputBackends(std::shared_ptr<Graph>& graph) {
  // Move inputs to Vulkan backend
  // 遍历图的输入值
  for (Value* input : graph->inputs()) {
    // 创建一个命名值对象，命名为空，值为当前输入值
    NamedValue named_input = NamedValue("", input);
    // 检查输入值的类型是否为张量类型并且有使用
    if (named_input.type()->kind() == TypeKind::TensorType &&
        !input->uses().empty()) {
      // 找到插入点，在第一个使用节点之前
      WithInsertPoint ip(input->uses()[0].user->prev());
      // 在图中插入一个新节点来将输入转移到 Vulkan 后端
      Value* replaced_input = graph->insert(
          Symbol::fromQualString("aten::to"), {named_input, "vulkan"});
      // 替换输入值
      input->replaceAllUsesAfterNodeWith(
          replaced_input->node(), replaced_input);
    }
  }

  // Move outputs to CPU backend
  // 获取图的输出数组引用
  at::ArrayRef<Value*>&& outputs = graph->outputs();
  // 遍历输出
  for (size_t i = 0; i < outputs.size(); i++) {
    Value* output = outputs[i];
    // 创建一个命名值对象，命名为空，值为当前输出值
    NamedValue named_output = NamedValue("", output);
    // 检查输出值的类型是否为张量类型
    if (named_output.type()->kind() == TypeKind::TensorType) {
      // 找到插入点，在当前输出节点之后
      WithInsertPoint ip(output->node()->next());
      // 在图中插入一个新节点来将输出转移到 CPU 后端
      Value* replaced_output = graph->insert(
          Symbol::fromQualString("aten::to"), {named_output, "cpu"});
      // 替换输出值
      graph->block()->replaceOutput(i, replaced_output);
    }
  }

  // 运行子图重写器
  SubgraphRewriter rewriter;
  rewriter.runOnGraph(graph);
}

void transferInputOutputBackends(script::Module& module) {
  // 获取模块的第一个方法的图对象
  std::shared_ptr<Graph> graph = module.get_methods()[0].graph();
  // 调用图的后端转换函数
  transferInputOutputBackends(graph);
}

void eliminateDeadCode(script::Module& module) {
  // 遍历模块中的每个方法
  for (auto& method : module.get_methods()) {
    // 调用消除死代码的函数
    EliminateDeadCode(method.graph());
  }
}

void rewriteQuantizedOps(std::shared_ptr<Graph>& graph) {
  // 定义 quantized::add 的模式字符串
  std::string quantized_add_pattern = R"(
    graph(%a_quant, %b_quant, %r_scale, %r_zero_point) :
      %res = quantized::add(%a_quant, %b_quant, %r_scale, %r_zero_point)
      return (%res) )";
  // 定义 vulkan_quantized::add 的模式字符串
  std::string vk_quantized_add_pattern = R"(
    graph(%a_quant, %b_quant, %r_scale, %r_zero_point) :
      %res = vulkan_quantized::add(%a_quant, %b_quant, %r_scale, %r_zero_point)
      return (%res) )";

  // 创建量化加法的子图重写器
  torch::jit::SubgraphRewriter quantized_add_rewriter;
  // 注册 quantized::add 的重写模式
  quantized_add_rewriter.RegisterRewritePattern(
      quantized_add_pattern, vk_quantized_add_pattern);
  // 在图上运行重写器
  quantized_add_rewriter.runOnGraph(graph);

  // 定义 quantized::mul 的模式字符串
  std::string quantized_mul_pattern = R"(
    graph(%a_quant, %b_quant, %r_scale, %r_zero_point) :
      %res = quantized::mul(%a_quant, %b_quant, %r_scale, %r_zero_point)
      return (%res) )";
  // 定义 vulkan_quantized::mul 的模式字符串
  std::string vk_quantized_mul_pattern = R"(
    graph(%a_quant, %b_quant, %r_scale, %r_zero_point) :
      %res = vulkan_quantized::mul(%a_quant, %b_quant, %r_scale, %r_zero_point)
      return (%res) )";

  // 创建量化乘法的子图重写器
  torch::jit::SubgraphRewriter quantized_mul_rewriter;
  // 注册 quantized::mul 的重写模式
  quantized_mul_rewriter.RegisterRewritePattern(
      quantized_mul_pattern, vk_quantized_mul_pattern);
  // 在图上运行重写器
  quantized_mul_rewriter.runOnGraph(graph);

  // quantized::conv2d 的模式字符串未完整提供，所以这里不进行注释
}
    graph(%a_quant, %packed_params, %r_scale, %r_zero_point) :
      %res = quantized::conv2d(%a_quant, %packed_params, %r_scale, %r_zero_point)
      return (%res) )";
  std::string vk_quantized_conv2d_pattern = R"(
    graph(%a_quant, %packed_params, %r_scale, %r_zero_point):
      %output_min_max : None = prim::Constant()
      %vk_packed_params : __torch__.torch.classes.vulkan.Conv2dPackedContext = vulkan_quantized_prepack::convert_qconv2d_context(
        %packed_params, %output_min_max, %output_min_max)
      %res = vulkan_prepack::run_qconv2d_context(
        %a_quant, %r_scale, %r_zero_point, %vk_packed_params)
      return (%res) )";
  // 创建量化卷积模式匹配和替换规则
  torch::jit::SubgraphRewriter quantized_conv2d_rewriter;
  // 注册量化卷积模式匹配和替换规则
  quantized_conv2d_rewriter.RegisterRewritePattern(
      quantized_conv2d_pattern, vk_quantized_conv2d_pattern);
  // 在图上执行替换
  quantized_conv2d_rewriter.runOnGraph(graph);

  // quantized::conv_transpose2d
  // 创建量化反卷积模式匹配和替换规则
  std::string quantized_conv_transpose2d_pattern = R"(
    graph(%a_quant, %packed_params, %r_scale, %r_zero_point) :
      %res = quantized::conv_transpose2d(%a_quant, %packed_params, %r_scale, %r_zero_point)
      return (%res) )";
  std::string vk_quantized_conv_transpose2d_pattern = R"(
    graph(%a_quant, %packed_params, %r_scale, %r_zero_point):
      %output_min_max : None = prim::Constant()
      %vk_packed_params : __torch__.torch.classes.vulkan.Conv2dPackedContext = vulkan_quantized_prepack::convert_qtconv2d_context(
        %packed_params, %output_min_max, %output_min_max)
      %res = vulkan_prepack::run_qconv2d_context(
        %a_quant, %r_scale, %r_zero_point, %vk_packed_params)
      return (%res) )";
  // 创建量化反卷积模式匹配和替换规则
  torch::jit::SubgraphRewriter quantized_conv_transpose2d_rewriter;
  // 注册量化反卷积模式匹配和替换规则
  quantized_conv_transpose2d_rewriter.RegisterRewritePattern(
      quantized_conv_transpose2d_pattern,
      vk_quantized_conv_transpose2d_pattern);
  // 在图上执行替换
  quantized_conv_transpose2d_rewriter.runOnGraph(graph);

  // quantized::conv2d_relu
  // 创建量化卷积ReLU模式匹配和替换规则
  std::string quantized_conv2d_relu_pattern = R"(
    graph(%a_quant, %packed_params, %r_scale, %r_zero_point) :
      %res = quantized::conv2d_relu(%a_quant, %packed_params, %r_scale, %r_zero_point)
      return (%res) )";


  std::string vk_quantized_conv2d_relu_pattern = R"(
    graph(%a_quant, %packed_params, %r_scale, %r_zero_point):
      %output_min_max : None = prim::Constant()
      %vk_packed_params : __torch__.torch.classes.vulkan.Conv2dPackedContext = vulkan_quantized_prepack::convert_qconv2d_context(
        %packed_params, %output_min_max, %output_min_max)
      %res = vulkan_prepack::run_qconv2d_context(
        %a_quant, %r_scale, %r_zero_point, %vk_packed_params)
      return (%res) )";
    // 定义一个名为 graph 的函数，接受四个参数：%a_quant, %packed_params, %r_scale, %r_zero_point
    graph(%a_quant, %packed_params, %r_scale, %r_zero_point):
      // 创建一个浮点类型常量节点 %output_min，其值为 0.0
      %output_min: float = prim::Constant[value=0.0]()
      // 创建一个空值常量节点 %output_max
      %output_max: None = prim::Constant()
      // 调用 Vulkan 后端的函数 vulkan_quantized_prepack::convert_qconv2d_context，将 %packed_params、%output_min 和 %output_max 作为参数，返回 Vulkan 特定的卷积上下文对象 %vk_packed_params
      %vk_packed_params : __torch__.torch.classes.vulkan.Conv2dPackedContext = vulkan_quantized_prepack::convert_qconv2d_context(
        %packed_params, %output_min, %output_max)
      // 调用 Vulkan 后端的函数 vulkan_prepack::run_qconv2d_context，将 %a_quant、%r_scale、%r_zero_point 和 %vk_packed_params 作为参数，执行量化卷积操作并返回结果 %res
      %res = vulkan_prepack::run_qconv2d_context(
        %a_quant, %r_scale, %r_zero_point, %vk_packed_params)
      // 返回量化卷积的结果 %res
      return (%res) );
    
    // 创建一个名为 quantized_conv2d_relu_rewriter 的 SubgraphRewriter 对象
    torch::jit::SubgraphRewriter quantized_conv2d_relu_rewriter;
    // 注册重写模式，将 quantized_conv2d_relu_pattern 替换为 vk_quantized_conv2d_relu_pattern
    quantized_conv2d_relu_rewriter.RegisterRewritePattern(
        quantized_conv2d_relu_pattern, vk_quantized_conv2d_relu_pattern);
    // 在图 graph 上运行重写器 quantized_conv2d_relu_rewriter
    quantized_conv2d_relu_rewriter.runOnGraph(graph);
    
    // 创建一个名为 quantized_linear_pattern 的字符串，包含量化线性操作的图定义
    std::string quantized_linear_pattern = R"(
      graph(%a_quant, %packed_params, %r_scale, %r_zero_point) :
        // 执行量化线性操作，将 %a_quant、%packed_params、%r_scale 和 %r_zero_point 作为参数，将结果存储在 %res 中
        %res = quantized::linear(%a_quant, %packed_params, %r_scale, %r_zero_point)
        // 返回线性操作的结果 %res
        return (%res) )";
    // 创建一个名为 vk_quantized_linear_pattern 的字符串，包含 Vulkan 后端量化线性操作的图定义
    std::string vk_quantized_linear_pattern = R"(
      graph(%a_quant, %packed_params, %r_scale, %r_zero_point):
        // 调用 Vulkan 后端的函数 vulkan_quantized_prepack::convert_linear_context，将 %packed_params 作为参数，返回 Vulkan 特定的线性上下文对象 %vk_packed_params
        %vk_packed_params : __torch__.torch.classes.vulkan.LinearPackedContext = vulkan_quantized_prepack::convert_linear_context(
          %packed_params)
        // 调用 Vulkan 后端的函数 vulkan_prepack::run_qlinear_context，将 %a_quant、%r_scale、%r_zero_point 和 %vk_packed_params 作为参数，执行量化线性操作并返回结果 %res
        %res = vulkan_prepack::run_qlinear_context(
          %a_quant, %r_scale, %r_zero_point, %vk_packed_params)
        // 返回量化线性操作的结果 %res
        return (%res) )";
    
    // 创建一个名为 quantized_linear_rewriter 的 SubgraphRewriter 对象
    torch::jit::SubgraphRewriter quantized_linear_rewriter;
    // 注册重写模式，将 quantized_linear_pattern 替换为 vk_quantized_linear_pattern
    quantized_linear_rewriter.RegisterRewritePattern(
        quantized_linear_pattern, vk_quantized_linear_pattern);
    // 在图 graph 上运行重写器 quantized_linear_rewriter
    quantized_linear_rewriter.runOnGraph(graph);
void fuseHardtanhWithPackedOps(std::shared_ptr<Graph>& graph) {
  // 创建一个用于重写子图的重写器对象
  SubgraphRewriter rewriter;

  // 定义一个包含预打包运算的 Hardtanh 融合模式字符串
  std::string conv2d_prepack_run_hardtanh_fused = R"(
      // 定义一个图模式，匹配输入和参数列表
      graph(%input.1, %weight.1, %bias.1, %scale.1, %zero_point.1):
        // 执行预打包的卷积操作
        %conv_prepacked : Tensor = conv2d_prepack::conv2d_run(
            %input.1, %weight.1, %bias.1, %scale.1, %zero_point.1)
        // 执行 Hardtanh 激活函数
        %output : Tensor = aten::hardtanh(%conv_prepacked)
        return (%output) )";

  // 定义一个 Hardtanh 与预打包运算融合的模式
  std::string prepacked_ops_pattern = R"(
      // 定义一个图模式，匹配输入和参数列表
      graph(%input.1, %weight.1, %bias.1, %scale.1, %zero_point.1):
        // 执行预打包的卷积操作
        %conv_prepacked : Tensor = conv2d_prepack::conv2d_run(
            %input.1, %weight.1, %bias.1, %scale.1, %zero_point.1)
        // 执行预打包的 Hardtanh 操作
        %output : Tensor = conv2d_prepack::run_hardtanh(%conv_prepacked)
        return (%output) )";

  // 定义一个用于过滤的 Lambda 函数，检查节点是否输出为 Tensor[]
  auto filter = [&](const Match& match,
                    const std::unordered_map<std::string, Value*>& vmap) {
    auto node = match.values_map.at(vmap.at("weight.1"))->node();
    return node->output()->type()->str() == "Tensor[]";
  };

  // 注册重写模式，将卷积和 Hardtanh 融合为一个预打包操作
  rewriter.RegisterRewritePattern(conv2d_prepack_run_hardtanh_fused, prepacked_ops_pattern);

  // 在图上运行重写器，应用注册的模式和过滤器
  rewriter.runOnGraph(graph, filter);
}
  std::string conv2d_prepack_run_hardtanh = R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[],
          %dilation:int[], %groups:int, %output_min, %output_max, %dummy_min_max):
        %packed_weight_bias = vulkan_prepack::create_conv2d_context(
            %weight, %bias, %stride, %padding, %dilation, %groups,
            %dummy_min_max, %dummy_min_max)
        %conv2d_res = vulkan_prepack::run_conv2d_context(%input, %packed_weight_bias)
        %r = aten::hardtanh(%conv2d_res, %output_min, %output_max)
        return (%r) )";

// 注册硬性tanh函数的卷积前打包运行优化模式
rewriter.RegisterRewritePattern(
    conv2d_prepack_run_hardtanh, conv2d_prepack_run_hardtanh_fused);

std::string conv2d_prepack_run_hardtanh_inplace = R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[],
          %dilation:int[], %groups:int, %output_min, %output_max, %dummy_min_max):
        %packed_weight_bias = vulkan_prepack::create_conv2d_context(
            %weight, %bias, %stride, %padding, %dilation, %groups,
            %dummy_min_max, %dummy_min_max)
        %conv2d_res = vulkan_prepack::run_conv2d_context(%input, %packed_weight_bias)
        %r = aten::hardtanh_(%conv2d_res, %output_min, %output_max)
        return (%r) )";

// 注册原地硬性tanh函数的卷积前打包运行优化模式
rewriter.RegisterRewritePattern(
    conv2d_prepack_run_hardtanh_inplace, conv2d_prepack_run_hardtanh_fused);

// 运行重写器，以便在图中执行 Clamp 合并操作
rewriter.runOnGraph(graph, torch::jit::graph_rewrite_helper::isClampFusable);
void fuseReluWithPackedOps(std::shared_ptr<Graph>& graph) {
  // 创建一个子图重写器对象
  SubgraphRewriter rewriter;

  // 定义一个包含 fused Conv2d 和 ReLU 的图表达式字符串
  std::string conv2d_prepack_run_relu_fused = R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[],
          %dilation:int[], %groups:int, %dummy_min_max):
        %output_min: float = prim::Constant[value=0.0]()
        %output_max: None = prim::Constant()
        %packed_weight_bias : __torch__.torch.classes.vulkan.Conv2dPackedContext = vulkan_prepack::create_conv2d_context(
            %weight, %bias, %stride, %padding, %dilation, %groups,
            %output_min, %output_max)
        %r = vulkan_prepack::run_conv2d_context(%input, %packed_weight_bias)
        return (%r) )";

  // 定义一个包含单独 Conv2d 运行并应用 ReLU 的图表达式字符串
  std::string conv2d_prepack_run_relu = R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[],
          %dilation:int[], %groups:int, %dummy_min_max):
        %packed_weight_bias = vulkan_prepack::create_conv2d_context(
            %weight, %bias, %stride, %padding, %dilation, %groups,
            %dummy_min_max, %dummy_min_max)
        %conv2d_res = vulkan_prepack::run_conv2d_context(%input, %packed_weight_bias)
        %r = aten::relu(%conv2d_res)
        return (%r) )";

  // 注册将单独的 Conv2d 运行并应用 ReLU 转换为 fused Conv2d 和 ReLU 的重写模式
  rewriter.RegisterRewritePattern(
      conv2d_prepack_run_relu, conv2d_prepack_run_relu_fused);

  // 定义一个包含 inplace 操作的 fused Conv2d 和 ReLU 的图表达式字符串
  std::string conv2d_prepack_run_relu_inplace = R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[],
          %dilation:int[], %groups:int, %dummy_min_max):
        %packed_weight_bias = vulkan_prepack::create_conv2d_context(
            %weight, %bias, %stride, %padding, %dilation, %groups,
            %dummy_min_max, %dummy_min_max)
        %conv2d_res = vulkan_prepack::run_conv2d_context(%input, %packed_weight_bias)
        %r = aten::relu_(%conv2d_res)
        return (%r) )";

  // 注册将 inplace 操作的 Conv2d 运行并应用 ReLU 转换为 fused Conv2d 和 ReLU 的重写模式
  rewriter.RegisterRewritePattern(
      conv2d_prepack_run_relu_inplace, conv2d_prepack_run_relu_fused);

  // 在图中运行重写器，使用 isClampFusable 函数作为过滤器
  rewriter.runOnGraph(graph, torch::jit::graph_rewrite_helper::isClampFusable);
}
    return (
        (n->kind() ==  // 检查节点 n 的类型是否等于指定的符号类型
         Symbol::fromQualString("vulkan_prepack::create_conv2d_context")) ||  // 检查是否为 create_conv2d_context 符号
        (n->kind() ==  // 继续检查下一个符号类型
         Symbol::fromQualString("vulkan_prepack::create_tconv2d_context")) ||  // 检查是否为 create_tconv2d_context 符号
        (n->kind() ==  // 继续检查下一个符号类型
         Symbol::fromQualString("vulkan_prepack::create_qconv2d_context")) ||  // 检查是否为 create_qconv2d_context 符号
        (n->kind() ==  // 继续检查下一个符号类型
         Symbol::fromQualString("vulkan_prepack::create_qtconv2d_context")) ||  // 检查是否为 create_qtconv2d_context 符号
        (n->kind() ==  // 继续检查下一个符号类型
         Symbol::fromQualString(  // 检查是否为指定的符号类型
             "vulkan_quantized_prepack::convert_qconv2d_context")) ||  // 检查是否为 convert_qconv2d_context 符号
        (n->kind() ==  // 继续检查下一个符号类型
         Symbol::fromQualString("vulkan_prepack::create_conv1d_context")) ||  // 检查是否为 create_conv1d_context 符号
        (n->kind() ==  // 继续检查下一个符号类型
         Symbol::fromQualString(  // 检查是否为指定的符号类型
             "vulkan_quantized_prepack::convert_qtconv2d_context")) ||  // 检查是否为 convert_qtconv2d_context 符号
        (n->kind() ==  // 继续检查下一个符号类型
         Symbol::fromQualString(  // 检查是否为指定的符号类型
             "vulkan_quantized_prepack::convert_linear_context")) ||  // 检查是否为 convert_linear_context 符号
        (n->kind() ==  // 继续检查下一个符号类型
         Symbol::fromQualString("vulkan_prepack::create_linear_context")) ||  // 检查是否为 create_linear_context 符号
        (n->kind() ==  // 继续检查下一个符号类型
         Symbol::fromQualString("vulkan_prepack::create_layernorm_context")) ||  // 检查是否为 create_layernorm_context 符号
        (n->kind() ==  // 继续检查下一个符号类型
         Symbol::fromQualString("vulkan_prepack::create_gru_context")) ||  // 检查是否为 create_gru_context 符号
        (n->kind() ==  // 继续检查下一个符号类型
         Symbol::fromQualString("vulkan_prepack::create_lstm_context")) ||  // 检查是否为 create_lstm_context 符号
        (n->kind() ==  // 继续检查下一个符号类型
         Symbol::fromQualString("vulkan_prepack::create_batchnorm_context")));  // 检查是否为 create_batchnorm_context 符号
  };
  PrePackingOpsFolder(m, filter_fn, "prepack_folding");  // 将 m, filter_fn 和 "prepack_folding" 传递给 PrePackingOpsFolder 函数
} // 结束命名空间 torch
} // 结束命名空间 jit

static void vulkanRemoveMutation(script::Module& module) {
  // 获取模块中名为 "forward" 的方法的计算图
  auto graph = module.get_method("forward").graph();
  // 在计算图中移除张量变异操作
  RemoveTensorMutation(graph);
}

static void vulkanRunCanonicalOptimizations(script::Module& module) {
  // 获取模块中名为 "forward" 的方法的计算图
  auto graph = module.get_method("forward").graph();
  // 遍历模块中所有方法
  for (const auto& method : module.get_methods()) {
    // 获取每个方法的计算图
    auto method_graph = method.graph();
    // 运行优化函数，不进行循环展开
    runOptimization(method_graph, false /* no loop unrolling */);
  }
}

// 为移动设备优化模块
script::Module vulkanOptimizeForMobile(
    const script::Module& m,
    const std::set<MobileOptimizerType>& optimization_blocklist,
    const std::vector<std::string>& preserved_methods) {
  // 克隆输入的模块
  auto cloned_module = m.clone();
  // 将克隆的模块设置为评估模式
  cloned_module.eval();
  // 折叠卷积层和批归一化层
  cloned_module = FoldConvBatchNorm(cloned_module);
  // 冻结模块，保留指定的方法
  cloned_module = freeze_module(cloned_module, preserved_methods);
  // 在模块中插入预打包操作
  vulkanInsertPrePackedOps(cloned_module);
  // 融合预打包卷积和clamp操作
  vulkanFusePrePackedConvWithClamp(cloned_module);
  // 折叠预打包操作
  vulkanFoldPrePackingOps(cloned_module);
  // 移除dropout操作
  removeDropout(cloned_module);
  // 移除变异操作
  vulkanRemoveMutation(cloned_module);

  // 如果不在优化阻止列表中包含自动GPU转移优化类型
  if (!optimization_blocklist.count(
          MobileOptimizerType::VULKAN_AUTOMATIC_GPU_TRANSFER)) {
    // 转移输入输出后端
    transferInputOutputBackends(cloned_module);
    // 注册属性，指示需要后端转移
    cloned_module.register_attribute(
        "requires_backend_transfers", BoolType::get(), false);
  }

  // 运行规范优化，包括删除重复常量和消除死代码
  vulkanRunCanonicalOptimizations(cloned_module);
  eliminateDeadCode(cloned_module);

  // 注册属性，指示模块已优化为Vulkan
  cloned_module.register_attribute(
      "optimized_for_vulkan", BoolType::get(), true);
  // 返回优化后的模块
  return cloned_module;
}
```