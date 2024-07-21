# `.\pytorch\torch\csrc\jit\passes\frozen_conv_add_relu_fusion_cuda.cpp`

```py
# 包含 ATen 库中的必要头文件和命名空间
#include <ATen/Utils.h>

#include <ATen/code_template.h>
#include <ATen/cuda/CUDAConfig.h>
#include <torch/csrc/jit/ir/constants.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/subgraph_matcher.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/frozen_conv_add_relu_fusion.h>
#include <torch/csrc/jit/passes/graph_rewrite_helper.h>
#include <torch/csrc/jit/passes/remove_mutation.h>
#include <torch/csrc/jit/passes/subgraph_rewrite.h>

namespace torch {
namespace jit {

namespace {
# 实现函数：将冻结的卷积-加法-ReLU融合到图中
void fuseFrozenConvAddReluImpl(std::shared_ptr<Graph>& graph) {
#if AT_CUDNN_ENABLED() || AT_ROCM_ENABLED()
  # 在进行融合之前，打印图的调试信息
  GRAPH_DEBUG("Before fuseFrozenConvAddReluImpl: ", *graph);
  # 定义子图重写器对象
  SubgraphRewriter rewriter;

  # 不支持的操作符列表：CUDNN 不支持 conv1d
  std::array<std::string, 2> conv_operators = {"conv2d", "conv3d"};
  std::array<std::string, 2> add_operators = {"add", "add_"};
  std::array<std::string, 2> relu_operators = {"relu", "relu_"};

  # 定义模板化的卷积-ReLU图的字符串表示
  auto conv_relu_rstring = at::jit::CodeTemplate(R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[], %dilation:int[], %groups:int):
      %x = aten::${conv}(%input, %weight, %bias, %stride, %padding, %dilation, %groups)
      %res = aten::${relu}(%x)
      return (%res))");

  # ROCm 平台下的融合字符串表示
  std::string conv_relu_fused = R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[], %dilation:int[], %groups:int):
        %res = aten::miopen_convolution_relu(%input, %weight, %bias, %stride, %padding, %dilation, %groups)
        return (%res))";

  # CUDA 平台下的融合字符串表示
  std::string conv_relu_fused = R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[], %dilation:int[], %groups:int):
        %res = aten::cudnn_convolution_relu(%input, %weight, %bias, %stride, %padding, %dilation, %groups)
        return (%res))";

  # 定义模板化的卷积-加法-ReLU图的字符串表示
  auto conv_add_relu_rstring = at::jit::CodeTemplate(R"(
    graph(%input, %weight, %bias, %z, %alpha, %stride:int[], %padding:int[], %dilation:int[], %groups:int):
      %x = aten::${conv}(%input, %weight, %bias, %stride, %padding, %dilation, %groups)
      %y = aten::${add}(%x, %z, %alpha)
      %res = aten::${relu}(%y)
      return (%res))");

  # ROCm 平台下的融合字符串表示
  std::string conv_add_relu_fused = R"(
    graph(%input, %weight, %bias, %z, %alpha, %stride:int[], %padding:int[], %dilation:int[], %groups:int):
        %res = aten::miopen_convolution_add_relu(%input, %weight, %z, %alpha, %bias, %stride, %padding, %dilation, %groups)
        return (%res))";

  # CUDA 平台下的融合字符串表示
  std::string conv_add_relu_fused = R"(
    graph(%input, %weight, %bias, %z, %alpha, %stride:int[], %padding:int[], %dilation:int[], %groups:int):
        %res = aten::cudnn_convolution_add_relu(%input, %weight, %z, %alpha, %bias, %stride, %padding, %dilation, %groups)
        return (%res))";

  # 遍历卷积操作符数组，开始融合
  for (const auto& conv : conv_operators) {
    // 遍历 relu_operators 中的每个 relu 操作符
    for (const auto& relu : relu_operators) {
      // 创建模板环境变量 env
      at::jit::TemplateEnv env;
      // 在 env 中设置 "conv" 变量为 conv
      env.s("conv", conv);
      // 在 env 中设置 "relu" 变量为当前的 relu 操作符
      env.s("relu", relu);
      // 注册重写模式，将 conv_relu_rstring 格式化后注册到 rewriter 中
      rewriter.RegisterRewritePattern(
          conv_relu_rstring.format(env), conv_relu_fused);
      // 遍历 add_operators 中的每个 add 操作符
      for (const auto& add : add_operators) {
        // 在 env 中设置 "add" 变量为当前的 add 操作符
        env.s("add", add);
        // 注册重写模式，将 conv_add_relu_rstring 格式化后注册到 rewriter 中
        rewriter.RegisterRewritePattern(
            conv_add_relu_rstring.format(env), conv_add_relu_fused);
      }
    }
  }

  // 定义一个 lambda 函数 filter，用于过滤匹配项
  auto filter = [](const Match& match,
                   const std::unordered_map<std::string, Value*>& vmap) {
    // 获取匹配项中 "weight" 变量的值并转换为 IValue
    auto weight = toIValue(match.values_map.at(vmap.at("weight")));
    // 如果 weight 不存在或者不是 Tensor 类型，则返回 false
    if (!weight.has_value() || !weight.value().isTensor()) {
      return false;
    }
    // 获取 weight 对应的 Tensor
    const at::Tensor& weight_t = weight.value().toTensor();
    // 如果 weight_t 不在 CUDA 设备上或者不是连续的，则返回 false
    if (!weight_t.device().is_cuda() || !weight_t.is_contiguous()) {
      return false;
    }

    // bias 是可选的
    if (vmap.find("bias") != vmap.end()) {
      // 获取匹配项中 "bias" 变量的值并转换为 IValue
      auto bias = toIValue(match.values_map.at(vmap.at("bias")));
      // 如果 bias 存在并且是 Tensor 类型
      if (bias.has_value() && bias.value().isTensor()) {
        // 获取 bias 对应的 Tensor
        const at::Tensor& bias_t = bias.value().toTensor();
        // 如果 bias_t 的数据类型与 weight_t 不同，或者维度不是 1，
        // 或者第一个维度大小不匹配，或者不在 CUDA 设备上，则返回 false
        if (bias_t.dtype() != weight_t.dtype() || bias_t.ndimension() != 1 ||
            bias_t.size(0) != weight_t.size(0) || !bias_t.device().is_cuda()) {
          return false;
        }
      }
    }

    // z 是可选的
    if (vmap.find("z") != vmap.end()) {
      // 获取匹配项中 "z" 变量的值并转换为 IValue
      auto z = toIValue(match.values_map.at(vmap.at("z")));
      // 如果 z 存在并且是 Tensor 类型
      if (z.has_value() && z.value().isTensor()) {
        // 获取 z 对应的 Tensor
        const at::Tensor& z_t = z.value().toTensor();
        // 如果 z_t 的数据类型与 weight_t 不同，或者第一个维度大小不匹配，
        // 或者不是连续的，或者不在 CUDA 设备上，则返回 false
        if (z_t.dtype() != weight_t.dtype() ||
            z_t.size(0) != weight_t.size(0) || !z_t.is_contiguous() ||
            !z_t.device().is_cuda()) {
          return false;
        }
      }
    }
    // 若所有条件都满足，则返回 true
    return true;
  };

  // 使用 graph_rewrite_helper 工具替换 _convolution 和原地操作符以便进行简单的替换模式匹配
  graph_rewrite_helper::replaceConvolutionWithAtenConv(graph);

  // 在图上运行重写器 rewriter，并使用 filter 函数进行过滤
  rewriter.runOnGraph(graph, filter);
  // 打印调试信息，显示经过 fuseFrozenConvAddReluImpl 处理后的图形状态
  GRAPH_DEBUG("After fuseFrozenConvAddReluImpl: ", *graph);
#endif

# 结束一个预处理指令块，通常与 `#ifdef` 或 `#if` 配对使用，用于条件编译。


}

# 结束一个函数或代码块。


auto dummyInitializer = []() {

# 定义一个匿名的 lambda 函数，并将其赋值给 `dummyInitializer` 变量，该 lambda 函数没有参数。


  getFuseFrozenConvAddReluImpl() = fuseFrozenConvAddReluImpl;

# 调用 `getFuseFrozenConvAddReluImpl()` 函数，并将其返回值与 `fuseFrozenConvAddReluImpl` 赋值给左侧的结果。


  return true;

# lambda 函数的返回语句，返回 `true`。


}();

# 立即调用定义的匿名 lambda 函数。


} // namespace

# 结束一个命名空间 `jit`。


} // namespace torch

# 结束一个命名空间 `torch`。
```