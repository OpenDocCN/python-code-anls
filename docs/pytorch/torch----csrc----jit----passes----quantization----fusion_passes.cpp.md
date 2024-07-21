# `.\pytorch\torch\csrc\jit\passes\quantization\fusion_passes.cpp`

```
#include <torch/csrc/jit/passes/quantization/fusion_passes.h>
#include <torch/csrc/jit/passes/subgraph_rewrite.h>

namespace torch {
namespace jit {

namespace {
// 定义一个函数，用于将 quantized::add 和 aten::relu 合并成 quantized::add_relu
void fuseQuantizeAddReluImpl(std::shared_ptr<Graph>& graph) {
  // 创建一个子图重写器对象
  SubgraphRewriter fused_add_relu_rewriter;

  // 定义 quantized::add 和 aten::relu 合并的模式字符串及合并后的模式字符串
  std::string quantized_add_relu_pattern = R"(
    graph(%a_quant, %b_quant, %scale, %zero_point):
         %add_out = quantized::add(%a_quant, %b_quant, %scale, %zero_point)
         %r = aten::relu(%add_out)
         return (%r) )";
  std::string fused_add_relu_pattern = R"(
    graph(%a_quant, %b_quant, %scale, %zero_point):
         %r = quantized::add_relu(%a_quant, %b_quant, %scale, %zero_point)
         return (%r) )";
  // 注册 quantized::add + aten::relu 到 quantized::add_relu 的重写模式
  fused_add_relu_rewriter.RegisterRewritePattern(
      quantized_add_relu_pattern, fused_add_relu_pattern);

  // 同样的操作，定义并注册其他模式的重写规则
  std::string quantized_add_out_relu_pattern = R"(
    graph(%a_quant, %b_quant, %out_quant):
         %add_out = quantized::add_out(%a_quant, %b_quant, %out_quant)
         %r = aten::relu(%add_out)
         return (%r) )";
  std::string fused_add_out_relu_pattern = R"(
    graph(%a_quant, %b_quant, %out_quant):
         %r = quantized::add_relu_out(%a_quant, %b_quant, %out_quant)
         return (%r) )";
  fused_add_relu_rewriter.RegisterRewritePattern(
      quantized_add_out_relu_pattern, fused_add_out_relu_pattern);

  std::string quantized_add_scalar_relu_pattern = R"(
    graph(%a_quant, %b_scalar):
         %add_out = quantized::add_scalar(%a_quant, %b_scalar)
         %r = aten::relu(%add_out)
         return (%r) )";
  std::string fused_add_scalar_relu_pattern = R"(
    graph(%a_quant, %b_scalar):
         %r = quantized::add_scalar_relu(%a_quant, %b_scalar)
         return (%r) )";
  fused_add_relu_rewriter.RegisterRewritePattern(
      quantized_add_scalar_relu_pattern, fused_add_scalar_relu_pattern);

  std::string quantized_add_scalar_out_relu_pattern = R"(
    graph(%a_quant, %b_scalar, %out_quant):
         %add_out = quantized::add_scalar_out(%a_quant, %b_scalar, %out_quant)
         %r = aten::relu(%add_out)
         return (%r) )";
  std::string fused_add_scalar_out_relu_pattern = R"(
    graph(%a_quant, %b_scalar, %out_quant):
         %r = quantized::add_scalar_relu_out(%a_quant, %b_scalar, %out_quant)
         return (%r) )";
  fused_add_relu_rewriter.RegisterRewritePattern(
      quantized_add_scalar_out_relu_pattern, fused_add_scalar_out_relu_pattern);

  // 在输入的图上运行子图重写器，将定义的模式应用到图中
  fused_add_relu_rewriter.runOnGraph(graph);
}
} // namespace

// 对外暴露的函数，用于调用内部实现的量化加和ReLU合并操作
void FuseQuantizedAddRelu(std::shared_ptr<Graph>& graph) {
  fuseQuantizeAddReluImpl(graph);
}

} // namespace jit
} // namespace torch
```