# `.\pytorch\torch\csrc\jit\passes\fuse_relu.cpp`

```
#include <torch/csrc/jit/passes/fuse_relu.h>

#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/subgraph_matcher.h>
#include <torch/csrc/jit/passes/subgraph_rewrite.h>

namespace torch {
namespace jit {

namespace {
// 实现将 add 和 relu 操作融合的函数
void fuseAddReluImpl(std::shared_ptr<Graph>& graph) {
  // 创建子图重写器对象
  SubgraphRewriter rewriter;

  // 定义第一个模式，匹配 add 和 relu 的图模式
  std::string add_relu_0 = R"(
    graph(%a, %b, %alpha):
        %add_res = aten::add(%a, %b, %alpha)
        %res = aten::relu(%add_res)
        return (%res))";
  // 定义将 add 和 relu 替换为单一操作的图模式
  std::string add_relu_fused = R"(
    graph(%a, %b, %alpha):
        %res = aten::_add_relu(%a, %b, %alpha)
        return (%res))";
  // 注册第一个模式和其对应的融合后的图模式
  rewriter.RegisterRewritePattern(add_relu_0, add_relu_fused);

  // 定义第二个模式，匹配 add 和 relu_ 的图模式
  std::string add_relu_1 = R"(
    graph(%a, %b, %alpha):
        %add_res = aten::add(%a, %b, %alpha)
        %res = aten::relu_(%add_res)
        return (%res))";
  // 注册第二个模式和其对应的融合后的图模式
  rewriter.RegisterRewritePattern(add_relu_1, add_relu_fused);

  // 定义第三个模式，匹配 add_ 和 relu_ 的图模式
  std::string add_inplace_relu_1 = R"(
    graph(%a, %b, %alpha):
        %add_res = aten::add_(%a, %b, %alpha)
        %res = aten::relu_(%add_res)
        return (%res))";
  // 定义将 add_ 和 relu_ 替换为单一操作的图模式
  std::string add_inplace_relu_fused = R"(
    graph(%a, %b, %alpha):
        %res = aten::_add_relu_(%a, %b, %alpha)
        return (%res))";
  // 注册第三个模式和其对应的融合后的图模式
  rewriter.RegisterRewritePattern(add_inplace_relu_1, add_inplace_relu_fused);

  // 定义第四个模式，匹配 add 和 relu_ 输出到指定张量的图模式
  std::string add_out_relu = R"(
    graph(%a, %b, %alpha, %out):
        %add_res = aten::add(%a, %b, %alpha, %out)
        %res = aten::relu_(%add_res)
        return (%res))";
  // 定义将 add 和 relu_ 输出到指定张量替换为单一操作的图模式
  std::string add_out_relu_fused = R"(
    graph(%a, %b, %alpha, %out):
        %res = aten::_add_relu(%a, %b, %alpha, %out)
        return (%res))";
  // 注册第四个模式和其对应的融合后的图模式
  rewriter.RegisterRewritePattern(add_out_relu, add_out_relu_fused);

  // 在给定的图上运行重写器
  rewriter.runOnGraph(graph);
  // 注意事项：未包含的模式包括 add_ + relu 和 add_out + relu
  // 这是因为 add_ 的原位变异会在实际执行 add+relu 时丢失
}
} // namespace

// 对脚本模块中的前向方法应用 add 和 relu 融合
void FuseAddRelu(script::Module& module) {
  auto graph = module.get_method("forward").graph();
  fuseAddReluImpl(graph);
}

// 对给定的图应用 add 和 relu 融合
void FuseAddRelu(std::shared_ptr<Graph>& graph) {
  fuseAddReluImpl(graph);
}
} // namespace jit
} // namespace torch
```