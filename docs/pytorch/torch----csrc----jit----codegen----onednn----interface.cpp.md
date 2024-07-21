# `.\pytorch\torch\csrc\jit\codegen\onednn\interface.cpp`

```
// 包含头文件：一API DNNL 图形处理、OneDNN 的 SILU 分解、延迟大小检查、图融合、形状保护、接口、内核、布局传播、二进制准备、JIT 日志、分解操作、Pass 管理、移除突变、Tensorexpr 融合、自定义运算符、图执行器、运算符选项
#include <oneapi/dnnl/dnnl_graph.hpp>
#include <torch/csrc/jit/codegen/onednn/decompose_silu.h>
#include <torch/csrc/jit/codegen/onednn/defer_size_check.h>
#include <torch/csrc/jit/codegen/onednn/graph_fuser.h>
#include <torch/csrc/jit/codegen/onednn/guard_shape.h>
#include <torch/csrc/jit/codegen/onednn/interface.h>
#include <torch/csrc/jit/codegen/onednn/kernel.h>
#include <torch/csrc/jit/codegen/onednn/layout_propagation.h>
#include <torch/csrc/jit/codegen/onednn/prepare_binary.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/decompose_ops.h>
#include <torch/csrc/jit/passes/pass_manager.h>
#include <torch/csrc/jit/passes/remove_mutation.h>
#include <torch/csrc/jit/passes/tensorexpr_fuser.h>
#include <torch/csrc/jit/runtime/custom_operator.h>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <torch/csrc/jit/runtime/operator_options.h>

// 定义命名空间：torch::jit::fuser::onednn
namespace torch {
namespace jit {
namespace fuser {
namespace onednn {

// 定义函数：将图融合到一个单独的节点中
void fuseGraph(std::shared_ptr<Graph>& g) {
  // 根据 tensorexpr_fuser 的分析模式进行处理：
  // 移除 prim::profile 节点，并将 profile 信息直接嵌入 IR 中的值类型中，以避免破坏融合模式。
  // 在 LLGA 优化 Passes 之后添加形状保护，并从 IR 中清除张量类型信息，以免被其他 Pass 不小心使用。

  // 我们依赖形状特化和形状保护来确保内核中缓存编译的有效性，因此仅支持分析模式。
  // TODO: 在 oneDNNFusionGroup 上添加检查，确保要融合的节点上的 allShapesAreKnown：torch/csrc/jit/passes/tensorexpr_fuser.cpp: allShapesAreKnown
  if (getProfilingMode()) {
    GRAPH_DUMP(
        "Before RemoveProfileNodesAndSpecializeTypes. Beginning of LLGA "
        "optimization pass",
        g);
    // 移除 profile 节点并特化类型
    RemoveProfileNodesAndSpecializeTypes(g);
    GRAPH_DUMP(
        "After RemoveProfileNodesAndSpecializeTypes. Before mutation removal",
        g);

    // 移除张量突变，并保留特定的运算符类型
    RemoveTensorMutation(g, [](Node* nodeToFunctionalize) {
      static std::unordered_set<Symbol> supportedOps = {
          aten::add_,
          aten::mul_,
          aten::tanh_,
          aten::elu_,
          aten::relu_,
          aten::relu6_,
          aten::gelu_,
          aten::sqrt_,
          aten::sigmoid_,
          aten::hardtanh_,
          aten::abs_,
          aten::square_,
          aten::pow_,
          aten::leaky_relu_,
          aten::round_,
          aten::exp_,
          aten::abs_,
          aten::hardswish_,
          aten::silu_};
      return supportedOps.count(nodeToFunctionalize->kind()) != 0;
    });
    // 移除列表突变
    RemoveListMutation(g);
    GRAPH_DUMP("After mutation removal. Before DecomposeSiluForLlga", g);
    // 对 SILU 进行 LLGA 分解
    DecomposeSiluForLLGA(g);
    GRAPH_DUMP("After DecomposeSiluForLlga. Before PrepareBinaryForLLGA", g);
    // 为 LLGA 准备二进制
    PrepareBinaryForLLGA(g);
    GRAPH_DUMP("After PrepareBinaryForLLGA. Before DeferSizeCheck", g);
    // 延迟大小检查
    DeferSizeCheck(g);
    # 在执行 DeferSizeCheck 后，记录当前图形状态并输出以便调试
    GRAPH_DUMP("After DeferSizeCheck. Before CreateLlgaSubgraphs", g);
    
    # 设置常量张量缓存，优化图形操作
    dnnl::graph::set_constant_tensor_cache(true);
    
    # 创建 LLGA 子图形，用于特定优化处理
    CreateLlgaSubgraphs(g);
    
    # 在执行 CreateLlgaSubgraphs 后，记录当前图形状态并输出以便调试
    GRAPH_DUMP("After CreateLlgaSubgraphs. Before PropagateLayout", g);
    
    # 根据布局传播规则调整图形结构
    PropagateLayout(g);
    
    # 在执行 PropagateLayout 后，记录当前图形状态并输出以便调试
    GRAPH_DUMP(
        "After PropagateLayout. Before prepareFusionGroupAndGuardOutputs", g);
    
    # 为了在分析模式下增加形状保护，同时从 IR 中清除张量类型信息
    prepareFusionGroupAndGuardOutputs(g->block());
    
    # 在执行 prepareFusionGroupAndGuardOutputs 后，记录当前图形状态并输出以便调试
    GRAPH_DUMP(
        "After prepareFusionGroupAndGuardOutputs. Before "
        "RemoveTensorTypeSpecializations",
        g);
    
    # 移除张量类型的特定优化，以确保最终的 LLGA 优化步骤
    RemoveTensorTypeSpecializations(g);
    
    # 在执行 RemoveTensorTypeSpecializations 后，记录当前图形状态并输出以便调试，标志 LLGA 优化步骤结束
    GRAPH_DUMP(
        "After RemoveTensorTypeSpecializations. End of LLGA optimization pass",
        g);
} // namespace onednn
} // namespace fuser



static Operation createLlgaKernel(const Node* node) {
  // 创建一个名为 kernel 的 shared_ptr，类型为 fuser::onednn::LlgaKernel，初始化传入当前节点 node
  auto kernel = std::make_shared<fuser::onednn::LlgaKernel>(node);
  // 返回一个 lambda 函数，该函数记录 kernel 的调试名称，并运行 kernel 的 run 方法
  return [kernel](Stack& stack) {
    RECORD_FUNCTION(kernel->debugName(), std::vector<c10::IValue>());
    kernel->run(stack);
    return 0;
  };
}

RegisterOperators oneDNNFusionGroupOp({
    torch::jit::Operator(
        prim::oneDNNFusionGroup,
        createLlgaKernel,
        AliasAnalysisKind::INTERNAL_SPECIAL_CASE),
});



// Currently, we convert some scalar inputs, such as the second argument of
// binary ops to a 1D tensor. Other scalar inputs are prim::Constant nodes.
// But if we have any scalar inputs to guard in the future, some logic here
// would have to be changed.
static Operation createLlgaGuardKernel(const Node* node) {
  // 返回一个 lambda 函数，该函数用于检查节点 node 的输入类型是否符合预期
  return [node](Stack& stack) {
#ifdef GRAPH_DEBUG_ENABLED
    // 如果开启了 GRAPH_DEBUG_ENABLED，则输出当前正在保护的节点信息
    GRAPH_DEBUG("Guarding node: ", node->kind().toQualString());
#endif
    // 获取节点的输入类型列表
    std::vector<TypePtr> types = node->tys(attr::types);
    // 获取输入的数量
    const auto num_inputs = types.size();
#ifdef GRAPH_DEBUG_ENABLED
    // 如果开启了 GRAPH_DEBUG_ENABLED，则输出要保护的输入数量信息
    GRAPH_DEBUG("num_inputs to guard: ", num_inputs);
#endif
    // 遍历所有输入
    for (size_t i = 0; i < num_inputs; i++) {
#ifdef GRAPH_DEBUG_ENABLED
      // 如果开启了 GRAPH_DEBUG_ENABLED，则输出当前正在检查的输入信息
      GRAPH_DEBUG("checking input ", i);
#endif
      // 获取当前输入的引用
      auto& input = peek(stack, i, num_inputs);
      // 获取当前输入期望的 tensor 类型
      const c10::TensorTypePtr& guard_tensor_type =
          types[i]->cast<TensorType>();

      // 如果当前输入不是 tensor 类型
      if (!input.isTensor()) {
#ifdef GRAPH_DEBUG_ENABLED
        // 如果开启了 GRAPH_DEBUG_ENABLED，则输出当前输入不是 tensor 的信息
        GRAPH_DEBUG("input ", i, " is not a tensor, return false");
#endif
        // 将 false 压入栈中，表示检查失败
        push(stack, IValue(false));
        return;
      }
      // 获取当前输入的 tensor
      const at::Tensor& tensor = input.toTensor();

      // 如果输入 tensor 是 mkldnn 类型，表示它来自一个通过输入形状检查的 LLGA 分区
      // 可以继续执行检查，因为 oneDNN 图分区的输出形状由输入形状决定
      if (tensor.is_mkldnn()) {
#ifdef GRAPH_DEBUG_ENABLED
        // 如果开启了 GRAPH_DEBUG_ENABLED，则输出当前输入是 mkldnn 的信息
        GRAPH_DEBUG("input ", i, " is_mkldnn, continue");
#endif
        continue;
      }

      // 如果输入 tensor 类型与期望的类型不匹配
      if (!guard_tensor_type->matchTensor(tensor)) {
#ifdef GRAPH_DEBUG_ENABLED
        // 如果开启了 GRAPH_DEBUG_ENABLED，则输出当前输入检查失败的信息
        GRAPH_DEBUG("input ", i, " check failed, return false");
#endif
        // 将 false 压入栈中，表示检查失败
        push(stack, IValue(false));
        return;
      }
    }
#ifdef GRAPH_DEBUG_ENABLED
    // 如果开启了 GRAPH_DEBUG_ENABLED，则输出所有检查完成的信息
    GRAPH_DEBUG("all check done, return true");
#endif
    // 将 true 压入栈中，表示所有检查都通过
    push(stack, IValue(true));
    return;
  };
}

// 注册一个名为 oneDNNGuardOp 的操作符，用于处理 prim::oneDNNFusionGuard 操作
RegisterOperators oneDNNGuardOp({
    torch::jit::Operator(
        prim::oneDNNFusionGuard,
        createLlgaGuardKernel,
        AliasAnalysisKind::FROM_SCHEMA),
});

} // namespace jit
} // namespace torch
```