# `.\pytorch\torch\csrc\jit\passes\peephole.cpp`

```
#include <torch/csrc/jit/passes/peephole.h>

#include <ATen/core/jit_type.h>
#include <c10/util/irange.h>
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/ir/ir_views.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/concat_opt.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/peephole_alias_sensitive.h>
#include <torch/csrc/jit/passes/peephole_dict_idioms.h>
#include <torch/csrc/jit/passes/peephole_list_idioms.h>
#include <torch/csrc/jit/passes/peephole_non_tensor.h>
#include <torch/csrc/jit/runtime/graph_executor.h>

namespace torch {
namespace jit {

// Conservatively compare two optionals. If both are undefined, assume
// they aren't equal
template <typename T>
static bool mustBeEqual(const std::optional<T>& a, const std::optional<T>& b) {
  // 比较两个可选类型的值是否相等，如果都未定义，则假定它们不相等
  return a == b && a.has_value();
}

struct PeepholeOptimizeImpl {
  PeepholeOptimizeImpl(
      // NOLINTNEXTLINE(modernize-pass-by-value)
      const std::shared_ptr<Graph>& graph,
      bool disable_shape_peepholes)
      : graph_(graph), shape_peepholes_(!disable_shape_peepholes) {}

  bool run() {
    // 运行优化，包括基本块的优化和各种局部优化
    bool changed = optimizeBlock(graph_->block());
    changed |= PeepholeOptimizeListIdioms(graph_);
    changed |= PeepholeOptimizeDictIdioms(graph_);
    changed |= PeepholeOptimizeAliasSensitive(graph_, shape_peepholes_);
    changed |= PeepholeOptimizeNonTensor(graph_);
    changed |= CombineConcats(graph_);
    return changed;
  }

  // The intent for this optimization pass is to catch all of the small, easy to
  // catch peephole optimizations you might be interested in doing.
  //
  // TODO: Decide what kind of fixed point strategy we will have
  // 优化基本块，目的是捕捉所有可以轻松捕捉的小型 peephole 优化
  bool optimizeBlock(Block* block) {
    bool changed = false;
    // 暂时没有实现具体的优化，所以没有任何变化
    return changed;
  }

 private:
  std::shared_ptr<Graph> graph_;
  bool shape_peepholes_;
};

static bool FuseAddMM(Block* block) {
  bool changed = false;
  for (Node* node : block->nodes()) {
    // XXX: remember that if you want to simplify an expression by combining
    // multiple nodes into a different one, then you need to check that they
    // all belong to the given block
    // 如果要通过将多个节点合并成一个不同的节点来简化表达式，请确保它们都属于给定的基本块
    for (Block* b : node->blocks()) {
      changed |= FuseAddMM(b);
    }
  }
  return changed;
}

// FuseAddMM is a separate pass from peephole optimize because it is currently
// used for exporting to ONNX.
// Today, fusing add + MM has no benefit within PyTorch running ATen
// ops. However, we rely on seeing the fused version of AddMM for ONNX export,
// since otherwise after ONNX translation we would see redundant Gemm ops with
// sub-optimal inputs.
// It won't be helpful for ATen until we're able to represent
//   torch.addmm(a, b, c, out=a).
// That's because addmm dispatches internally to gemm, which computes:
//   C = beta * C + alpha * A @ B
// but aten::addmm(a, b, c, 1, 1) is really:
//   D = beta * C + alpha * A @ B
// and because it works out of place on C, we're only trading off an
// 将 addmm 函数内部的显式添加副本的逻辑，添加到 addmm_fusion_enabled 参数控制的融合过程中。
// 注意，这并不会减少内存读取次数，因为对于 mm 函数，C 矩阵根本就不会加载（因为它的 beta == 0）。
bool FuseAddMM(const std::shared_ptr<Graph>& graph) {
  // 在图的基本块上执行 AddMM 融合操作，返回是否有修改
  bool changed = FuseAddMM(graph->block());
  // 输出优化后的图的调试信息
  GRAPH_DUMP("After FuseAddMM: ", graph);
  return changed;
}

// 使用给定的图和 addmm_fusion_enabled 标志来进行 Peephole 优化
bool PeepholeOptimize(
    const std::shared_ptr<Graph>& graph,
    bool addmm_fusion_enabled) {
  // 创建 PeepholeOptimizeImpl 对象，传入图和融合标志
  PeepholeOptimizeImpl peephole(graph, addmm_fusion_enabled);
  // 运行 Peephole 优化，并返回是否有修改
  bool changed = peephole.run();
  // 输出优化后的图的调试信息
  GRAPH_DUMP("After PeepholeOptimize: ", graph);
  // 如果有修改，则消除由 Peephole 优化引入的死代码
  if (changed) {
    EliminateDeadCode(graph->block());
  }
  return changed;
}

// 命名空间 jit 的结束
} // namespace jit
// 命名空间 torch 的结束
} // namespace torch
```