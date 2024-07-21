# `.\pytorch\torch\csrc\jit\passes\frozen_graph_optimizations.cpp`

```py
#include <c10/util/irange.h>
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/ir/ir_views.h>
#include <torch/csrc/jit/passes/frozen_concat_linear.h>
#include <torch/csrc/jit/passes/frozen_conv_folding.h>
#include <torch/csrc/jit/passes/frozen_graph_optimizations.h>
#include <torch/csrc/jit/passes/frozen_linear_folding.h>
#include <torch/csrc/jit/passes/remove_dropout.h>
#include <torch/csrc/jit/runtime/graph_executor.h>

namespace torch {
namespace jit {

// 对冻结图进行优化的函数，根据参数决定是否优化数值运算
void OptimizeFrozenGraph(
    std::shared_ptr<Graph>& graph,
    bool optimize_numerics) {
  
  // 移除图中的 dropout 节点
  removeDropout(graph);

  // 对图中的冻结的 ConcatLinear 进行优化
  FrozenConcatLinear(graph);

  // 如果需要优化数值计算
  if (optimize_numerics) {
    bool changed = false;
    
    // 反复运行优化，以捕捉 Conv -> Mul -> Add 等模式
    do {
      changed = false;
      
      // 尝试折叠 Conv -> BatchNorm 的模式
      changed |= FoldFrozenConvBatchnorm(graph);
      
      // 尝试折叠 Conv -> Add/Sub 的模式
      changed |= FoldFrozenConvAddOrSub(graph);
      
      // 尝试折叠 Conv -> Mul/Div 的模式
      changed |= FoldFrozenConvMulOrDiv(graph);
      
      // 尝试折叠 Linear -> BatchNorm 的模式
      changed |= FoldFrozenLinearBatchnorm(graph);
      
    } while (changed); // 继续直到不再有优化发生
  }
}

} // namespace jit
} // namespace torch
```