# `.\pytorch\torch\csrc\jit\passes\peephole_alias_sensitive.cpp`

```
#include <ATen/core/jit_type.h>
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/ir/ir_views.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/peephole.h>
#include <torch/csrc/jit/passes/peephole_alias_sensitive.h>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <unordered_set>

namespace torch {
namespace jit {

// This pass only does optimizations which requires Alias Analysis
// It is separated out from Peephole Pass so that Peephole does not have
// maintain alias db correctness throughout the pass.
struct PeepholeOptimizeAliasSensitiveImpl {
  PeepholeOptimizeAliasSensitiveImpl(
      std::shared_ptr<Graph> graph,
      bool shape_peepholes)
      : graph_(std::move(graph)),
        aliasDb_(std::make_unique<AliasDb>(graph_)),
        shape_peepholes_(shape_peepholes) {}

  // Run the alias-sensitive peephole optimization on the given graph.
  bool run() {
    return runBlock(graph_->block());
  }

 private:
  // Replace a value in the graph with a constant IValue.
  void replaceWithIValue(Value* v, IValue val) {
    WithInsertPoint guard(v->node());
    v->replaceAllUsesWith(v->owningGraph()->insertConstant(val));
  }

  // Check if the tensor type is floating point.
  bool isFloatingPoint(TensorType& t) {
    auto input_dtype = t.scalarType();
    return (
        shape_peepholes_ && input_dtype && at::isFloatingType(*input_dtype));
  }

  // Run the alias-sensitive peephole optimization on a block within the graph.
  bool runBlock(Block* block) {
    bool changed = false;
    // Placeholder for actual optimization logic
    // (Currently there are no statements in this function in the provided code)
    return changed;
  }

  // Try to replace the output value with the input value if safe to do so.
  bool tryToReplaceOutputWithInput(Value* input, Value* output) {
    // Check if it's safe to change the aliasing relationship between input and output.
    if (!aliasDb_->safeToChangeAliasingRelationship(input, output)) {
      return false;
    }
    
    // Check if input and output may alias any stale alias values.
    if (aliasDb_->mayAlias({input, output}, stale_alias_values_)) {
      return false;
    }
    
    // Replace all uses of output with input and mark both as stale for alias analysis.
    output->replaceAllUsesWith(input);
    stale_alias_values_.insert(input);
    stale_alias_values_.insert(output);
    return true;
  }

  ValueSet stale_alias_values_; // Set of values with stale aliasing properties
  std::shared_ptr<Graph> graph_; // The graph to optimize
  std::unique_ptr<AliasDb> aliasDb_; // Alias database for the graph
  bool shape_peepholes_; // Flag indicating whether to consider shape peepholes
};

// Top-level function to run alias-sensitive peephole optimization on a graph.
bool PeepholeOptimizeAliasSensitive(
    const std::shared_ptr<Graph>& graph,
    bool shape_peepholes) {
  PeepholeOptimizeAliasSensitiveImpl opt(graph, shape_peepholes);
  return opt.run();
}

} // namespace jit
} // namespace torch
```