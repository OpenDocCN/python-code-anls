# `.\pytorch\torch\csrc\jit\passes\peephole_non_tensor.cpp`

```
/**
 * Include headers for specific Torch JIT passes and utility libraries.
 */
#include <torch/csrc/jit/passes/peephole.h>
#include <torch/csrc/jit/passes/peephole_non_tensor.h>
#include <ATen/core/jit_type.h>
#include <c10/util/irange.h>
#include <torch/csrc/jit/ir/ir_views.h>
#include <torch/csrc/jit/jit_log.h>

/**
 * Define the namespace for Torch JIT functionalities.
 */
namespace torch {
namespace jit {

/**
 * Anonymous namespace for internal helper functions and structures.
 */
namespace {

/**
 * Check if a given arithmetic node involves integer operands and return a constant
 * integer value if one of the operands is constant.
 *
 * @param node The node to analyze.
 * @return An optional int64_t containing the constant value if found, otherwise std::nullopt.
 *
 * @pre node is an integer arithmetic operation.
 * @post If one operand is a constant, the function returns that constant.
 */
std::optional<int64_t> checkArithNode(Node& node) {
  if (node.inputs().size() != 2 || node.input(0)->type() != IntType::get() ||
      node.input(1)->type() != IntType::get()) {
    return {};
  }

  if (node.kind() == aten::mul || node.kind() == aten::add) {
    if (auto i = constant_as<int64_t>(node.input(0))) {
      node.permuteInputs({1, 0});
      return i;
    }
  }

  return constant_as<int64_t>(node.input(1));
}

/**
 * Attempt to simplify a multiplication or division node if it multiplies or divides
 * by the constant value 1.
 *
 * @param node The node to simplify.
 * @return True if simplification was successful, otherwise false.
 *
 * @pre node is either aten::mul, aten::floordiv, or aten::div.
 */
bool trySimplifyMulOrDiv(Node& node) {
  auto constant = checkArithNode(node);
  if (!constant || *constant != 1) {
    return false;
  }

  node.output()->replaceAllUsesWith(node.inputs()[0]);
  return true;
}

/**
 * Simplify an addition or subtraction node by merging constant parts together.
 *
 * @param node The node to simplify.
 * @return True if simplification was successful, otherwise false.
 *
 * @pre node is either aten::add or aten::sub.
 */
bool trySimplifyAddOrSub(Node& node) {
  auto constant = checkArithNode(node);
  if (!constant) {
    return false;
  }

  if (constant == 0) {
    node.output()->replaceAllUsesWith(node.input(0));
    return true;
  }

  auto& dep = *node.inputs()[0]->node();
  if (dep.kind() != aten::add && dep.kind() != aten::sub) {
    return false;
  }

  auto delta = checkArithNode(dep);
  if (!delta) {
    return false;
  }
  auto merged =
      dep.kind() == node.kind() ? *constant + *delta : *constant - *delta;

  if (merged == 0) {
    node.output()->replaceAllUsesWith(dep.inputs()[0]);
  } else {
    WithInsertPoint g(&node);
    node.replaceInput(0, dep.inputs()[0]);
    node.replaceInput(1, node.owningGraph()->insertConstant(merged));
  }
  return true;
}

} // namespace

/**
 * Struct to implement peephole optimization for non-tensor operations within a Torch JIT graph.
 */
struct PeepholeOptimizeNonTensorImpl {
  /**
   * Constructor to initialize with a shared pointer to the graph.
   *
   * @param graph Shared pointer to the Torch JIT graph.
   */
  // NOLINTNEXTLINE(modernize-pass-by-value)
  PeepholeOptimizeNonTensorImpl(const std::shared_ptr<Graph>& graph)
      : graph_(graph) {}

  /**
   * Run the peephole optimization on the graph's block.
   *
   * @return True if any optimizations were applied, otherwise false.
   */
  bool run() {
    return optimizeBlock(graph_->block());
  }

  /**
   * Perform peephole optimization on a specific block within the graph.
   *
   * @param block Pointer to the block to optimize.
   * @return True if any optimizations were applied, otherwise false.
   */
  bool optimizeBlock(Block* block) {
    bool changed = false;
    // Optimization logic goes here.
    return changed;
  }

 private:
  std::shared_ptr<Graph> graph_;
};

/**
 * Peephole optimization entry point for non-tensor operations within a Torch JIT graph.
 *
 * @param graph Shared pointer to the Torch JIT graph to optimize.
 * @return True if any optimizations were applied, otherwise false.
 */
bool PeepholeOptimizeNonTensor(const std::shared_ptr<Graph>& graph) {
  PeepholeOptimizeNonTensorImpl peephole(graph);
  bool changed = peephole.run();
  // Dump the graph after optimization for debugging purposes.
  GRAPH_DUMP("After PeepholeOptimize: ", graph);
  return changed;
}

} // namespace jit
} // namespace torch
```