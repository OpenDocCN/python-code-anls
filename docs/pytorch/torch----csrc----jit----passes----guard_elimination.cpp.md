# `.\pytorch\torch\csrc\jit\passes\guard_elimination.cpp`

```py
#include <torch/csrc/jit/passes/guard_elimination.h>

#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/peephole.h>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <memory>
#include <unordered_set>

namespace torch {
namespace jit {

// GuardElimination 结构体的构造函数，接受一个共享指针指向图对象，并初始化成员变量
struct GuardElimination {
  GuardElimination(std::shared_ptr<Graph> graph)
      : graph_(std::move(graph)), aliasDb_(std::make_unique<AliasDb>(graph_)) {}

  // 运行保护消除优化的主函数
  void run() {
    const size_t MAX_ATTEMPTS = 5;
    size_t attempts = MAX_ATTEMPTS;
    // 尝试进行多次保护移动到定义处操作，最多尝试 MAX_ATTEMPTS 次
    while (attempts-- && moveGuardsToDefs(graph_->block())) {
    }
    // 在日志中输出优化后的图信息
    GRAPH_DUMP("After moveGuardsToDefs", graph_);
    // 合并相邻的保护节点
    coalesceGuards(graph_->block());
    // 在日志中输出合并后的图信息
    GRAPH_DUMP("After coalesceGuards", graph_);
    // 移除被主导的保护节点
    removeDominatedGuards(graph_->block());
    // 在日志中输出移除被主导保护后的图信息
    GRAPH_DUMP("After removeDominatedGuards", graph_);
    // 消除多余的保护节点
    eliminateRedundantGuards(graph_->block());
    // 在日志中输出消除多余保护后的图信息
    GRAPH_DUMP("After eliminateRedundantGuards", graph_);
  }

  // 静态方法，用于判断节点是否为降低的梯度节点
  static bool isLoweredGradOf(Node* n) {
    if (n->kind() != prim::If) {
      return false;
    }

    return n->input(0)->node()->kind() == prim::AutogradAnyNonZero;
  }

  // 将保护节点移动到其定义处的方法
  bool moveGuardsToDefs(Block* b) {
    bool changed = false;
    // 遍历基本块中的节点
    for (auto it = b->nodes().begin(); it != b->nodes().end();) {
      auto n = *it;
      if (n->kind() == prim::Guard) {
        // 备份当前节点的下一个节点，因为将当前节点移动可能会改变迭代器的指向
        it++;
        auto guardee = n->inputs().at(0)->node();
        // 如果保护对象在不同的基本块中，将其移动到当前基本块的第一个节点位置
        if (guardee->owningBlock() != n->owningBlock()) {
          guardee = *n->owningBlock()->nodes().begin();
        }
        // 利用别名分析移动保护节点到合适的位置
        bool moved = aliasDb_->moveAfterTopologicallyValid(n, guardee);
        changed |= moved;
        // 如果成功移动，记录日志
        if (moved) {
          GRAPH_UPDATE(
              "Moved ",
              n->output()->debugName(),
              " to ",
              n->inputs().at(0)->debugName());
        }
      } else {
        it++;
        // 递归处理节点的子块
        for (Block* ib : n->blocks()) {
          moveGuardsToDefs(ib);
        }
      }
    }

    // 如果当前基本块属于降低的梯度条件的所有者，并且包含保护节点，则将这些节点移动到条件节点之前
    if (b->owningNode() &&
        isLoweredGradOf(
            b->owningNode()) /*b->owningNode()->kind() == prim::If*/) {
      for (auto it = b->nodes().begin(); it != b->nodes().end();) {
        auto block_node = *it++;
        if (block_node->kind() != prim::Guard) {
          break;
        }
        block_node->moveBefore(b->owningNode());
        changed = true;
      }
    }

    return changed;
  }

  // 合并相邻的保护节点的方法
  void coalesceGuards(Block* b) {
    // 将所有参数的使用移动到同一个锚点节点，并且它们可能会在锚点节点之后以不同的顺序出现
    // 例如 (anchor, guard_x, guard_y, guard_x, guard_y)
    // 这一步骤识别连续的保护节点序列


继续部分代码的注释，请让我知道是否继续。
    // 维护每个 def 所见过的 guard，下次在相同的 def 上遇到时直接移除
    std::unordered_map<Value*, Node*> inputs_to_guards;
    // 遍历基本块中的所有节点
    for (auto it = b->nodes().begin(); it != b->nodes().end(); it++) {
      auto n = *it;
      // 如果节点是 prim::Guard 类型
      if (n->kind() == prim::Guard) {
        // 如果已经存在相同输入的 guard，则进行替换操作
        if (inputs_to_guards.count(n->input())) {
          auto prev = inputs_to_guards[n->input()];
          // 用之前的 guard 输出替换当前 guard 输出的所有使用
          n->output()->replaceAllUsesWith(prev->output());
          // 输出替换的更新日志
          GRAPH_UPDATE(
              "Replacing ",
              n->output()->debugName(),
              " with ",
              prev->output()->debugName());
          // 销毁当前节点
          it.destroyCurrent();
        } else {
          // 否则，将当前节点作为新的 guard 加入 map 中
          inputs_to_guards.insert({n->input(), n});
        }
      } else if (n->kind() != prim::Constant) {
        // 如果节点不是常量，则清空 inputs_to_guards，准备处理下一个节点的 guards
        inputs_to_guards.clear();
        // 对于节点内部的每一个子块，递归调用 coalesceGuards 函数
        for (Block* ib : n->blocks()) {
          coalesceGuards(ib);
        }
      }
    }
  }

  // 移除被支配的 guards
  void removeDominatedGuards(Block* b) {
    // 如果一个节点 guard 的值没有被修改，则该节点可以替换掉它支配的所有其他 guards
    for (auto it = b->nodes().begin(); it != b->nodes().end(); it++) {
      auto n = *it;
      // 如果节点是 prim::Guard 类型
      if (n->kind() == prim::Guard) {
        Value* input = n->input();
        // 如果输入有写入操作，则继续下一个节点
        if (aliasDb_->hasWriters(input)) {
          continue;
        }
        Value* guard_output = n->output();

        // 找到被该 guard 节点支配的所有输入的使用
        std::vector<Use> uses = input->uses();
        while (!uses.empty()) {
          auto use = uses.at(uses.size() - 1);
          uses.pop_back();

          // 不是所有的使用都被 guard
          if (use.user->kind() != prim::Guard) {
            continue;
          }

          // 如果当前使用节点被当前节点支配
          if (!use.user->isDominatedBy(n)) {
            continue;
          }

          // 当前 guard 的输出类型可能与被支配的 guard 的输出类型不同
          // 在继续之前检查类型是否相等
          auto dominator_type = guard_output->type();
          auto dominated_type = use.user->output()->type();

          if (*dominator_type == *dominated_type) {
            // 用当前 guard 输出替换被支配 guard 的输入
            use.user->replaceInput(use.offset, guard_output);
          }
        }

        // 移除多余的被支配 guards
        std::vector<Use> users = n->output()->uses();
        for (auto use : users) {
          auto user = use.user;
          if (user->kind() == prim::Guard) {
            // 输出日志更新，移除被支配的 guard，并替换为当前 guard
            GRAPH_UPDATE(
                "Removing dominated guard ", user, " and replacing with ", n);
            user->output()->replaceAllUsesWith(guard_output);
            // 销毁被支配的 guard 节点
            user->destroy();
          }
        }
      } else {
        // 如果节点不是 guard，则递归处理其内部的每一个子块
        for (Block* ib : n->blocks()) {
          removeDominatedGuards(ib);
        }
      }
  }
  // 结束一个函数定义

  // we need to make sure there are no ops in between guardee's
  // output and its guard except for other guards as they can
  // invalidate shape information.
  // 确保在 guardee 的输出和它的 guard 之间没有操作，除非是其他的 guard，
  // 因为它们可能会使形状信息失效。
  bool guardsOutput(Node* guard) {
    // 获取 guardee 的输出节点
    auto output = guard->input()->node();
    // 从当前 guard 节点追溯到输出节点，检查中间是否有不符合预期的节点
    auto it = guard;
    while (it != output) {
      // 如果发现不是 Guard 或 Constant 类型的节点，则输出调试信息并返回 false
      if (it->kind() != prim::Guard && it->kind() != prim::Constant) {
        GRAPH_DEBUG(
            "found an unexpected node ",
            *it,
            " while trying to eliminate ",
            *guard);
        return false;
      }
      it = it->prev();  // 继续向前查找
    }

    return true;  // 没有发现异常节点，返回 true
  }

  // eliminateRedundantGuards 函数的目的是消除多余的 guards 节点
  void eliminateRedundantGuards(Block* b) {
    // 一个简单的遍历函数，用于消除那些其输出完全由输入决定的操作的多余 guards 节点
    // 如果这些操作的输入受到保护，我们可以移除这些操作输出的 guards
    for (auto it = b->nodes().rbegin(); it != b->nodes().rend();) {
      auto n = *it;
      // 如果当前节点是 Guard 类型，并且 guardsOutput 函数返回 true，
      // 同时 removableGuard 函数也返回 true，则可以移除这个 guard
      if (n->kind() == prim::Guard && guardsOutput(n) &&
          removableGuard(n->inputs().at(0)->node())) {
        // 获取当前 guard 节点的输出类型
        auto pttp = n->output()->type();
        // 替换掉所有使用当前 guard 输出的地方为它的输入节点
        n->output()->replaceAllUsesWith(n->inputs().at(0));
        // 设置输入节点的类型为之前保存的类型
        n->inputs().at(0)->setType(pttp);
        // 输出调试信息，表明正在消除多余的 guard
        GRAPH_UPDATE(
            "Eliminating the redundant guard ", n->output()->debugName());
        it.destroyCurrent();  // 从节点列表中移除当前节点
      } else {
        it++;
        // 递归处理当前节点的所有子块
        for (Block* ib : n->blocks()) {
          eliminateRedundantGuards(ib);
        }
      }
    }
  }

  // `checkInputs` check the invariants specified in `removableGuard`
  // on inputs to `n`. The invariants must hold, or an input must
  // be a `prim::Constant` or be included as an exception in `except`
  // `checkInputs` 函数用于检查在 `removableGuard` 中指定的不变量是否适用于 `n` 的输入。
  // 这些不变量必须保持不变，或者输入必须是 `prim::Constant` 或者在 `except` 中作为例外包含。
  bool checkInputs(
      Node* n,
      const std::unordered_set<size_t>& except,
      bool allow_numbers) {
    // 初始化变量，用于记录所有输入是否都受保护
    bool all_inputs_guarded = true;
    size_t i = 0;
    // 遍历节点 `n` 的所有输入
    for (auto input : n->inputs()) {
      // 如果输入节点是 Guard 类型并且没有被汇总（summarized），或者是 Constant 类型，
      // 或者允许数字类型（allow_numbers 为真），或者在例外集合 `except` 中，
      // 则满足条件，否则输出调试信息并标记输入未受保护，然后跳出循环
      if ((input->node()->kind() == prim::Guard &&
           !input->type()->expectRef<TensorType>().isSummarized()) ||
          input->node()->kind() == prim::Constant ||
          (allow_numbers && input->type()->isSubtypeOf(*NumberType::get())) ||
          except.count(i) != 0) {
        AT_ASSERT(
            input->node()->kind() != prim::Guard ||
            input->type()->expect<TensorType>());
      } else {
        GRAPH_DEBUG(
            "input ",
            input->debugName(),
            " isn't guarded, type ",
            *input->type());
        all_inputs_guarded = false;
        break;
      }
      i++;
    }
  // 返回 `all_inputs_guarded` 变量，表示是否所有输入都被保护
  return all_inputs_guarded;
}

private:
// `removableGuard` 函数依赖于 `isSummarized()` 函数检查的属性，
// 并且不应该在保护和使用之间插入可能修改这些属性的节点。
// `removableGuard` 函数期望类型信息直接来自 Profiler。
// 通行应该避免尝试修改由分析提供的类型信息。
// 尽管我们可以推导出非常简单的规则，说明在某些情况下删除 `prim::Guard`
// 是有效的，例如如果所有输入都已被保护，则可以删除操作输出上的 `prim::Guard`。
// 对于某些操作的类别，没有涵盖 PyTorch 中所有可用操作的综合规则。
// 如果您的操作属于以下描述的某个类别，请将其添加到下面的 switch 语句中，
// 其中包含其他操作的所述类别。
// 否则，您需要自行推导出您的情况的规则。
// 一般来说，任何以任何方式具有状态或使用其底层数据计算任何属性的操作，
// `isSummarized()` 并不适合于保护消除。
// 类别：
// * 具有广播语义的类似函数（例如 add、sub、le）操作
//   如果所有输入都被保护，并且 `isSummarized()` 返回 false，或者输入是 `prim::Constant`，
//   则可以移除保护
bool removableGuard(Node* n) {
  const static auto no_exceptions = std::unordered_set<size_t>{};
}

std::shared_ptr<Graph> graph_;
std::unique_ptr<AliasDb> aliasDb_;
static std::unordered_set<Symbol> simple_ops_;
};

void EliminateRedundantGuards(std::shared_ptr<Graph> graph) {
  // 创建 GuardElimination 对象 ge，传入图对象 graph 并移动所有权
  GuardElimination ge(std::move(graph));
  // 调用 GuardElimination 对象的 run 方法执行冗余保护删除操作
  ge.run();
}

} // namespace jit
} // namespace torch
```