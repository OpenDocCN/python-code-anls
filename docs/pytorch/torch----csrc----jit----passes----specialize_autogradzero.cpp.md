# `.\pytorch\torch\csrc\jit\passes\specialize_autogradzero.cpp`

```
#include <torch/csrc/jit/passes/specialize_autogradzero.h>

#include <c10/util/Exception.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <torch/csrc/jit/runtime/profiling_record.h>

#include <ATen/core/symbol.h>
#include <c10/util/irange.h>

namespace torch {
namespace jit {

// 定义常量，用于表示计数属性的符号
static const auto countsAttribute = Symbol::attr("none_counts");

// 判断给定值是否在使用 aten::_grad_sum_to_size 操作
static bool hasGradSumToSizeUses(Value* v) {
  return std::any_of(v->uses().begin(), v->uses().end(), [](const Use& use) {
    return use.user->kind() == aten::_grad_sum_to_size;
  });
}

// 在指定的基本块中插入分析节点，用于特化 AutogradZero
static void insertProfileNodesForSpecializeAutogradZero(
    Block* block,
    ProfilingRecord* pr) {
  // 遍历基本块中的每个节点
  for (auto it = block->nodes().begin(); it != block->nodes().end(); ++it) {
    auto n = *it;
    // 遍历节点的每个输入值
    for (const auto offset : c10::irange(n->inputs().size())) {
      auto i = n->input(offset);
      // 检查输入值是否为 OptionalType，且在使用 aten::_grad_sum_to_size 操作
      if (i->type()->cast<OptionalType>() && hasGradSumToSizeUses(i)) {
        // 创建用于分析的 ProfileIValueNode
        auto opt_pn = pr->createProfileIValueNode(i);

        // 初始化 noneCountsDict 字典
        c10::Dict<std::string, int64_t> noneCountsDict;
        noneCountsDict.insert("num_none", 0);
        noneCountsDict.insert("num_present", 0);
        IValue init_val(noneCountsDict);

        // 设置初始计数属性值
        opt_pn->ival_(countsAttribute, init_val);

        // 定义用于处理 Optional 值的回调函数
        std::function<void(Stack&)> optional_profiler = [pr,
                                                         opt_pn](Stack& stack) {
          std::lock_guard<std::mutex> lock(pr->mutex_);

          // 断言检查计数属性是否存在
          TORCH_INTERNAL_ASSERT(opt_pn->hasAttribute(countsAttribute));
          // frame_id 未使用
          int64_t frame_id = 0;
          pop(stack, frame_id);

          // 获取计数属性值和输入值
          const auto& counts_attr = opt_pn->ival(countsAttribute);
          auto noneCounts = c10::impl::toTypedDict<std::string, int64_t>(
              counts_attr.toGenericDict());
          IValue value;
          pop(stack, value);
          // 根据输入值是否为 None，更新计数属性值
          if (value.isNone()) {
            noneCounts.insert_or_assign(
                "num_none", noneCounts.at("num_none") + 1);
          } else {
            noneCounts.insert_or_assign(
                "num_present", noneCounts.at("num_present") + 1);
          }
          push(stack, value);
        };
        // 设置回调函数
        opt_pn->setCallback(optional_profiler);
        // 在节点之后插入分析节点，替换所有使用节点
        opt_pn->insertAfter(i->node());
        i->replaceAllUsesAfterNodeWith(opt_pn, opt_pn->output());
      }
    }

    // 递归处理子块
    for (auto ib : n->blocks()) {
      insertProfileNodesForSpecializeAutogradZero(ib, pr);
    }
  }
}

// 向特化 AutogradZero 插入分析节点
void InsertProfileNodesForSpecializeAutogradZero(ProfilingRecord* pr) {
  insertProfileNodesForSpecializeAutogradZero(pr->profiled_graph_->block(), pr);
}

// AutogradZero 特化器结构
struct AutogradZeroSpecializer {
  enum class State { Nonzero, Zero, Unknown };

  // 构造函数，初始化图形
  AutogradZeroSpecializer(std::shared_ptr<Graph> graph)
      : graph_(std::move(graph)) {}

  // 执行特化 AutogradZero 的主要方法
  void run() {
    // 如果不是反向图，则直接返回，不执行后续操作
    if (!isBackwardGraph()) {
      return;
    }
    // 如果处于执行器模式下
    if (getExecutorMode()) {
      // 尝试对特化进行保护
      if (auto versioning_if = guardSpecializations()) {
        // 对自动求导操作进行特化
        specializeAutogradOps(versioning_if->blocks()[0]);
        // 输出特化后的图结构
        GRAPH_DUMP("After versioning graph", graph_);
      }
    } else {
      // 在图输入上设置状态
      setStatesOnGraphInputs();
      // 对自动求导操作进行特化
      specializeAutogradOps(graph_->block());
    }
    // 输出自动求导特化操作后的图结构
    GRAPH_DUMP("After specializeAutogradOps graph", graph_);
  }

 private:
  // 判断是否为反向图
  bool isBackwardGraph() {
    return std::any_of(
        graph_->nodes().begin(), graph_->nodes().end(), [](Node* n) {
          switch (n->kind()) {
            // 检查结点类型，如果是以下类型之一，则返回 true
            case prim::AutogradAnyNonZero:
            case prim::AutogradAdd:
            case aten::_grad_sum_to_size:
              return true;
            default:
              return false;
          }
        });
  }

  // 将块的输入替换为图的输入
  void replaceBlockInputsWithGraphInputs(Block* b) {
    TORCH_INTERNAL_ASSERT(graph_->inputs().size() == b->inputs().size());
    size_t num_inputs = graph_->inputs().size();
    // 逐个替换块的输入为图的输入
    for (const auto i : c10::irange(num_inputs)) {
      b->inputs().at(i)->replaceAllUsesWith(graph_->inputs().at(i));
    }
    // 逐个删除块的输入
    for (const auto i : c10::irange(num_inputs)) {
      b->eraseInput(num_inputs - (1 + i));
    }
  }

  // 在图的输入上设置状态
  void setStatesOnGraphInputs() {
    for (Value* input : graph_->inputs()) {
      const auto& tp = input->type();
      if (auto tt = tp->cast<TensorType>()) {
        if (tt->undefined()) {
          // 如果是未定义的张量类型，根据具体情况设置状态为 Zero 或 Nonzero
          if (*tt->undefined()) {
            state_[input] = State::Zero;
          } else {
            state_[input] = State::Nonzero;
          }
        } else {
          // 如果不是未定义的张量类型，则设置状态为 Unknown
          state_[input] = State::Unknown;
        }
      } else if (
          tp->isSubtypeOf(*TensorType::get()) ||
          tp->isSubtypeOf(*ListType::ofTensors())) {
        // 如果是张量或张量列表类型，则设置状态为 Nonzero
        state_[input] = State::Nonzero;
      } else {
        // 否则设置状态为 Unknown
        state_[input] = State::Unknown;
      }
    }
  }

  // 获取具有指定属性的使用节点
  static void getUsesWithAttribute_(
      Value* inp,
      Symbol attr,
      std::vector<Node*>& uses) {
    for (auto use : inp->uses()) {
      if (use.user->kind() != prim::profile_ivalue) {
        continue;
      }
      if (use.user->hasAttribute(attr)) {
        uses.push_back(use.user);
      }
      // 递归查找具有指定属性的使用节点
      getUsesWithAttribute_(use.user->output(), attr, uses);
    }
  }

  // 获取具有指定属性的使用节点的向量
  // 用于处理 `specializeAutogradZero` 插入的 `prim::profile_ivalue`
  static std::vector<Node*> getUsesWithAttribute(Value* inp, Symbol attr) {
    std::vector<Node*> uses;
    getUsesWithAttribute_(inp, attr, uses);
    return uses;
  }

  // 获取具有指定类型的使用节点
  static Node* getUse(Value* inp, Symbol kind) {
    for (auto use : inp->uses()) {
      if (use.user->kind() == kind) {
        return use.user;
      }
    }
    return nullptr;
  }

  // 删除具有指定使用的可选使用节点
  void removeProfiledOptionalUses(const std::vector<Node*>& uses) {
    TORCH_INTERNAL_ASSERT(!uses.empty());
    auto inp = uses[0]->input();
    // 断言第一个使用节点的输入不为空
    // （实际上，这里需要一个更详细的功能说明，但根据要求，只能提供注释）
    // 遍历当前节点的所有使用情况
    for (auto u : uses) {
      // 用新的输入替换所有使用当前节点输出的地方
      u->output()->replaceAllUsesWith(inp);
    }
  }

  // 用于生成条件分支的节点，用于版本化特殊处理
  Node* guardSpecializations() {
    // 创建一个带有多个输出的条件节点
    auto versioning_if = graph_->create(prim::If, {}, graph_->outputs().size());
    // 定义一个值映射函数
    auto value_map = [](Value* v) { return v; };
    // 添加条件节点的真分支和假分支
    auto true_block = versioning_if->addBlock();
    auto false_block = versioning_if->addBlock();

    // 对真分支进行优化
    true_block->cloneFrom(graph_->block(), value_map);
    // 将真分支的输入替换为图的输入
    replaceBlockInputsWithGraphInputs(true_block);
    false_block->cloneFrom(graph_->block(), value_map);
    // 将假分支替换为带有图输入的回退图
    replaceBlockWithFallbackGraph(false_block, graph_->inputs());

    // 设置插入点为参数节点的下一个节点
    WithInsertPoint wip{graph_->block()->param_node()->next()};
    // 在图中插入一个常量值
    Value* none_val = graph_->insertConstant(IValue());
    // 定义多个值的向量
    std::vector<Value*> checks;
    std::vector<Value*> zero_values;
    std::vector<Value*> nonzero_values;

    // 遍历图的输入
    for (auto inp : graph_->inputs()) {
      // 获取具有指定属性的使用节点
      std::vector<Node*> iprofile_counts_nodes =
          getUsesWithAttribute(inp, countsAttribute);
      if (!iprofile_counts_nodes.empty()) {
        // 如果存在使用节点，复制到真分支和假分支中
        auto profile_ivalue_node = iprofile_counts_nodes[0];
        // 断言节点具有指定属性
        TORCH_INTERNAL_ASSERT(
            profile_ivalue_node->hasAttribute(countsAttribute));
        // 获取属性的值
        const auto& counts_attr =
            profile_ivalue_node->ival(countsAttribute).toGenericDict();
        auto num_present = counts_attr.at(IValue{"num_present"}).toInt();
        auto num_none = counts_attr.at(IValue{"num_none"}).toInt();
        if (num_present == 0 && num_none != 0) {
          // 在图中插入一个新节点来检查是否相等
          auto check = graph_->insert(aten::__is__, {inp, none_val})->node();
          checks.push_back(check->output());
          // 将输入插入已配置为零的状态
          profiled_none_.insert(inp);
        }
        // 移除使用节点
        removeProfiledOptionalUses(iprofile_counts_nodes);
        continue;
      }

      if (inp->uses().empty() || !inp->type()->cast<TensorType>()) {
        continue;
      }

      // TODO: 检查多个用途？
      // 获取使用节点
      auto pout = getUse(inp, prim::profile);
      if (!pout) {
        continue;
      }

      auto pttp = pout->ty(attr::profiled_type)->expect<TensorType>();
      if (!pttp->undefined().has_value()) {
        continue;
      }

      // 状态为零或非零
      state_[inp] = *pttp->undefined() ? State::Zero : State::Nonzero;

      if (*pttp->undefined()) {
        zero_values.push_back(inp);
      } else {
        nonzero_values.push_back(inp);
      }
    }
    // 在循环后对图进行转储
    GRAPH_DUMP("After for loop", graph_);
    // 无法特殊化任何输入
    // 检查非零值集合和零值集合是否均为空
    if (nonzero_values.empty() && zero_values.empty()) {
      // 在图中记录无法添加任何专门化保护的消息
      GRAPH_DUMP("Unable to add any specialization guards", graph_);
      // 销毁 versioning_if 节点，之后的死代码消除（DCE）会清理插入的检查
      versioning_if->destroy();
      // 返回空指针，表示无法添加专门化保护
      return nullptr;
    }

    // 创建并插入一个 AutogradAllNonZero 节点
    Node* nonzero_check = graph_->insert(prim::AutogradAllNonZero, {})->node();
    // 将所有非零值作为输入添加到 AutogradAllNonZero 节点
    for (Value* v : nonzero_values) {
      nonzero_check->addInput(v);
    }
    // 将 AutogradAllNonZero 节点的输出添加到 checks 中
    checks.push_back(nonzero_check->output());

    // 创建并插入一个 AutogradAllZero 节点
    Node* zero_check = graph_->insert(prim::AutogradAllZero, {})->node();
    // 将所有零值作为输入添加到 AutogradAllZero 节点
    for (Value* v : zero_values) {
      zero_check->addInput(v);
    }
    // 将 AutogradAllZero 节点的输出添加到 checks 中
    checks.push_back(zero_check->output());

    // 创建一个布尔类型的列表，其中包含 checks 中所有值
    Value* bool_list =
        graph_->insertNode(graph_->createList(BoolType::get(), checks))
            ->output();
    // 执行 and 操作，检查列表中的所有值是否为真
    Value* conjunction = graph_->insert(aten::all, {bool_list});

    // 将 and 操作的结果作为 versioning_if 节点的输入
    versioning_if->addInput(conjunction);
    // 在图中插入 versioning_if 节点
    graph_->insertNode(versioning_if);

    // 获取返回节点 ret
    auto ret = graph_->return_node();
    // 替换 ret 节点的输入为 versioning_if 的输出
    for (const auto i : c10::irange(ret->inputs().size())) {
      auto ogo = ret->input(i);
      auto ngo = versioning_if->output(i);
      ngo->copyMetadata(ogo);
      ret->replaceInput(i, ngo);
    }

    // 我们创建了如下结构：
    // successful_checks = Guards(...)
    // if (successful_checks)
    // -> optimized graph
    // else:
    // -> fallback graph
    // original graph
    //
    // 移除死代码的原始图
    // 从图块的最后一个节点开始遍历，直到找到 versioning_if 节点为止，然后销毁该节点
    for (auto it = graph_->block()->nodes().reverse().begin();
         *it != versioning_if;) {
      Node* n = *it;
      it++;
      n->destroy();
    }

    // 在 "After guardSpecializations" 后记录当前图的状态
    GRAPH_DUMP("After guardSpecializations", graph_);
    // 返回 versioning_if 节点，表示成功添加了专门化保护
    return versioning_if;
  }
};

// 继续传播自动求导的零信息通过梯度图，并在可能时移除 grad_of 块。
// 注意：这是一个非常有限的处理过程。它只传播由符号自动微分代码生成的操作的自动求导零，并在可能时清理 AutogradAdds。
// 其他节点的输出被保守地标记为 Unknown，并且不进行优化。

// 专门处理自动求导零信息的函数，接受一个共享指针指向图形对象 g。
void specializeAutogradZero(std::shared_ptr<Graph> g) {
  // 创建 AutogradZeroSpecializer 对象 azs，使用给定的图形对象初始化。
  AutogradZeroSpecializer azs(std::move(g));
  // 运行 AutogradZeroSpecializer 对象的处理逻辑。
  azs.run();
}

// 命名空间 jit 的结束标记
} // namespace jit

// 命名空间 torch 的结束标记
} // namespace torch
```