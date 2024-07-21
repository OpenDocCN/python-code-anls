# `.\pytorch\torch\csrc\jit\passes\bailout_graph.cpp`

```
#include <torch/csrc/jit/passes/bailout_graph.h>

#include <ATen/core/function.h>
#include <c10/util/irange.h>
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/ir/ir_views.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/clear_profiling.h>
#include <torch/csrc/jit/passes/constant_pooling.h>
#include <torch/csrc/jit/passes/liveness.h>
#include <memory>
#include <unordered_set>
#include <utility>

namespace torch {
namespace jit {

// 确定一个节点是否应该被捕获到 bailout 图中
static bool shouldBeCapturedInByBailOut(Node* n) {
  return n->kind() != prim::Constant;
}

// 为单个节点构建 bailout 图的构造器
struct BailOutGraphBuilderForNode {
  explicit BailOutGraphBuilderForNode(
      std::shared_ptr<Graph> graph,
      std::shared_ptr<Graph> target)
      : graph_(std::move(graph)), copy_graph_(std::move(target)) {}

  // 向 bailout 图中添加一个新的输入值，捕获旧值并映射到新输入
  Value* addNewInputForValue(Value* old_value) {
    auto node = old_value->node();
    // 如果节点是常量节点，直接克隆并创建一个新常量节点
    if (node->kind() == prim::Constant) {
      TORCH_INTERNAL_ASSERT(!shouldBeCapturedInByBailOut(node));
      auto new_const = copy_graph_->createClone(node, {nullptr});
      copy_graph_->block()->prependNode(new_const);
      return new_const->output();
    }

    // 否则，将旧值加入到 live_inputs_ 列表中，并为其创建新的输入值
    live_inputs_.push_back(old_value);
    auto new_value = copy_graph_->block()->addInput();
    GRAPH_DEBUG(
        "Adding a new value %",
        new_value->debugName(),
        " for %",
        old_value->debugName());
    return mapValueAndCopyMetadata(old_value, new_value);
  }

  // 映射值并复制元数据，将旧值映射到新值并复制元数据
  Value* mapValueAndCopyMetadata(Value* old_value, Value* new_value) {
    this->old_to_new_[old_value] = new_value;
    new_value->copyMetadata(old_value);
    return new_value;
  }

  // 获取或添加值的输入，如果值尚未映射到输入，则添加新的输入值
  Value* getOrAddInputForValue(Value* v) {
    if (this->old_to_new_.count(v) == 0) {
      return addNewInputForValue(v);
    } else {
      return this->old_to_new_[v];
    }
  }

  // 获取值的输入，确保值已经映射到输入中
  Value* getInputForValue(Value* v) {
    TORCH_INTERNAL_ASSERT(this->old_to_new_.count(v));
    return this->old_to_new_[v];
  }

  // 克隆节点并构建 bailout 块，从给定节点 `n` 开始直到块的结尾
  // 如果 `n` 属于 `prim::If` 或 `prim::Loop`，则继续从块的所有者节点构建
  void buildBailOutBlockFrom(Node* n) {
    auto b = n->owningBlock();
    for (auto it = n->iterator(); it != b->nodes().end(); it++) {
      cloneNode(*it);
    }
  }

  // 节点克隆函数，克隆给定的节点，并更新输出值的映射
  Node* cloneNode(Node* node) {
    auto* block = copy_graph_->block();
    // 环境函数，用于获取或添加节点输入的映射
    auto env = [this](Value* v) { return getOrAddInputForValue(v); };

    // 克隆节点，并更新输出值的映射
    auto new_node = block->appendNode(copy_graph_->createClone(node, env));
    for (size_t i = 0; i < node->outputs().size(); ++i) {
      auto oo = node->outputs()[i];
      auto no = new_node->outputs()[i];
      old_to_new_[oo] = no;
    }

    return new_node;
  }

 private:
  std::shared_ptr<Graph> graph_;    // 原始图
  std::shared_ptr<Graph> copy_graph_;  // bailout 图的副本
  std::vector<Value*> live_inputs_;   // bailout 图的输入值列表
  std::unordered_map<Value*, Value*> old_to_new_;  // 旧值到新值的映射
};

} // namespace jit
} // namespace torch
    // 获取当前节点所属块的所属节点，用于后续继续构建“bailout”图
    auto outer_node = n->owningBlock()->owningNode();
    // 如果存在外部节点
    if (outer_node) {
      // 如果外部节点是循环节点
      if (outer_node->kind() == prim::Loop) {
        // 调用构建循环“bailout”的函数
        buildBailOutLoop(outer_node);
      } else if (outer_node->kind() == prim::If) {
        // 如果外部节点是条件节点
        // 调用构建条件“bailout”的函数，传入当前块的输出和外部节点
        buildBailOutIf(b->outputs(), outer_node);
      } else {
        // 如果外部节点既不是循环也不是条件节点，抛出异常
        AT_ERROR("Unexpected outer node");
      }
    }
  }

  // 将块输出映射到当前值，用于构建“bailout”图
  void mapValues(
      const at::ArrayRef<Value*> block_outputs,
      const at::ArrayRef<Value*> carried_deps) {
    TORCH_INTERNAL_ASSERT(block_outputs.size() == carried_deps.size());
    // 遍历块输出和对应的依赖
    for (const auto i : c10::irange(block_outputs.size())) {
      // 获取或添加块输出对应的输入值
      auto nv = getOrAddInputForValue(block_outputs[i]);
      // 将旧依赖映射到新值
      old_to_new_[carried_deps[i]] = nv;
    }
  }

  // 构建循环“bailout”图
  void buildBailOutLoop(Node* outer_node) {
    // 获取循环视图
    LoopView lv(outer_node);
    // 获取或添加最大迭代次数的输入值
    auto old_max_count = getOrAddInputForValue(lv.maxTripCount());
    // 获取当前迭代次数的输入值
    auto cur_iter = getInputForValue(lv.currentTripCount());
    // 获取循环体块的输出
    auto block_outputs = lv.bodyBlock()->outputs();

    auto* block = copy_graph_->block();
    // 在块的末尾插入点
    WithInsertPoint guard(*block->nodes().end());
    // 计算更新后的最大迭代次数
    auto updated_max_trip_count =
        copy_graph_->insert(aten::sub, {old_max_count, cur_iter});
    // 插入常数值 `1`
    auto one = copy_graph_->insertConstant({1});
    // 计算更新后的最大迭代次数，再减去 `1`
    updated_max_trip_count =
        copy_graph_->insert(aten::sub, {updated_max_trip_count, one});
    // 计算当前迭代次数加 `1`
    auto cur_plus_one = copy_graph_->insert(aten::add, {one, cur_iter});

    // 当映射 `block_outputs` 到继续循环的输入时，需要小心
    // 因为 `cloneFrom` 将会在 `prim::Loop` 和下面示例中的 `aten::cat` 中
    // 用相同的值替换 `%4`:
    //
    // ... : Tensor = prim::Loop(%MAX_TRIP_COUNT, %COND, ..., %4)
    //   block0(%i.2 : int, ...):
    //     ...
    //     %y.5 : Double(3) = aten::cat(%22, %4)
    //     ...
    //
    // 然而在克隆的循环节点中，值应该是不同的。
    // 即 `prim::Loop` 中的值应来自于当前迭代的输出，而 `aten::cat` 中的 `%4`
    // 需要映射到“bailout”图中 `%4` 的克隆值。为了解决这个问题，我们手动克隆循环节点

    // 将残余循环的输入映射到当前迭代的输出 (`block_outputs`)
    auto new_loop =
        copy_graph_->insertNode(copy_graph_->create(prim::Loop, {}, 0))
            ->setSourceRange(outer_node->sourceRange());
    new_loop->addInput(updated_max_trip_count);
    for (auto bo : block_outputs) {
      new_loop->addInput(getOrAddInputForValue(bo));
    }

    // 克隆循环体并将旧循环的输出映射到新循环的输出
    auto new_loop_body = new_loop->addBlock();
    auto env = [this](Value* v) { return getOrAddInputForValue(v); };
    new_loop_body->cloneFrom(lv.bodyBlock(), env);
    // 对于lv.carriedOutputs()中的每个输出ov，将其添加到new_loop的输出中
    for (auto ov : lv.carriedOutputs()) {
      auto no = new_loop->addOutput();
      mapValueAndCopyMetadata(ov, no);
    }
    
    // 创建一个新的LoopView对象new_lv，使用new_loop初始化它
    LoopView new_lv(new_loop);
    
    {
      // 在新循环体的第一个节点之前设置插入点
      WithInsertPoint guard_in_loop(*new_lv.bodyBlock()->nodes().begin());
      
      // 插入一个新的aten::add节点adj_iter_ctr，将cur_plus_one和one作为输入
      auto adj_iter_ctr = copy_graph_->insert(aten::add, {cur_plus_one, one});
      
      // 用adj_iter_ctr替换new_lv.currentTripCount()的所有用途
      new_lv.currentTripCount()->replaceAllUsesWith(adj_iter_ctr);
      
      // 将adj_iter_ctr节点的输入one替换为new_lv.currentTripCount()
      adj_iter_ctr->node()->replaceInputWith(one, new_lv.currentTripCount());
    }

    // 如果outer_node有下一个节点，则构建从下一个节点到BailOut块的跳转
    if (outer_node->next()) {
      buildBailOutBlockFrom(outer_node->next());
    }
  }

  // 根据outer_node的条件输出和block_outputs，映射对应的值
  void buildBailOutIf(
      const at::ArrayRef<Value*> block_outputs,
      Node* outer_node) {
    auto if_outputs = outer_node->outputs();
    mapValues(block_outputs, if_outputs);
    
    // 构建从outer_node的下一个节点到BailOut块的跳转
    buildBailOutBlockFrom(outer_node->next());
  }

  // 从节点n构建一个BailOut图形，添加需要的输入和输出
  std::shared_ptr<Graph> buildBailOutGraphFrom(Node* n) {
    // 为n的输入添加图形输入，确保我们可以正确对齐BailOut节点的输入参数
    for (auto bi : n->inputs()) {
      getOrAddInputForValue(bi);
    }

    // 构建从节点n到BailOut块的跳转
    buildBailOutBlockFrom(n);
    
    // 添加图形的输出
    for (auto ov : graph_->outputs()) {
      copy_graph_->registerOutput(getOrAddInputForValue(ov));
    }
    
    // 返回复制后的图形copy_graph_
    return copy_graph_;
  }

  // 原始图形和复制后的图形的指针
  std::shared_ptr<Graph> graph_;
  std::shared_ptr<Graph> copy_graph_;
  
  // 存储live_inputs_和old_to_new_的向量和映射
  std::vector<Value*> live_inputs_;
  std::unordered_map<Value*, Value*> old_to_new_;
};

// `BailOutInserter` 结构体用于在图中插入 prim::BailOut 节点，用于指示解释器从特定点恢复执行原始图的未优化（去优化）版本
struct BailOutInserter {
  // 构造函数，接受一个图的共享指针作为参数
  explicit BailOutInserter(std::shared_ptr<Graph> graph)
      : graph_(std::move(graph)), bailout_index_(0) {}

  // 运行插入操作的主函数
  void run() {
    // 构建 liveness_sets_，用于记录图中每个点的活跃信息
    liveness_sets_ = BuildLivenessSets(graph_);
    // 在图的块中插入 BailOut 节点
    insertBailOuts(graph_->block());
    // 将 Guards 替换为 BailOuts
    replaceGuardsWithBailouts();
    // 将原始未优化图嵌入到 BailOut 中
    addUnoptimizedFuncToBailouts();
  }

  // 将原始未优化图打包成一个 Function 常量，并将其作为 prim::BailOut 点的第一个输入
  // 用于计算给定 BailOut 点的去优化图
  void addUnoptimizedFuncToBailouts() {
    // 复制原始图
    auto unoptimized_graph = graph_->copy();
    // 创建一个 prim::BailoutTemplate 节点，并插入到参数节点之后
    auto unopt_func = graph_->create(prim::BailoutTemplate)
                          ->insertAfter(graph_->param_node());

    // 设置输出类型为 IntType，以便进行图遍历
    unopt_func->output()->setType(IntType::get());
    // 将未优化图作为子图保存到 unopt_func 节点的属性中
    unopt_func->g_(attr::Subgraph, std::move(unoptimized_graph));
    // 将 unopt_func 的输出作为每个 bailout 点的第一个输入
    for (auto bn : bailouts_) {
      bn->insertInput(0, unopt_func->output());
    }
  }

  // 递归地移除图中的 Guards 节点
  void removeGuards(Block* b) {
    for (auto it = b->nodes().begin(); it != b->nodes().end(); ++it) {
      if (it->kind() == prim::Guard) {
        // 设置输入类型为 TensorType，需要重新进行分析
        it->input()->setType(TensorType::get());
        // 替换 Guard 节点的输出为其输入，并销毁 Guard 节点
        it->output()->replaceAllUsesWith(it->input());
        it.destroyCurrent();
      }

      // 递归处理子块
      for (auto ib : it->blocks()) {
        removeGuards(ib);
      }
    }
  }

  // 将每个 prim::Guard 替换为对应的 prim::BailOut 节点
  void replaceGuardsWithBailouts() {
    for (auto e : replacements_) {
      // 替换 Guard 节点的所有使用为 BailOut 节点
      e.first->replaceAllUsesWith(e.second);
      // 在 BailOut 节点之后插入 e.first 节点
      e.second->node()->insertAfter(e.first->node());
      // 销毁 e.first 节点
      e.first->node()->destroy();
    }
  }

  // 在每个 prim::Guard 点插入 prim::BailOut 节点
  // 每个 BailOut 点获取在特定执行点活跃的输入集合
  // 输入在 Guard/BailOut 点之后计算图的输出时是活跃的
  void insertBailOuts(Block* b) {
    // 实现略，未完整展示
    // 遍历基本块中的节点
    for (auto it = b->nodes().begin(); it != b->nodes().end(); ++it) {
      // 如果节点类型是 prim::Guard
      if (it->kind() == prim::Guard) {
        // 创建一个 prim::BailOut 类型的节点
        auto bailout_node = b->owningGraph()->create(prim::BailOut);
        // 将新创建的 BailOut 节点添加到 bailouts_ 列表中
        bailouts_.push_back(bailout_node);

        // 获取当前 Guard 节点的活跃输入集合
        const auto& live_inputs = liveness_sets_[*it];

        // 将 Guarded 输入添加到 BailOut 节点中
        // 当前，通常只有一个 Guarded 输入
        bailout_node->addInput(it->input());
        for (auto li : live_inputs) {
          // Guarded 输入已经被添加过了
          // 跳过一些应由 BailOutGraphBuilder 直接材料化为 bailout 图的输入
          if (!shouldBeCapturedInByBailOut(li->node()) || li == it->input()) {
            continue;
          }
          // 将符合条件的输入 li 添加到 BailOut 节点中
          bailout_node->addInput(li);
        }

        // 设置 BailOut 节点的输出类型与 Guard 节点的输出类型相同
        bailout_node->output()->setType(it->output()->type());
        // 设置 BailOut 节点的 index 属性为 bailout_index_ 的当前值，并递增 bailout_index_
        bailout_node->i_(attr::index, bailout_index_++);
        // 由于如果它们的参数是 BailOut 节点本身，那么此时无法立即替换节点，因为这样会破坏后续 BailOut 节点的 liveness 集合
        replacements_.insert({it->output(), bailout_node->output()});

      } else {
        // 如果节点类型不是 prim::Guard，则递归处理其内部的每一个块
        for (auto ib : it->blocks()) {
          insertBailOuts(ib);
        }
      }
    }
  }

  // 图的共享指针
  std::shared_ptr<Graph> graph_;
  // 子图映射，将节点映射到其子图中的节点
  std::map<Node*, Node*> subgraphs;
  // BailOut 节点的索引计数器
  std::size_t bailout_index_;
  // 记录节点活跃输入的哈希表
  std::unordered_map<Node*, std::vector<Value*>> liveness_sets_;
  // 存储所有 BailOut 节点的列表
  std::vector<Node*> bailouts_;
  // 替换映射，用于替换节点
  std::map<Value*, Value*> replacements_;
};

// 在给定的图中插入BailOut节点
void InsertBailOuts(std::shared_ptr<Graph> graph) {
  // 创建BailOutInserter对象，使用移动语义传入图对象
  BailOutInserter ibo(std::move(graph));
  // 运行插入BailOut节点的操作
  ibo.run();
}

// 在未优化的图中线性扫描节点，定位匹配给定索引的prim::BailOut节点
static Node* locateBailOutNodeInUnoptimizedGraph(Block* b, int64_t index) {
  // 遍历块b中的所有节点
  for (auto n : b->nodes()) {
    // 如果节点n的类型是prim::BailOut或prim::Guard，并且具有index属性且属性值等于给定的index
    if ((n->kind() == prim::BailOut || n->kind() == prim::Guard) &&
        n->hasAttribute(attr::index) && n->i(attr::index) == index) {
      return n;  // 返回找到的节点n
    }
    // 递归遍历节点n中的子块，寻找匹配的BailOut节点
    for (auto ib : n->blocks()) {
      if (auto bn = locateBailOutNodeInUnoptimizedGraph(ib, index)) {
        return bn;  // 返回找到的节点bn
      }
    }
  }
  return nullptr;  // 如果未找到匹配的节点，返回空指针
}

// 移除prim::BailOut和prim::Guard节点，并将受保护的输入直接连接到其用户
static void removeBailouts(Block* b) {
  // 迭代遍历块b中的所有节点
  for (auto it = b->nodes().begin(); it != b->nodes().end(); it++) {
    if (it->kind() == prim::BailOut || it->kind() == prim::Guard) {
      // 清除节点的分析信息
      it->inputs().at(0)->setType(TensorType::get());
      // 替换节点输出为其输入的第一个
      it->output()->replaceAllUsesWith(it->inputs().at(0));
      // 销毁当前节点
      it.destroyCurrent();
    } else {
      // 递归移除节点it中的子块中的BailOut节点
      for (auto ib : it->blocks()) {
        removeBailouts(ib);
      }
    }
  }
}

// 从原始图中构建BailOut图，参考'bailout_graph.h'
TORCH_API std::shared_ptr<Graph> BuildBailOutGraphFrom(
    int64_t bailout_index,
    const std::shared_ptr<Graph>& orig,
    const std::shared_ptr<Graph>& target) {
  // 在未优化的图orig的块中定位给定bailout_index的BailOut节点
  auto orig_bailout_node =
      locateBailOutNodeInUnoptimizedGraph(orig->block(), bailout_index);

  // 打印调试信息，显示触发BailOut的节点信息
  GRAPH_DEBUG("bailout triggered for ", *orig_bailout_node);
  // 打印原始BailOut图的调试信息
  GRAPH_DUMP("original bailout graph ", orig);
  // 内部断言，确保原始BailOut节点的输入类型不是FunctionType
  TORCH_INTERNAL_ASSERT(
      orig_bailout_node->inputs().at(0)->type()->cast<FunctionType>() ==
      nullptr);
  // 内部断言，确保orig_bailout_node存在且其类型是prim::BailOut或prim::Guard，并且其索引与bailout_index匹配
  TORCH_INTERNAL_ASSERT(
      orig_bailout_node &&
      (orig_bailout_node->kind() == prim::BailOut ||
       orig_bailout_node->kind() == prim::Guard) &&
      bailout_index == orig_bailout_node->i(attr::index));
  // 使用原始图orig和目标图target构建BailOut图的构建器对象
  BailOutGraphBuilderForNode bg(orig, target);
  // 构建从orig_bailout_node开始的BailOut图
  auto bailout_graph = bg.buildBailOutGraphFrom(orig_bailout_node);

  // 移除BailOuts节点
  removeBailouts(bailout_graph->block());
  // 清除BailOut图的分析信息
  ClearProfilingInformation(bailout_graph);
  // 打印BailOut图的调试信息
  GRAPH_DUMP("bailout_graph ", bailout_graph);
  // 返回构建的BailOut图
  return bailout_graph;
}

} // namespace jit
} // namespace torch
```