# `.\pytorch\torch\csrc\jit\passes\graph_fuser.cpp`

```py
// 包含 Torch 的 JIT 编译器 Passes 中的图融合器头文件
#include <torch/csrc/jit/passes/graph_fuser.h>

// 包含 C10 库中的异常处理和范围遍历功能
#include <c10/util/Exception.h>
#include <c10/util/irange.h>

// 包含 Torch 中的代码生成 Fuser 接口和 IR 发射器
#include <torch/csrc/jit/codegen/fuser/interface.h>
#include <torch/csrc/jit/frontend/ir_emitter.h>

// 包含 Torch 中的别名分析和常用子表达式消除 Pass
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/passes/common_subexpression_elimination.h>

// 包含 Torch 中的常量池化和死代码消除 Pass
#include <torch/csrc/jit/passes/constant_pooling.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>

// 包含 Torch 中的 TensorExpr 融合器和子图工具函数
#include <torch/csrc/jit/passes/tensorexpr_fuser.h>
#include <torch/csrc/jit/passes/utils/subgraph_utils.h>

// 包含 Torch 中的自动微分和自定义操作运行时支持
#include <torch/csrc/jit/runtime/autodiff.h>
#include <torch/csrc/jit/runtime/custom_operator.h>
#include <torch/csrc/jit/runtime/operator.h>

// 包含标准库中的队列和无序映射支持
#include <queue>
#include <unordered_map>
#include <utility>

namespace torch {
namespace jit {

namespace {

// 图融合器结构体，用于将图中的节点融合为 FusionGroup
struct GraphFuser {
  using FusionCallback = std::function<bool(GraphFuser*, Node*)>;

  Block* block_;  // 当前操作的基本块
  AliasDb* aliasDb_;  // 别名数据库，用于节点间的别名分析
  std::shared_ptr<Graph> graph_;  // 当前操作的图
  FusionCallback callback_ = [](GraphFuser* gf, Node* n) {
    return gf->isFusableDefault(n, gf->strict_fuser_check_);
  };
  Symbol kind_ = prim::FusionGroup;  // 融合节点的类型
  bool strict_fuser_check_ = false;  // 是否进行严格的融合检查

  // 子图参数限制，用于 CUDA 内核中的参数数量限制
  size_t subgraph_arg_limit_ = 128;

  // 构造函数，初始化融合器
  GraphFuser(AliasDb* aliasDb, Block* block, bool strict_fuser_check)
      : block_(block),
        aliasDb_(aliasDb),
        strict_fuser_check_(strict_fuser_check) {}

  // 自定义 Pass 需要指定类型
  GraphFuser(
      AliasDb* aliasDb,
      Block* block,
      FusionCallback callback,
      Symbol kind,
      bool strict_fuser_check = false)
      : block_(block),
        aliasDb_(aliasDb),
        callback_(std::move(callback)),
        kind_(kind),
        strict_fuser_check_(strict_fuser_check) {}

  // 设置子图参数限制
  void setInputArgLimit(size_t limit) {
    subgraph_arg_limit_ = limit;
  }


  // 将成员变量 subgraph_arg_limit_ 设置为传入的 limit 值
  subgraph_arg_limit_ = limit;
}



  value_list tensorInputs(Node* node) {
    return filter(node->inputs(), [](Value* v) {
      return v->type()->isSubtypeOf(*TensorType::get());
    });
  }


  // 返回节点的输入中，类型为 TensorType 的值列表
  value_list tensorInputs(Node* node) {
    return filter(node->inputs(), [](Value* v) {
      return v->type()->isSubtypeOf(*TensorType::get());
    });
  }



  bool isFusable(Node* node) {
    return callback_(this, node);
  }


  // 调用回调函数 callback_，判断节点是否可融合
  bool isFusable(Node* node) {
    return callback_(this, node);
  }



  bool isFusableDevice(Value* v, bool strict_fuser_check) {
    if (!v->type()->isSubtypeOf(*TensorType::get())) {
      return true;
    }
    auto device = v->type()->expectRef<TensorType>().device();
    if (!device) {
      return !strict_fuser_check;
    }
    if ((*device).is_cpu()) {
      return canFuseOnCPULegacy();
    } else if ((*device).is_cuda()) {
      return canFuseOnGPU();
    } else if ((*device).is_xpu()) {
      return false;
    } else {
      TORCH_CHECK_NOT_IMPLEMENTED(false, "Unknown device for graph fuser");
    }
  }


  // 判断值 v 所在的设备是否支持融合
  bool isFusableDevice(Value* v, bool strict_fuser_check) {
    if (!v->type()->isSubtypeOf(*TensorType::get())) {
      return true;
    }
    auto device = v->type()->expectRef<TensorType>().device();
    if (!device) {
      return !strict_fuser_check;
    }
    if ((*device).is_cpu()) {
      return canFuseOnCPULegacy();
    } else if ((*device).is_cuda()) {
      return canFuseOnGPU();
    } else if ((*device).is_xpu()) {
      return false;
    } else {
      TORCH_CHECK_NOT_IMPLEMENTED(false, "Unknown device for graph fuser");
    }
  }



  // 默认的融合性检查函数，当用户没有传入回调函数时使用
  bool isFusableDefault(Node* node, bool strict_fuser_check) {
    bool fusableDevice = true;
    for (const auto& output : node->outputs()) {
      if (!output->uses().empty()) {
        fusableDevice &= isFusableDevice(output, strict_fuser_check);
      }
    }
    return fusableDevice && isFusableMap(node);
  }


  // 默认的融合性检查函数，当用户没有传入回调函数时使用
  bool isFusableDefault(Node* node, bool strict_fuser_check) {
    bool fusableDevice = true;
    for (const auto& output : node->outputs()) {
      if (!output->uses().empty()) {
        fusableDevice &= isFusableDevice(output, strict_fuser_check);
      }
    }
    return fusableDevice && isFusableMap(node);
  }



  bool isFusableMap(Node* node) {
    // 我们不希望处理跨块节点移动，因为它们不一定是正确的。
    if (node->owningBlock() != block_)
      return false;
    return node->kind() == prim::FusionGroup || isSimpleMap(node);
  }


  // 判断节点是否可以作为融合组的一部分
  bool isFusableMap(Node* node) {
    // 我们不希望处理跨块节点移动，因为它们不一定是正确的。
    if (node->owningBlock() != block_)
      return false;
    return node->kind() == prim::FusionGroup || isSimpleMap(node);
  }



  bool isFusableCatNode(Node* node) {
    if (node->kind() != aten::cat)
      return false;
    if (!node->is_constant(attr::dim))
      return false;

    auto tensors_node = node->namedInput(attr::tensors)->node();
    if ((tensors_node->inputs().size() + node->outputs().size()) >
        subgraph_arg_limit_) {
      return false;
    }
    if (tensors_node->kind() != prim::ListConstruct)
      return false;
    // 注意：从技术上讲，列表的其他用法对我们来说并不是一个大问题。
    // 我们只需将 prim::FusedConcat 放置在 prim::ListConstruct 之前，
    // 并且所有用户都是这个消费者或者在它之后出现的，就能满足所有需求。
    // 然而，我不指望这很快就会发生，因此我们简单地假设我们不必处理它。
    if (tensors_node->output()->uses().size() > 1)
      return false;
    return true;
  }


  // 判断节点是否是可以融合的 cat 节点
  bool isFusableCatNode(Node* node) {
    if (node->kind() != aten::cat)
      return false;
    if (!node->is_constant(attr::dim))
      return false;

    auto tensors_node = node->namedInput(attr::tensors)->node();
    if ((tensors_node->inputs().size() + node->outputs().size()) >
        subgraph_arg_limit_) {
      return false;
    }
    if (tensors_node->kind() != prim::ListConstruct)
      return false;
    // 注意：从技术上讲，列表的其他用法对我们来说并不是一个大问题。
    // 我们只需将 prim::FusedConcat 放置在 prim::ListConstruct 之前，
    // 并且所有用户都是这个消费者或者在它之后出现的，就能满足所有需求。
    // 然而，我不指望这很快就会发生，因此我们简单地假设我们不必处理它。
    if (tensors_node->output()->uses().size() > 1)
      return false;
    return true;
  }



  bool calculatesSize(Node* node) {
    return node->matches("aten::size(Tensor self) -> int[]");
  }


  // 判断节点是否计算了尺寸大小
  bool calculatesSize(Node* node) {
    return node->matches("aten::size(Tensor self) -> int[]");
  }



  bool allUsersAreThisConsumerOrCalcSizes(Node* consumer, Value* producer) {
    auto defining_node = producer->node();
    for (auto o : defining_node->outputs()) {
      for (auto u : o->uses()) {
        if (u.user != consumer && !calculatesSize(u.user))
          return false;
      }
    }
    return true;
  }


  // 检查所有使用者是否都是这个消费者或者计算了尺寸大小
  bool allUsersAreThisConsumerOrCalcSizes(Node* consumer, Value* producer) {
    auto defining_node = producer->node();
    for (auto o : defining_node->outputs()) {
      for (auto u : o->uses()) {
        if (u.user != consumer && !calculatesSize(u.user))
          return false;
      }
    }
    return true;
  }



  Graph& getSubgraph(Node* n) {
    AT_ASSERT(n->kind() == kind_);
    return *n->g(attr::Subgraph);
  }


  // 获取节点 n 的子图
  Graph& getSubgraph(Node* n) {
    AT_ASSERT(n->kind() == kind_);
    return *n->g(attr::Subgraph);
  }



  void mergeFusionGroups(Node* consumer_group, Node* producer_group) {


  // 合并融合组节点，将 producer_group 合并到 consumer_group 中
  void mergeFusionGroups(Node* consumer_group, Node* producer_group) {
    // 现在我们有两个融合组！

    // 恢复融合操作 - 将生产者的所有内部节点放回外部图中。
    std::vector<Node*> temporary_nodes;  // 临时存储节点的容器
    auto producer_subgraph = &getSubgraph(producer_group);  // 获取生产者子图的指针

    // 初始化内部图值到外部图值的映射
    std::unordered_map<Value*, Value*> inner_to_outer;
    auto inner_inputs = producer_subgraph->inputs();  // 获取生产者子图的输入值
    auto outer_inputs = producer_group->inputs();  // 获取生产者组的输入值
    for (const auto i : c10::irange(inner_inputs.size())) {
      inner_to_outer[inner_inputs[i]] = outer_inputs[i];
    }

    // 克隆所有节点
    for (auto inner : producer_subgraph->nodes()) {
      Node* outer = block_->owningGraph()->createClone(
          inner, [&](Value* k) -> Value* { return inner_to_outer.at(k); });  // 使用映射克隆节点
      outer->insertBefore(producer_group);  // 在生产者组之前插入新创建的外部节点
      temporary_nodes.emplace_back(outer);  // 将外部节点添加到临时节点容器中
      auto inner_outputs = inner->outputs();  // 获取内部节点的输出值
      auto outer_outputs = outer->outputs();  // 获取外部节点的输出值
      for (const auto i : c10::irange(inner_outputs.size())) {
        inner_to_outer[inner_outputs[i]] = outer_outputs[i];  // 更新内部到外部值的映射
      }
    }

    // 替换生产者组输出的使用并销毁生产者组
    auto subgraph_outputs = producer_subgraph->outputs();  // 获取生产者子图的输出值
    for (const auto i : c10::irange(subgraph_outputs.size())) {
      auto outer_output = inner_to_outer.at(subgraph_outputs[i]);  // 获取对应外部输出值
      producer_group->outputs()[i]->replaceAllUsesWith(outer_output);  // 替换生产者组的输出使用
      // 新的生产者输出与外部输出具有相同的别名属性
      aliasDb_->replaceWithNewValue(producer_group->outputs()[i], outer_output);
    }
    producer_group->destroy();  // 销毁生产者组
    producer_group =
        nullptr; // 仅用于在某人使用它时获得清晰的错误提示

    // 将临时节点内联到第一个组中
    auto consumer_subgraph = &getSubgraph(consumer_group);  // 获取消费者子图的指针
    for (auto it = temporary_nodes.rbegin(); it != temporary_nodes.rend();
         ++it) {
      Node* node = *it;  // 获取当前节点
      Node* merged = mergeNodeIntoGroup(consumer_group, node);  // 将节点合并到消费者组中
      // 如果任何输出仍在使用，则需要添加它们
      auto outputs = node->outputs();  // 获取节点的输出值
      for (const auto i : c10::irange(outputs.size())) {
        auto output = outputs[i];  // 获取当前输出
        if (output->uses().empty())
          continue;  // 如果输出未被使用则继续下一个循环

        consumer_subgraph->registerOutput(merged->outputs()[i]);  // 注册合并后节点的输出到消费者子图中
        auto new_output = consumer_group->addOutput();  // 添加新的输出到消费者组中
        output->replaceAllUsesWith(new_output);  // 替换所有使用当前输出的地方为新输出
        aliasDb_->replaceWithNewValue(output, new_output);  // 更新别名数据库中的值
        new_output->setType(output->type());  // 设置新输出的类型与原输出相同
      }
      node->destroy();  // 销毁当前节点
    }
  }

  // 将一个生产者节点插入到一个消费者融合组中。
  // 如果 n 是融合组输出的消费者，则无法工作
  // 返回代表节点的组内节点
  Node* mergeNodeIntoGroup(Node* group, Node* n) {
    AT_ASSERT(n->kind() != kind_);  // 断言节点的类型与 kind_ 不相同
    auto& subgraph = getSubgraph(group);  // 获取组的子图引用
    // 从周围图中的节点到融合中的参数的映射
    // 创建一个无序映射表，用于将原始图中的输入值映射到子图中对应的值
    std::unordered_map<Value*, Value*> inputs_map;
    // 初始化计数器 i 和张量插入索引 tensor_insert_idx
    size_t i = 0;
    size_t tensor_insert_idx = 0;
    // 断言：融合组的输入数量应该与子图的输入数量相等
    AT_ASSERT(group->inputs().size() == subgraph.inputs().size());
    // 遍历融合组的输入，将其映射到子图中的对应输入，并在遇到张量类型时更新张量插入索引
    for (auto input : group->inputs()) {
      inputs_map[input] = subgraph.inputs()[i++];
      if (input->type()->isSubtypeOf(*TensorType::get()))
        tensor_insert_idx = i;
    }
    // 设置插入点为子图中的第一个节点
    WithInsertPoint guard(*subgraph.nodes().begin());
    // 遍历当前节点 n 的输入，如果输入尚未映射到融合组的输入中，则根据类型进行不同处理
    for (auto input : n->inputs()) {
      if (inputs_map.count(input) == 0) {
        if (input->type()->isSubtypeOf(*TensorType::get())) {
          // 如果输入是张量类型，插入到指定索引位置，并更新映射和融合组的输入列表
          auto in_group = subgraph.insertInput(tensor_insert_idx);
          in_group->setType(input->type());
          inputs_map[input] = in_group;
          group->insertInput(tensor_insert_idx, input);
          tensor_insert_idx++;
        } else if (
            (input->type()->isSubtypeOf(*FloatType::get()) &&
             input->node()->kind() != prim::Constant) ||
            (n->kind() == aten::_grad_sum_to_size &&
             input->type()->isSubtypeOf(*ListType::ofInts()))) {
          // 如果输入是浮点数类型且不是常量，或者节点类型为 grad_sum_to_size 且输入是整数列表类型，
          // 则将其添加为子图的新输入，并更新映射和融合组的输入列表
          auto in_group = subgraph.addInput();
          in_group->setType(input->type());
          inputs_map[input] = in_group;
          group->addInput(input);
        } else {
          // 对于不支持的情况，通常是传递标量作为融合内核的参数，除非标量是常量，否则会抛出错误
          // 在这种情况下，通过创建常量的克隆节点并将其插入到子图中来处理
          AT_ASSERT(input->node()->kind() == prim::Constant);
          Node* in_const =
              subgraph.createClone(input->node(), [](Value*) -> Value* {
                throw std::runtime_error("unexpected input");
              });
          subgraph.insertNode(in_const);
          inputs_map[input] = in_const->output();
        }
      }
    }
    // 将节点 n 的克隆版本插入到子图中，同时重映射其输入到内部节点
    Node* in_graph = subgraph.createClone(
        n, [&](Value* k) -> Value* { return inputs_map[k]; });
    // 如果节点 n 的输出已经是融合组的输入，则需要将其从融合组中移除，因为 n 现在已经在融合组内部
    //
    // 例如：
    // x = f(w); group(x, y, z) 变为 group(w, y, z)
    // x, y, z = f(w); group(x, y, z) 变为 group(w)
    //
    // 重新映射使用输入的节点到新合并的节点，当融合组为空时，节点 n 不是输入
    auto inputs = group->inputs();
    // 遍历节点 n 的所有输出
    for (size_t i = 0; i < n->outputs().size(); ++i) {
      // 在输入列表中查找当前输出
      auto it = std::find(inputs.begin(), inputs.end(), n->outputs()[i]);
      // 如果找到当前输出在输入列表中
      if (it != inputs.end()) {
        // 计算找到的位置 p
        size_t p = it - inputs.begin();
        // 从 fusion group 中移除对应的输入
        group->removeInput(p);
        // 用 in_graph 的输出替换子图中相应的输入
        subgraph.inputs()[p]->replaceAllUsesWith(in_graph->outputs()[i]);
        // 在子图中删除相应的输入
        subgraph.eraseInput(p);
      }
    }
    // 将 fusion group 插入到子图中
    return subgraph.insertNode(in_graph);
  }

  // 将消费节点 n 转换为只包含 n 的融合组，为融合做准备，并用新组替换对 n 的使用
  Node* createSingletonFusionGroup(Node* n) {
    // 在 owningGraph 中创建一个包含子图的融合组
    auto group = block_->owningGraph()->createWithSubgraph(kind_);
    // 将新节点插入到节点 n 的位置之前
    group->insertBefore(n);
    // 将节点 n 合并到融合组中
    Node* mergedNode = mergeNodeIntoGroup(group, n);
    // 注册融合组的输出
    getSubgraph(group).registerOutput(mergedNode->output());
    // 添加一个输出，并复制元数据
    auto sel = group->addOutput();
    sel->copyMetadata(n->output());
    // 使用新值替换 n 的输出
    aliasDb_->replaceWithNewValue(n->output(), sel);
    // 替换所有对 n 的使用为融合组
    n->replaceAllUsesWith(group);
    // 销毁节点 n
    n->destroy();
    // 返回创建的融合组
    return group;
  }

  // 尝试将生产者节点与消费者节点进行融合
  at::optional<Node*> tryFuse(Node* consumer, Value* producer) {
    // 处理生产者可以被移动到消费者的融合组内的情况
    bool shouldFuse = isFusable(producer->node()) &&
        // 重新排列节点，使生产者的所有使用在消费者之后
        aliasDb_->moveBeforeTopologicallyValid(producer->node(), consumer);

    // 如果不应该融合，返回空
    if (!shouldFuse) {
      return at::nullopt;
    }

    // 检查节点数是否超过子图参数限制
    if ((consumer->inputs().size() + consumer->outputs().size() +
         producer->node()->inputs().size() +
         producer->node()->outputs().size()) > subgraph_arg_limit_) {
      return at::nullopt;
    }

    // 如果消费者节点的类型不是当前的融合类型，创建一个只包含消费者的融合组
    auto group = consumer;
    if (consumer->kind() != kind_) {
      group = createSingletonFusionGroup(consumer);
    }

    // 如果生产者节点的类型是当前的融合类型，将两个融合组进行合并
    if (producer->node()->kind() == kind_) {
      mergeFusionGroups(group, producer->node());
      return group;
    }

    // 断言生产者节点的输出大小为1
    AT_ASSERT(producer->node()->outputs().size() == 1);
    // 将生产者节点合并到融合组中
    Node* merged = mergeNodeIntoGroup(group, producer->node());
    // 如果仍有对生产者的使用，则将它们重新路由到融合组中生成的版本中
    // 如果生产者节点有使用者
    if (!producer->uses().empty()) {
      // 将合并节点的输出注册到子图中
      getSubgraph(group).registerOutput(merged->output());
      // 在组中添加一个新的生产者节点
      Value* new_producer = group->addOutput();
      // 复制生产者节点的元数据到新的生产者节点
      new_producer->copyMetadata(producer);
      // 使用别名数据库将原始生产者节点替换为新的生产者节点
      aliasDb_->replaceWithNewValue(producer, new_producer);
      // 替换所有使用原始生产者节点的节点为新的生产者节点
      producer->replaceAllUsesWith(new_producer);
    }
    // 销毁原始生产者节点
    producer->node()->destroy();
    // 返回合并后的组节点
    return group;
  }

  // 判断是否可以通过重用现有的融合块来融合块节点
  bool canFuseChunk(Node* consumer, Value* producer) {
    // 如果消费者节点不是融合组节点，则返回 false
    if (consumer->kind() != prim::FusionGroup) {
      return false;
    }
    // 获取生产者节点的类型
    auto* chunk = producer->node();
    // 如果生产者节点不是常量块节点，则返回 false
    if (chunk->kind() != prim::ConstantChunk)
      return false;
    // 遍历块节点的所有输出
    for (auto s : chunk->outputs()) {
      // 遍历每个输出的所有使用者
      for (auto u : s->uses()) {
        // 如果使用者不是当前消费者节点，则返回 false
        if (u.user != consumer) {
          return false;
        }
      }
    }
    // 如果块节点的块数为 1，则返回 false，建议删除该节点而不是融合
    if (chunk->i(attr::chunks) == 1) {
      return false;
    }
    // 否则返回 true，表示可以融合块节点
    return true;
  }

  // 查找融合组中与给定输入关联的常量块节点
  std::optional<Node*> findFusedChunk(Node* group, Value* input) {
    // 断言当前组节点是融合组节点
    AT_ASSERT(group->kind() == prim::FusionGroup);
    // 在组节点的输入中查找指定的输入值
    auto it = std::find(group->inputs().begin(), group->inputs().end(), input);
    // 如果未找到，返回空的可选值
    if (it == group->inputs().end()) {
      return c10::nullopt;
    }
    // 获取输入在子图中的索引
    size_t input_index = it - group->inputs().begin();
    // 获取子图对象
    auto& subgraph = getSubgraph(group);
    // 获取子图中指定索引的输入值
    auto* subgraph_input = subgraph.inputs().at(input_index);
    // 如果子图输入值的使用次数为 1，并且使用者是常量块节点，则返回该节点
    auto* node = subgraph_input->uses().at(0).user;
    if (node->kind() == prim::ConstantChunk) {
      AT_ASSERT(subgraph_input->uses().size() == 1);
      return node;
    }
    // 否则返回空的可选值
    return c10::nullopt;
  }

  // 通过重用现有的融合块节点来融合块节点
  void fuseChunkByReusingExistingFusedChunk(
      Node* group,
      Node* chunk,
      Node* existingFusedChunk) {
    // 如果块节点的输出数量与现有融合块节点的输出数量不相等，则返回
    if (chunk->outputs().size() != existingFusedChunk->outputs().size()) {
      return;
    }
    // 获取组节点的子图对象
    auto& subgraph = getSubgraph(group);
    // 遍历块节点的所有输出
    for (size_t i = 0; i < chunk->outputs().size(); ++i) {
      // 查找融合组节点中与块节点输出对应的输入值
      auto* replacement_val = existingFusedChunk->outputs().at(i);
      auto* val = chunk->outputs().at(i);
      auto it = std::find(group->inputs().begin(), group->inputs().end(), val);
      auto input_index = it - group->inputs().begin();

      // 将子图中相应索引处的输入替换为替换值
      auto group_input = subgraph.inputs().at(input_index);
      group_input->replaceAllUsesWith(replacement_val);

      // 移除组节点中相应索引处的输入
      group->removeInput(input_index);
      // 在子图中删除相应索引处的输入
      subgraph.eraseInput(input_index);
    }
    chunk->destroy();
  }

  // prim::ConstantChunk有两个不变量：
  // (1) prim::ConstantChunk的张量输入必须是融合组的输入
  // (2) 同一融合组中的两个ConstantChunk不能共享张量输入
  graph_node_list::iterator fuseChunk(Node* consumer, Value* producer) {
    auto* chunk = producer->node();
    AT_ASSERT(consumer->kind() == prim::FusionGroup);
    AT_ASSERT(chunk->kind() == prim::ConstantChunk);

    // 如果生产者的输入已经是prim::ConstantChunk节点的输入，则由于不变量(2)，无法添加新的prim::ConstantChunk节点。
    auto* chunked_tensor = producer->node()->input();
    if (auto existingFusedChunk = findFusedChunk(consumer, chunked_tensor)) {
      fuseChunkByReusingExistingFusedChunk(
          consumer, chunk, *existingFusedChunk);
      return consumer->reverseIterator();
    }

    // 将prim::ConstantChunk移入融合组
    mergeNodeIntoGroup(consumer, chunk);
    chunk->destroy();
    return consumer->reverseIterator();
  }

  // 根据反向拓扑顺序对值进行排序
  value_list sortReverseTopological(ArrayRef<Value*> inputs) {
    value_list result;
    for (auto i : inputs) {
      if (i->node()->owningBlock() == block_) {
        result.push_back(i);
      }
    }
    // 按照反向拓扑顺序排序
    std::sort(result.begin(), result.end(), [&](Value* a, Value* b) {
      return a->node()->isAfter(b->node());
    });
    return result;
  }

  // 扫描节点以寻找ConstantChunk
  graph_node_list::iterator scanNodeForChunks(Node* consumer) {
    if (consumer->kind() == prim::FusionGroup) {
      auto inputs = sortReverseTopological(consumer->inputs());
      for (auto producer : inputs) {
        if (!canFuseChunk(consumer, producer)) {
          continue;
        }
        return fuseChunk(consumer, producer);
      }
    }
    return ++consumer->reverseIterator();
  }

  // 广播张量
  at::ArrayRef<Value*> broadcast_tensors(value_list inputs) {
    AT_ASSERT(!inputs.empty());
    auto* g = inputs[0]->owningGraph();
    auto* input_list =
        g->insertNode(g->createList(TensorType::get(), inputs))->output();
    aliasDb_->createValue(input_list);
    auto* output_list = g->insert(aten::broadcast_tensors, {input_list});
    aliasDb_->createValue(output_list);
    auto* unpack_node = g->insertNode(
        g->create(prim::ListUnpack, {output_list}, inputs.size()));

    // 我们正在做以下操作：
    //   input_list = listConstruct(a, b, ...)
    //   output_list = broadcast_tensors(input_list)
    //   a_broadcasted, b_broadcasted = listUnpack(output_list)
    // `a_broadcasted`应该接收与`a`相同的别名信息
    TORCH_INTERNAL_ASSERT(unpack_node->outputs().size() == inputs.size());
    for (const auto i : c10::irange(inputs.size())) {
      Value* original_input = inputs[i];
      Value* broadcasted_output = unpack_node->outputs()[i];
      aliasDb_->copyValue(original_input, broadcasted_output);
    }

    return unpack_node->outputs();
  }

  void insertExplicitBroadcast(Node* node) {
    // 在节点之前插入插入点保护
    WithInsertPoint insert_guard{node};
    // 获取节点的张量输入
    auto tensors = tensorInputs(node);
    // 对张量进行广播操作，生成新的张量
    auto new_tensors = broadcast_tensors(std::move(tensors));

    // 替换节点的张量输入为广播后的值
    auto new_tensors_it = new_tensors.begin();
    for (size_t i = 0; i < node->inputs().size(); ++i) {
      // 如果节点的输入是张量类型
      if (node->inputs()[i]->type()->isSubtypeOf(*TensorType::get())) {
        // 确保新张量迭代器不超出范围
        AT_ASSERT(new_tensors_it != new_tensors.end());
        // 替换节点的输入为新的张量
        node->replaceInput(i, *(new_tensors_it++));
      }
    }
  }

  // 将常量分块节点提升为广播分块节点
  Node* promoteChunkToBroadcastingChunk(Node* chunk) {
    // 确保节点是常量分块节点
    AT_ASSERT(chunk->kind() == prim::ConstantChunk);

    // 获取分块数量
    size_t nchunks = chunk->i(attr::chunks);
    // 在当前图中创建广播分块节点
    Node* bchunk =
        chunk->owningGraph()->create(prim::BroadcastingChunk, nchunks);
    // 将原始节点的输入作为广播分块节点的输入
    bchunk->addInput(chunk->input());
    // 遍历每个分块
    for (const auto i : c10::irange(nchunks)) {
      auto* old_output = chunk->outputs().at(i);
      auto* new_output = bchunk->outputs().at(i);
      // 复制旧输出的元数据到新输出
      new_output->copyMetadata(old_output);
      // 使用新值替换别名数据库中的旧输出
      aliasDb_->replaceWithNewValue(old_output, new_output);
      // 替换所有使用旧输出的地方为新输出
      old_output->replaceAllUsesWith(new_output);
    }
    // 复制节点的属性到广播分块节点
    bchunk->copyAttributes(*chunk);
    // 在原始节点之后插入广播分块节点
    bchunk->insertAfter(chunk);
    // 销毁原始节点
    chunk->destroy();
  // 尝试将 chunk 节点移动到消费者操作之前，以便进行融合
  bool tryToMoveChunk(Node* consumer, Value* producer) {
    // 检查生产者是否来自 chunk/bchunk 节点
    auto* chunk = producer->node();
    // 如果不是 chunk 节点，则无需移动，返回 false
    if (!chunk->kind().is_aten() || chunk->kind() != ::c10::aten::chunk && chunk->kind() != ::c10::aten::bchunk)
      return false;
    // 确定 chunk 节点的输出数量是否超过一个
    if (chunk->outputs().size() != 2)
      return false;
    // 检查是否可以将 chunk 节点融合到消费者节点中
    if (consumer->kind().is_prim() && consumer->kind() == ::c10::prim::FusionGroup)
      return true;
    // 如果消费者节点不支持融合，则返回 false
    if (consumer->kind().is_aten() || consumer->kind() != ::c10::aten::add)
      return false;
    // 移动 chunk 节点以确保操作的正确性和一致性
    return true;
  }
    // 检查 chunk 的类型是否为 ConstantChunk 或 BroadcastingChunk，若不是则返回 false
    if (chunk->kind() != prim::ConstantChunk &&
        chunk->kind() != prim::BroadcastingChunk)
      return false;

    // 尝试查找一个生产者节点，使其可以在 chunk/bchunk 之后进行融合。生产者节点必须能够被消费者节点所融合。
    auto it = std::find_if(
        chunk->inputs().begin(),
        chunk->inputs().end(),
        [&](Value* producer_for_chunk) {
          return isFusableMap(producer_for_chunk->node()) &&
              allUsersAreThisConsumerOrCalcSizes(chunk, producer_for_chunk);
        });
    // 如果找不到符合条件的生产者节点，则返回 false
    if (it == chunk->inputs().end()) {
      return false;
    }
    // 获取符合条件的生产者节点
    Value* producer_for_chunk = *it;
    size_t producer_index = it - chunk->inputs().begin();

    // 确保 chunk 的所有使用者都在当前消费者节点中
    for (auto s : chunk->outputs()) {
      for (auto u : s->uses()) {
        // 如果使用者不是当前消费者节点，则返回 false
        if (u.user != consumer)
          return false;
      }
    }
    // 断言生产者节点的输出数量为 1
    Node* producer_for_chunk_node = producer_for_chunk->node();
    AT_ASSERT(producer_for_chunk_node->outputs().size() == 1);

    // 如果 chunk 不是 BroadcastingChunk，则将其升级为 BroadcastingChunk。BroadcastingChunk 表示广播操作和一个或多个 chunk 操作。
    auto* bchunk = chunk;
    if (chunk->kind() == prim::ConstantChunk) {
      bchunk = promoteChunkToBroadcastingChunk(chunk);
    }
    // 获取 bchunk 的 chunk 数量
    size_t nchunks = bchunk->i(attr::chunks);
    // 在 bchunk->next() 位置插入节点
    WithInsertPoint guard(bchunk->next());

    // 存储生产者 chunk 输出的向量
    std::vector<Value*> producer_chunk_outputs;
    for (const auto i : c10::irange(nchunks)) {
      // 将 bchunk 的输出值添加到 producer_chunk_outputs 中
      producer_chunk_outputs.push_back(
          bchunk->output(nchunks * producer_index + i));
    }

    // 添加每个操作的操作数到 bchunk 节点中
    // chunked_inputs[input_nr][chunk_output_idx] 表示 inputs[input_nr] 的第 chunk_output_idx 个输出的节点
    std::vector<std::vector<Value*>> chunked_inputs;
    // 遍历生产者节点的每一个输入
    for (auto input : producer_for_chunk_node->inputs()) {
      // XXX: 在此处我们只处理逐点运算，因此我们知道仅通过张量参数推送连接是有效的，
      // 因此可以安全地忽略所有其他参数。
      // 检查输入是否为张量类型
      if (!input->type()->isSubtypeOf(*TensorType::get()))
        continue;

      // 如果 'input' 已经是 bchunk 的输入之一，则重用它。
      auto bchunk_inputs = bchunk->inputs();
      auto it = std::find(bchunk_inputs.begin(), bchunk_inputs.end(), input);
      if (it != bchunk_inputs.end()) {
        // 如果找到了，为每个分块添加对应的输出
        chunked_inputs.emplace_back();
        auto input_index = std::distance(bchunk_inputs.begin(), it);
        for (const auto chunki : c10::irange(nchunks)) {
          chunked_inputs.back().push_back(
              bchunk->outputs().at(nchunks * input_index + chunki));
        }
        continue;
      }

      // NB: 我决定不在这里使用 cloneFrom，因为如果将来 cloneFrom 复制选择节点，
      // 这绝对不是你想要的（选择节点有不同的类型）。
      // TODO: 或许现在我们应该使用 cloneFrom，因为现在重构后，Value 与 Node 不同。
      // 向 bchunk 添加输入节点
      bchunk->addInput(input);
      chunked_inputs.emplace_back(); // 不幸的是，这里不是 C++17
      for (auto chunk_sel : producer_chunk_outputs) {
        // 为广播块节点的每个输出元素添加一个新值。这是安全的，因为它只会被分块操作消耗。
        Value* input_chunk_sel = bchunk->addOutput();
        input_chunk_sel->setType(chunk_sel->type());
        aliasDb_->createValue(input_chunk_sel);
        chunked_inputs.back().push_back(input_chunk_sel);
      }
    }

    // 对每个块的操作应用操作，然后重写图形以使用它们！
    for (auto chunk_sel : producer_chunk_outputs) {
      auto original_inputs = producer_for_chunk_node->inputs();
      // 创建一个分块操作节点
      Node* chunked_op =
          block_->owningGraph()->create(producer_for_chunk_node->kind());
      chunked_op->copyAttributes(*producer_for_chunk_node);
      chunked_op->output()->setType(chunk_sel->type());
      auto chunked_inputs_it = chunked_inputs.begin();
      for (Value* original_input : original_inputs) {
        if (original_input->type()->isSubtypeOf(*TensorType::get())) {
          // 断言确保分块输入不为空
          AT_ASSERT(chunked_inputs_it != chunked_inputs.end());
          chunked_op->addInput(
              // NOLINTNEXTLINE(clang-analyzer-core.DivideZero)
              chunked_inputs_it->at(chunk_sel->offset() % nchunks));
          ++chunked_inputs_it;
        } else {
          chunked_op->addInput(original_input);
        }
      }
      // 插入节点到 bchunk 的图形中
      bchunk->owningGraph()->insertNode(chunked_op);
      // 替换 chunk_sel 的所有用途为 chunked_op 的输出
      chunk_sel->replaceAllUsesWith(chunked_op->output());
      aliasDb_->replaceWithNewValue(chunk_sel, chunked_op->output());
    }

    // 从 bchunk 中移除生产者索引的输入
    bchunk->removeInput(producer_index);
    // 遍历 nchunks 范围内的所有索引 i
    for (const auto i : c10::irange(nchunks)) {
      (void)i; // 忽略未使用变量的警告
      // 删除 bchunk 中第 nchunks * producer_index 个位置的输出
      bchunk->eraseOutput(nchunks * producer_index);
    }

    // producer_for_chunk_node 的输出可能在一些 aten::size 操作中被使用，
    // 因此我们需要清理这些使用情况（我们简单地将其张量输入广播处理）。
    // 我们需要在图中尽早插入这些操作，即在 producer_for_chunk_node 之后，
    // 因为我们可能会有 _size_if_not_same 在 bchunk 之前的情况。
    WithInsertPoint guard2(producer_for_chunk_node);
    // 获取 producer_for_chunk_node 输出的使用情况
    auto size_calc_uses = producer_for_chunk_node->output()->uses();
    if (!size_calc_uses.empty()) {
      // 过滤出张量类型的输入
      auto tensor_inputs = filter(
          producer_for_chunk_node->inputs(),
          [](Value* v) { return v->type()->isSubtypeOf(*TensorType::get()); });
      // 对每个张量输入执行尺寸计算
      auto tensor_sizes = fmap(tensor_inputs, [&](Value* v) {
        // 在图中插入 aten::size 操作，并创建对应的别名
        Value* output = v->owningGraph()->insert(aten::size, {v});
        aliasDb_->createValue(output);
        return output;
      });
      AT_ASSERT(!tensor_sizes.empty());
      // 如果张量尺寸计算结果大于 1 个，则执行尺寸广播
      Value* output_size = tensor_sizes.size() == 1
          ? tensor_sizes[0]
          : broadcastSizes(tensor_sizes, aliasDb_);
      // 替换尺寸计算的使用情况为新的输出尺寸，并销毁原使用
      for (Use u : size_calc_uses) {
        u.user->output()->replaceAllUsesWith(output_size);
        u.user->destroy();
      }
    }
    // 销毁 producer_for_chunk_node 节点
    producer_for_chunk_node->destroy();
    // 返回操作成功标志
    return true;
  }

  // 扫描节点，返回继续扫描的位置和是否进行了融合的标志
  std::pair<graph_node_list::iterator, bool> scanNode(Node* consumer) {
    // 如果 consumer 节点可以融合
    if (isFusable(consumer)) {
      // 按照逆拓扑顺序处理输入节点
      auto inputs = sortReverseTopological(consumer->inputs());
      for (auto producer : inputs) {
        // 尝试移动块以实现融合
        if (tryToMoveChunk(consumer, producer)) {
          // 块在 consumer 之前被重新排列以允许融合，重新扫描 consumer 节点以执行融合
          return std::make_pair(consumer->reverseIterator(), true);
        }
        // 尝试融合 producer 和 consumer 节点
        auto fusion_group = tryFuse(consumer, producer);
        if (fusion_group) {
          // 融合后，consumer 移动到 FusionGroup 中，inputs 不再有效，因此重新扫描 FusionGroup 进行更多融合
          return std::make_pair(fusion_group.value()->reverseIterator(), true);
        }
      }
    }
    // 返回继续扫描的位置和未进行融合的标志
    return std::make_pair(++consumer->reverseIterator(), false);
  }

  // 替换中间广播块
    // 从最后一个节点开始逆向遍历当前块的所有节点
    for (auto it = block_->nodes().rbegin(); it != block_->nodes().rend();) {
      auto* node = *it;
      ++it; // 因为可能会删除节点，所以现在递增迭代器。
      // 如果节点类型不是 prim::BroadcastingChunk，则跳过当前节点继续下一个节点的处理
      if (node->kind() != prim::BroadcastingChunk) {
        continue;
      }
      auto* bchunk = node;
      // 插入显式广播操作到当前节点 bchunk
      insertExplicitBroadcast(bchunk);

      auto* graph = block_->owningGraph();
      size_t nchunks = bchunk->i(attr::chunks);
      // 在 bchunk->next() 插入点创建 WithInsertPoint 保护块
      WithInsertPoint guard(bchunk->next());

      // 将 bchunk 拆分成 bchunk.inputs().size() 个 chunk 节点
      for (size_t input_offset = 0; input_offset < bchunk->inputs().size();
           input_offset++) {
        auto* input = bchunk->inputs().at(input_offset);

        // 创建新的 chunk 节点并插入到图中
        Node* new_chunk =
            graph->insertNode(graph->create(prim::ConstantChunk, input, 0));
        new_chunk->copyAttributes(*bchunk);
        // 复制所有输出端元数据并进行替换
        for (const auto output_offset : c10::irange(nchunks)) {
          auto new_output = new_chunk->addOutput();
          auto old_output =
              bchunk->outputs().at(input_offset * nchunks + output_offset);
          new_output->copyMetadata(old_output);
          aliasDb_->replaceWithNewValue(old_output, new_output);
          old_output->replaceAllUsesWith(new_output);
        }
      }
      // 销毁原始的 bchunk 节点
      bchunk->destroy();
    }
  }

  // 构建表达式，计算融合组合的所有中间值（和输出）的形状，基于输入的大小。应运行 DCE 以删除未使用的表达式。
  std::unordered_map<Value*, Value*> buildShapeExpressions(Node* fusion_group) {
    // 在 fusion_group->next() 插入点创建 WithInsertPoint 保护
    WithInsertPoint insert_guard{fusion_group->next()};
    // 创建形状映射表，将中间值的形状映射到其计算表达式
    std::unordered_map<Value*, Value*> shape_of;

    Graph* graph = fusion_group->owningGraph();
    // 获取子图并输入
    auto subgraph = fusion_group->g(attr::Subgraph);

    auto inputs = fusion_group->inputs();
    auto sinputs = subgraph->inputs();
    AT_ASSERT(inputs.size() == sinputs.size());
    // 遍历输入，如果是张量类型，则插入尺寸操作并创建别名
    for (const auto i : c10::irange(inputs.size())) {
      if (inputs[i]->type()->isSubtypeOf(*TensorType::get())) {
        Value* soutput = graph->insert(aten::size, {inputs[i]});
        aliasDb_->createValue(soutput);
        shape_of[sinputs[i]] = soutput;
      }
    }

    // 当有保证输出不会被移除时，因为它在不涉及尺寸检查的表达式中使用，可以使用其尺寸
    auto outputs = fusion_group->outputs();
    auto soutputs = subgraph->outputs();
    AT_ASSERT(outputs.size() == soutputs.size());
    // 遍历输出，如果仅在尺寸检查中使用，则跳过
    for (const auto i : c10::irange(outputs.size())) {
      if (usedOnlyInSize(outputs[i]))
        continue;
      // 插入尺寸操作并创建别名
      Value* soutput = graph->insert(aten::size, {outputs[i]});
      aliasDb_->createValue(soutput);
      shape_of[soutputs[i]] = soutput;
    }
    // 遍历子图中的每个节点
    for (Node* n : subgraph->nodes()) {
      // XXX: 使用 shape_of.emplace 是为了优化输出形状！
      // 当节点类型为 prim::FusedConcat 时，跳过处理，因为这种情况较为复杂，
      // 我们必须考虑输入具有不同形状的情况，但这些张量始终作为输出，因此我们可以简单地避免替换它们的查询，因为这不会对我们有所帮助。
      if (n->kind() == prim::FusedConcat) {
        continue;
      }
      // 当节点类型为 prim::Constant 时，跳过处理
      if (n->kind() == prim::Constant) {
        continue;
      }
      // 当节点类型为 prim::ConstantChunk 时，执行以下操作
      if (n->kind() == prim::ConstantChunk) {
        // 创建 ChunkSizes 节点，并设置其属性
        Node* sizes_node = graph->insertNode(
            graph->create(prim::ChunkSizes, shape_of.at(n->input()), 2));
        sizes_node->i_(attr::dim, n->i(attr::dim));
        sizes_node->i_(attr::chunks, n->i(attr::chunks));
        // 为 sizes_node 的每个输出创建别名
        for (Value* output : sizes_node->outputs()) {
          aliasDb_->createValue(output);
        }
        // 获取 regular_size 和 last_size 的值
        Value* regular_size = sizes_node->outputs().at(0);
        Value* last_size = sizes_node->outputs().at(1);
        // 设置 regular_size 和 last_size 的类型
        regular_size->setType(ListType::ofInts());
        last_size->setType(ListType::ofInts());
        // 获取当前节点的所有输出，并更新 shape_of 中的映射关系
        auto outputs = n->outputs();
        for (Value* o : outputs.slice(0, outputs.size() - 1)) {
          shape_of.emplace(o, regular_size);
        }
        shape_of.emplace(outputs.at(outputs.size() - 1), last_size);
        continue;
      }
      // 过滤出节点 n 的张量输入
      auto tensor_inputs = filter(n->inputs(), [](Value* v) {
        return v->type()->isSubtypeOf(*TensorType::get());
      });
      // 获取 tensor_inputs 对应的形状，并进行广播操作
      auto shapes =
          fmap(tensor_inputs, [&](Value* v) { return shape_of.at(v); });
      AT_ASSERT(!shapes.empty());
      // 将节点 n 的输出形状加入 shape_of 中
      shape_of.emplace(
          n->output(),
          shapes.size() == 1 ? shapes[0] : broadcastSizes(shapes, aliasDb_));
    }
    // 返回构建好的 shape_of 映射表
    return shape_of;
  }

  // 从融合组中移除仅在大小计算中使用的输出
  void removeOutputsUsedOnlyInSize(Node* fusion_group) {
    // 如果 fusion_group 的类型不是 prim::FusionGroup，则直接返回
    if (fusion_group->kind() != prim::FusionGroup)
      return;
    // 获取 fusion_group 的子图
    auto subgraph = fusion_group->g(attr::Subgraph);

    // 构建 shape_of 映射表
    auto shape_of = buildShapeExpressions(fusion_group);
    // 获取 fusion_group 和 subgraph 的输出向量
    auto outputs = fusion_group->outputs().vec();
    auto soutputs = subgraph->outputs().vec();
    // XXX: 反向迭代这个顺序不仅有助于性能！它对正确性至关重要（i 必须反映出当前输出的真实索引）！
    // 对于每一个输出，检查是否仅在大小计算中使用，并且 shape_of 中存在对应的 soutput
    for (int64_t i = static_cast<int64_t>(outputs.size()) - 1; i >= 0; --i) {
      auto output = outputs[i];
      auto soutput = soutputs[i];
      // 如果输出仅在大小计算中使用，并且 shape_of 中包含 soutput
      if (usedOnlyInSize(output) && shape_of.count(soutput) > 0) {
        // 替换所有使用 output 的 aten::size 节点为 shape_of.at(soutput)
        auto uses = output->uses();
        for (Use u : uses) {
          AT_ASSERT(u.user->matches("aten::size(Tensor self) -> int[]"));
          u.user->output()->replaceAllUsesWith(shape_of.at(soutput));
          u.user->destroy();
        }
        // 从 fusion_group 和 subgraph 中移除输出 i
        fusion_group->eraseOutput(i);
        subgraph->eraseOutput(i);
      }
    }
  }

  // 检查 producer 是否可以与 concat 融合
  bool canFuseWithConcat(Value* producer, Node* before_check) {
    // 如果 producer 所属节点不可融合，则返回 false
    if (!isFusable(producer->node())) {
      return false;
    }
    // 检查是否可以在拓扑上移动节点之前进行此检查，确保节点匹配并且不是特殊节点如prim::Param
    if (!aliasDb_->couldMoveBeforeTopologically(
            producer->node(), before_check)) {
      return false;
    }

    // 如果核心参数的数量可能超过限制，则跳过
    if ((before_check->inputs().size() + before_check->outputs().size() +
         producer->node()->inputs().size() +
         producer->node()->outputs().size()) > subgraph_arg_limit_) {
      return false;
    }

    // 只有当融合组产生的值不是来自concat时，融合组才能与concat的组合合并
    if (producer->node()->kind() == prim::FusionGroup) {
      auto subgraph = producer->node()->g(attr::Subgraph);
      auto* node = subgraph->outputs().at(producer->offset())->node();
      return node->kind() != prim::FusedConcat;
    }
    return true;
  }

  // 创建融合的concat节点
  Node* createFusedConcat(Node* node) {
    AT_ASSERT(node->kind() == aten::cat);

    Graph* graph = node->owningGraph();
    Node* list_construct = node->namedInput(attr::tensors)->node();
    int64_t dim = node->get<int64_t>(attr::dim).value();

    Node* fused_cat = graph->create(prim::FusedConcat, list_construct->inputs())
                          ->i_(attr::dim, dim);
    fused_cat->insertBefore(list_construct);
    fused_cat->output()->copyMetadata(node->output());
    aliasDb_->copyValue(node->output(), fused_cat->output());

    // 删除原始图中的融合的concat节点
    return createSingletonFusionGroup(fused_cat);
  }

  // 合并concat节点
  void fuseConcats() {
    for (auto it = block_->nodes().rbegin(); it != block_->nodes().rend();
         ++it) {
      Node* cat = *it;
      if (!isFusableCatNode(cat)) {
        continue;
      }
      Node* list_construct = cat->namedInput(attr::tensors)->node();
      Node* fused_cat = createFusedConcat(cat);
      Value* fused_cat_out = fused_cat->output();

      auto sorted_inputs = sortReverseTopological(fused_cat->inputs());
      size_t input_idx = 0;
      bool any_fused = false;
      while (input_idx < sorted_inputs.size()) {
        Value* input = sorted_inputs[input_idx++];
        if (!canFuseWithConcat(input, fused_cat)) {
          continue;
        }
        any_fused = true;
        auto maybe_group = tryFuse(fused_cat, input);
        AT_ASSERT(maybe_group && maybe_group == fused_cat);
        // 当执行此融合时，可能会破坏多个输入，因此必须重新计算列表并重新遍历
        sorted_inputs = sortReverseTopological(fused_cat->inputs());
        input_idx = 0;
      }

      if (any_fused) {
        cat->output()->replaceAllUsesWith(fused_cat_out);
        it.destroyCurrent();
        if (list_construct->output()->uses().empty()) {
          list_construct->destroy();
        }
      } else {
        fused_cat->destroy();
      }
  }
}
// TODO: old fuser is not maintained internally, somewhere it is being turned on
// inadvertently for certain workflows. make this a no-op until we identify
// location
#if defined(FBCODE_CAFFE2)
    return;
#endif

    // Run the pass until no changes are made.
    // This is necessary, because the algorithm can miss out on certain fusion
    // opportunities if ran only once. Consider this graph:
    //
    // %1 = f(...)
    // %2 = g(%1)
    // %3 = h(%1)
    // %4 = l(%3)
    // return (%4, %2)
    //
    // where f, g, h, l are simple map ops.
    // The first iteration will fuse %4 and %3, and see that %1 is an input, but
    // can't be fused, because it has a different use before the fusion group
    // in our topological ordering. Then, %2 will be considered, and fused with
    // %1. If we do another iteration, the algorithm will consider the fusion of
    // these two groups and fix the situation.
    bool any_changed = true;
    while (any_changed) {
      any_changed = false;
      // Iterate through the nodes of the block in reverse order
      for (auto it = block_->nodes().rbegin(); it != block_->nodes().rend();) {
        // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
        bool changed;
        // Apply a transformation to the current node and check if it changed
        std::tie(it, changed) = scanNode(*it);
        any_changed |= changed;
      }
    }

    // Fuse concatenation operations within the block
    fuseConcats();

    // Optimize fused sub-graphs within the block
    optimizeFusedGraphs();

    // The graph fuser can add intermediate prim::BroadcastingChunk nodes.
    // Replace these with separate broadcasting and chunk operations.
    replaceIntermediateBroadcastingChunks();

    // Fuse starting chunks into the fusion group within the block
    for (auto it = block_->nodes().rbegin(); it != block_->nodes().rend();) {
      // Scan each node to find starting chunks and integrate them
      it = scanNodeForChunks(*it);
    }

    // Remove outputs from nodes that are only used for determining sizes
    for (Node* n : block_->nodes()) {
      removeOutputsUsedOnlyInSize(n);
    }

    // Recursively apply peephole optimization to nested blocks within nodes
    for (Node* node : block_->nodes()) {
      for (Block* sub_block : node->blocks()) {
        GraphFuser(aliasDb_, sub_block, callback_, kind_, strict_fuser_check_)
            .run();
      }
    }
  }
};

// Perform peephole optimization of shape expressions within a given block
void PeepholeOptimizeShapeExpressions(Block* block, AliasDb* db) {
  // Obtain a list of nodes within the block
  auto nodes = block->nodes();
  // Iterate through each node in the block
  for (auto it = nodes.begin(); it != nodes.end(); ++it) {
    Node* node = *it;
    // Recursively optimize shape expressions within nested blocks
    for (Block* subblock : node->blocks()) {
      PeepholeOptimizeShapeExpressions(subblock, db);
    }
    // 如果节点的类型是 prim::BroadcastSizes
    if (node->kind() == prim::BroadcastSizes) {
      // 移除没有操作效果的广播
      if (node->inputs().size() == 1) {
        // 将节点的输出替换为其输入，并销毁当前节点
        node->output()->replaceAllUsesWith(node->input());
        it.destroyCurrent();
        continue;
      }
      // 去除重复的输入，但使用它们的 unique() 值确保此过程仅依赖于图的结构
      std::map<size_t, Value*> unique_to_value;
      for (Value* input : node->inputs()) {
        unique_to_value.emplace(input->unique(), input);
      }
      // 如果去重后的输入数量不等于原始输入数量
      if (unique_to_value.size() != node->inputs().size()) {
        std::vector<Value*> inputs;
        inputs.reserve(unique_to_value.size());
        // 将去重后的值放入 inputs 中
        for (auto& entry : unique_to_value) {
          inputs.push_back(entry.second);
        }
        // 如果输入只剩下一个，则将节点的输出替换为该输入
        if (inputs.size() == 1) {
          node->output()->replaceAllUsesWith(inputs[0]);
        } else {
          // 在节点插入点上下文中，将节点的输出替换为广播后的结果
          WithInsertPoint insert_guard{node};
          node->output()->replaceAllUsesWith(broadcastSizes(inputs, db));
        }
        it.destroyCurrent();
        --it; // 重新访问具有去重输入的节点
        continue;
      }
      // 将简单的广播链合并为单一节点
      const auto& uses = node->output()->uses();
      if (uses.size() == 1 && uses[0].user->kind() == prim::BroadcastSizes) {
        // 获取使用节点
        Node* user = uses[0].user;
        // 移除使用节点中的输出
        user->removeInput(uses[0].offset);
        // 将当前节点的输入添加到使用节点中
        for (Value* i : node->inputs()) {
          user->addInput(i);
        }
        // 销毁当前节点
        it.destroyCurrent();
      }
    }
}

} // anonymous namespace
// 定义一个静态变量，用于标记是否启用了旧版 CPU 融合
static bool cpu_fuser_enabled_legacy = false;

// 返回旧版 CPU 融合是否可用的状态
bool canFuseOnCPULegacy() {
  return cpu_fuser_enabled_legacy;
}

// 设置旧版 CPU 融合的状态
void overrideCanFuseOnCPULegacy(bool value) {
  cpu_fuser_enabled_legacy = value;
}

// 对图进行融合操作
void FuseGraph(std::shared_ptr<Graph>& graph, bool strict_fuser_check) {
  // 创建图的别名数据库
  AliasDb db(graph);
  // 创建图融合器对象，传入别名数据库和图的块
  GraphFuser(&db, graph->block(), strict_fuser_check).run();
  // 对别名数据库进行静态检查
  Lint(&db);
  // 在图融合后，可能会重新出现一些公共子表达式，进行消除
  EliminateCommonSubexpression(graph);
  // 可能生成了大量无用的形状传播代码，进行移除
  EliminateDeadCode(graph);
  // 改进剩余形状传播代码的质量
  PeepholeOptimizeShapeExpressions(graph->block(), &db);
}

// 自定义图融合操作
void CustomFuseGraph(
    std::shared_ptr<Graph>& graph,
    const std::function<bool(Node*)>& fn,
    Symbol kind,
    size_t arg_limit) {
  // 创建图的别名数据库
  AliasDb db(graph);
  // 创建图融合器对象，传入别名数据库、图的块，以及自定义的条件函数和符号种类
  auto g = GraphFuser(
      &db,
      graph->block(),
      [=](GraphFuser* gf, Node* n) { return fn(n) || n->kind() == kind; },
      kind);
  // 设置输入参数的限制
  g.setInputArgLimit(arg_limit);
  // 运行图融合操作
  g.run();
  // 对别名数据库进行静态检查
  Lint(&db);
}

// 命名空间 jit 的结束
} // namespace jit
// 命名空间 torch 的结束
} // namespace torch
```