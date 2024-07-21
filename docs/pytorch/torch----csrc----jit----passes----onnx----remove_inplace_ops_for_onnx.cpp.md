# `.\pytorch\torch\csrc\jit\passes\onnx\remove_inplace_ops_for_onnx.cpp`

```py
// 包含 Torch 的 JIT 模块中的头文件
#include <torch/csrc/jit/passes/onnx/helper.h>
#include <torch/csrc/jit/passes/onnx/remove_inplace_ops_for_onnx.h>
#include <torch/csrc/jit/passes/remove_inplace_ops.h>
#include <torch/csrc/jit/passes/remove_mutation.h>

// 包含 Torch 的 JIT 前端错误报告和日志记录的头文件
#include <torch/csrc/jit/frontend/error_report.h>
#include <torch/csrc/jit/jit_log.h>

// 包含 Torch 的 JIT 死代码消除和 ONNX 相关的帮助函数的头文件
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/onnx/helper.h>
#include <torch/csrc/jit/passes/onnx/pattern_conversion/pattern_encapsulation.h>

// 包含 C10 的实用工具中的头文件
#include <c10/util/irange.h>

// 包含标准库中的头文件
#include <limits>

// Torch 的命名空间
namespace torch {
namespace jit {

// 实现一个匿名命名空间，用于定义内部静态变量和函数
namespace {

// 定义了一组原地操作符号集合，用于识别和替换相应的操作
const std::set<c10::Symbol> inplace_ops =
    {aten::append, aten::index_put_, aten::pop, aten::insert, aten::Delete};

// InplaceConverter 类定义了一组函数，用于实现从 prim::GetAttr、prim::SetAttr 和 ATen 原地操作符号
// 到 ONNX 非原地操作符号的转换
struct InplaceConverter {
  // 构造函数初始化 InplaceConverter 对象
  InplaceConverter(
      std::shared_ptr<Graph> graph,
      MutationRemover* mr,
      Module* model = nullptr)
      : graph_(std::move(graph)), mr_(mr), module_(model) {}

  // 执行在 ONNX 中转换原地操作符号的操作
  void convertMutationForONNX();

 private:
  // 在一个块中收集属性名和初始值的映射关系
  void gatherAttrNameInitialValueMap(
      Block* block,
      std::unordered_map<std::string, Value*>& attr_name_value_map,
      std::unordered_map<Node*, std::string>& attr_node_fullname_map);
  
  // 替换块中的属性为原地操作符号
  void replaceAttrWithInplaceOps(
      Block* block,
      const std::unordered_map<std::string, Value*>& attr_name_value_map,
      const std::unordered_map<Node*, std::string>& attr_node_fullname_map);

  // 转换原地操作符号并跟踪别名
  void convertInplaceOpsAndTrackAlias();
  void convertInplaceOpsAndTrackAlias(Block* block);

  // 修正别名引用
  void correctAliasReferences();
  void correctAliasReferences(Block* block);
  void correctAliasReferences(Node* n);

  // 将 prim::GetAttr 和 prim::SetAttr 转换为原地操作符号
  void convertGetSetAttrToInplaceOps(Block* block);

  // ValueTracker 类提供了记录单个值别名和根据其在图中使用位置找到正确别名的接口
  struct ValueTracker {
    ValueTracker() : graph_(nullptr) {}

    // 初始化 ValueTracker
    void init(const std::shared_ptr<Graph>& graph);

    // 记录设置值的别名
    void recordSetValue(Value* old_v, Value* new_v);

    // 根据节点找到值的别名
    Value* findAliasForValueAtNode(Value* v, const Node* n) const;

    // 返回值追踪器的字符串表示
    std::string toString() const;

   private:
    std::shared_ptr<Graph> graph_;

    // 别名到根值的映射
    std::unordered_map<Value*, Value*> alias_to_value_;

    // 基于图中顺序对别名进行排序
    // 当两个不同块中的别名具有相同的祖先节点时，可能会发生并列现象
    // 用于比较别名的严格弱序关系
    // aliasComp 必须满足严格弱序关系
    // 定义一个结构体 aliasComp，用于比较 Value 对象的别名顺序
    struct aliasComp {
      // 重载 () 运算符，比较两个 Value 指针的节点顺序
      bool operator()(const Value* a, const Value* b) const {
        // 获取 a 和 b 对应的节点指针
        auto* n_a = a->node();
        auto* n_b = b->node();
        // 如果节点相同，则返回 false
        if (n_a == n_b) {
          return false;
        }
        // 比较节点间的顺序关系
        auto a_b = n_a->isBefore(n_b);
        auto b_a = n_b->isBefore(n_a);
        // 如果节点间顺序相同，则比较它们的唯一标识
        if (a_b == b_a) {
          return a->unique() < b->unique();
        }
        // 返回节点间的顺序比较结果
        return a_b;
      }
    };

    // 使用 Value 指针到有序的 Value 指针集合的无序映射
    std::unordered_map<Value*, std::set<Value*, aliasComp>>
        value_to_sorted_aliases_;
  };

  // 图的共享指针
  std::shared_ptr<Graph> graph_;

  // 变异移除器指针
  MutationRemover* mr_;

  // 模块指针
  Module* module_;

  // 值跟踪器对象
  ValueTracker vt_;
};

bool isAncestor(const Block* a, const Block* b) {
  // 检查从节点 b 到其所属节点的链路，判断节点 a 是否在该链路上
  while (b && b->owningNode()) {
    if (a == b) {
      return true;
    }
    b = b->owningNode()->owningBlock();
  }
  // 若遍历结束仍未找到 a，则返回 false
  return a == b;
}

Node* addDummyClone(
    Graph* graph,
    Value* orig_data,
    bool insertBefore,
    Node* referenceNode) {
  Node* newNode = nullptr;
  if (orig_data->type()->kind() == TypeKind::ListType) {
    // 如果原始数据是列表类型，则创建一个列表节点
    newNode = graph->create(aten::list, /*num_outputs=*/1);
    newNode->addInput(orig_data);
    newNode->output()->setType(orig_data->type());
    // 根据 insertBefore 参数决定节点插入方式
    if (insertBefore)
      newNode->insertBefore(referenceNode);
    else
      referenceNode->owningBlock()->prependNode(newNode);
  } else if (
      orig_data->type()->kind() == TypeKind::TensorType ||
      orig_data->type()->kind() == TypeKind::IntType ||
      orig_data->type()->kind() == TypeKind::FloatType ||
      orig_data->type()->kind() == TypeKind::BoolType) {
    // 如果原始数据是张量、整数、浮点数或布尔类型之一，则需进行克隆处理
    auto* noneNode = graph->create(prim::Constant);
    noneNode->output()->setType(NoneType::get());
    // 在脚本模式下，aten::clone 要求输入必须是张量类型
    // 因此，对于整数、浮点数或布尔数值，需要将其转换为相应的张量类型
    if (orig_data->type()->kind() == TypeKind::IntType &&
        insertBefore == false) {
      orig_data->setType(TensorType::fromNumberType(*IntType::get()));
    } else if (
        orig_data->type()->kind() == TypeKind::FloatType &&
        insertBefore == false) {
      orig_data->setType(TensorType::fromNumberType(*FloatType::get()));
    } else if (
        orig_data->type()->kind() == TypeKind::BoolType &&
        insertBefore == false) {
      orig_data->setType(TensorType::fromBoolType());
    }
    // 创建克隆节点，并设置其类型
    newNode = graph->create(aten::clone, /*num_outputs=*/1);
    newNode->addInput(orig_data);
    newNode->addInput(noneNode->output());
    newNode->output()->setType(orig_data->type());
    // 根据 insertBefore 参数决定节点插入方式
    if (insertBefore)
      newNode->insertBefore(referenceNode);
    else
      referenceNode->owningBlock()->prependNode(newNode);
    // 在克隆节点前插入 None 节点
    noneNode->insertBefore(newNode);
  }
  // 返回新创建的节点
  return newNode;
}

std::pair<Value*, Value*> PrepareIndexPutForONNX(Node* node) {
  // 断言节点类型为 aten::index_put 或 aten::index_put_
  TORCH_INTERNAL_ASSERT(
      node->kind() == aten::index_put || node->kind() == aten::index_put_);
  // 将节点封装为子块，返回输入和输出节点的对
  auto placeholder_node = EncapsulatePatternIntoSubblock(node).value();
  // 销毁原始节点
  node->destroy();
  // 返回输入和输出节点对
  return std::make_pair(placeholder_node->input(0), placeholder_node->output());
}
// 准备用于 ONNX 的复制操作。此函数处理 aten::copy_ 操作。
std::pair<Value*, Value*> PrepareCopyForONNX(Node* node) {
  // 断言节点类型必须是 aten::copy_
  TORCH_INTERNAL_ASSERT(node->kind() == aten::copy_);
  
  // aten::copy_ 可以视作 index_put 的特例，其中张量索引为空。
  // 移除 aten::copy_，用 index_put 替换。
  // 1. 创建一个空的 ListConstruct 节点作为 index_put 的索引输入。
  // 2. 创建 index_put 节点。

  // 在脚本化过程中，跟踪 aten::copy_ 并进行广播操作。
  // 3. 为脚本化应用广播操作。
  WithInsertPoint guard(node); // 设置插入点为当前节点
  auto graph = node->owningGraph(); // 获取节点所属的图

  // 创建一个空的 ListConstruct 节点作为 index_put 的索引输入
  auto dummy_list =
      graph->insertNode(graph->createList(OptionalType::ofTensor(), {}))
          ->output();

  // 插入 expand_as 节点，并复制元数据和源代码范围
  auto expanded_value =
      graph->insert(aten::expand_as, {node->input(1), node->input(0)});
  expanded_value->node()->setSourceRange(node->sourceRange());
  expanded_value->copyMetadata(node->input(1));
  expanded_value->node()->copyMetadata(node);

  // 插入 index_put_ 节点，并复制元数据
  auto index_put = graph->insert(
      aten::index_put_,
      {node->input(0), dummy_list, expanded_value, node->input(2)});
  index_put->node()->copyMetadata(node);
  index_put->copyMetadata(node->output());

  // 替换当前节点的所有使用为 index_put，并销毁当前节点
  node->output()->replaceAllUsesWith(index_put);
  node->destroy();

  // 返回准备好的 index_put 操作的结果
  return PrepareIndexPutForONNX(index_put->node());
}

// 准备用于 ONNX 的 set 操作。此函数处理 aten::set_ 操作。
auto PrepareSetForONNX(Node* n) {
  // 断言节点类型必须是 aten::set_
  TORCH_INTERNAL_ASSERT(n->kind() == aten::set_);

  // 在图中添加一个带有虚拟克隆的节点，并复制元数据
  auto clone_n = addDummyClone(n->owningGraph(), n->input(1), true, n);
  TORCH_INTERNAL_ASSERT(nullptr != clone_n);
  clone_n->copyMetadata(n);

  // 替换原始输入为克隆节点的输出，并销毁当前节点
  auto orig_input = n->input(0);
  n->output()->replaceAllUsesWith(clone_n->output());
  n->destroy();

  // 返回原始输入和克隆节点输出的 pair
  return std::make_pair(orig_input, clone_n->output());
}

// 准备用于 ONNX 中块内的原地操作。此函数处理原地操作节点。
std::pair<Value*, Value*> PrepareInplaceOpsInBlocksForONNX(Node* node) {
  // 如果节点不是 aten 操作，则返回空
  if (!node->kind().is_aten())
    return {};

  // 获取节点的操作名称和是否是原地操作
  auto name = node->schema().name();
  bool inplace_op = name.at(name.size() - 1) == '_';
  if (!inplace_op)
    return {};

  // 获取新的操作名称
  auto new_schema = name.substr(0, name.size() - 1);

  // 获取节点的输入节点
  Node* input_node = node->inputs().at(0)->node();

  // 获取当前节点所属的图
  auto graph = node->owningGraph();

  // 创建新的节点，并设置输入、输出类型，插入到当前节点之前
  auto new_node = graph->create(Symbol::fromQualString(new_schema), 1);
  for (Value* input : node->inputs()) {
    new_node->addInput(input);
  }
  new_node->output()->setType(node->output()->type());
  new_node->insertBefore(node);
  new_node->copyMetadata(node);

  // 替换当前节点的所有使用为新节点，并销毁当前节点
  node->replaceAllUsesWith(new_node);
  node->destroy();

  // 如果输入节点是 select 或者 slice，则进行特殊处理
  if (input_node->kind() == aten::select || input_node->kind() == aten::slice) {
    // 对于 a[i] = x 的情况，转换为 copy_，最终转换为 index_put_
    WithInsertPoint guard(new_node);
    auto false_val_ = graph->insertConstant(false);

    auto new_copy = graph->create(aten::copy_, 1);
    new_copy->addInput(new_node->inputs().at(0));
    new_copy->addInput(new_node->output());
    new_copy->addInput(false_val_);
    new_copy->insertAfter(new_node);
    new_copy->copyMetadata(new_node);

    return PrepareCopyForONNX(new_copy);
  } else {
    // 直接别名操作，节点是独立的原地操作
    return std::make_pair(new_node->input(0), new_node->output());


// 返回一个包含新节点输入和输出的 pair 对象
    return std::make_pair(new_node->input(0), new_node->output());
// Remove Mutation pass does not handle mutation on block inputs.
// To fix this, insert a clone node following the graph input:
// Example for graph input node %0:
// Before:
// graph(%0 : Tensor):
//   %5 : Tensor = aten::zero_(%0)
//   ...
// After:
// graph(%0 : Tensor):
//   %2 : None = prim::Constant()
//   %3 : Tensor = aten::clone(%0, %2)
//   %5 : Tensor = aten::zero_(%3)
//   ...

// 为了解决移除变异的问题，此函数在块输入后插入一个克隆节点：
// 例如，对于图输入节点 %0：
// 在之前：
// graph(%0 : Tensor):
//   %5 : Tensor = aten::zero_(%0)
//   ...
// 在之后：
// graph(%0 : Tensor):
//   %2 : None = prim::Constant()
//   %3 : Tensor = aten::clone(%0, %2)
//   %5 : Tensor = aten::zero_(%3)
//   ...

static void PrepareForRemoveMutations(MutationRemover& mr, Block* b) {
  // 递归遍历块中的每一个节点及其子块，以便处理移除变异
  for (auto it = b->nodes().begin(), end = b->nodes().end(); it != end; ++it) {
    for (auto* child_block : it->blocks()) {
      PrepareForRemoveMutations(mr, child_block);
    }
  }

  // 处理块的输入节点，需要在需要时重新开始处理
  for (auto input : b->inputs()) {
    bool needsRestart = false;
    // 使用 do-while 循环来处理可能需要多次重启的情况
    do {
      // 初始化标志，表示当前循环不需要重启
      needsRestart = false;
      // 遍历输入节点的使用情况
      for (auto use : input->uses()) {
        // 获取使用节点的指针
        Node* node = use.user;
        // 如果节点不支持原地操作的变体，则跳过处理
        if (!mr.inplaceOpVariant(node)) {
          continue;
        }
        // 在节点的输入中查找当前输入的位置
        auto it =
            std::find(node->inputs().begin(), node->inputs().end(), input);
        // 如果找到了输入的位置
        if (it != node->inputs().end()) {
          // 计算输入在节点输入列表中的索引
          int index = std::distance(node->inputs().begin(), it);
          // 发出警告，指出正在处理 ONNX 预处理，移除节点的变异
          TORCH_WARN(
              "ONNX Preprocess - Removing mutation from node ",
              node->kind().toQualString(),
              " on block input: '",
              (*it)->debugName(),
              "'. This changes graph semantics.");

          // 添加一个虚拟克隆节点到图中，该节点不进行原地操作，并确保返回节点有效
          Node* newNode =
              addDummyClone(b->owningGraph(), input, false, b->return_node());
          // 断言确保新节点非空
          TORCH_INTERNAL_ASSERT(nullptr != newNode);
          // 将新节点的元数据复制给原始节点
          newNode->copyMetadata(node);
          // 替换原始节点的输入索引处的输入为新节点的输出
          node->replaceInput(index, newNode->output());
          // 在节点之后使用新节点替换输入的所有用途
          input->replaceAllUsesAfterNodeWith(node, newNode->output());
          // 标记需要重新启动循环
          needsRestart = true;
          // 中断当前循环，重新开始执行 do-while 循环
          break;
        }
      }
    // 如果标志为 true，表示需要重新启动循环以处理更改
    } while (needsRestart);
}

// 准备移除变异操作，接收一个图形对象指针作为参数
static void PrepareForRemoveMutations(std::shared_ptr<Graph> graph) {
  // 使用MutationRemover类处理图形对象
  MutationRemover mr(graph);
  // 调用PrepareForRemoveMutations函数处理变异操作，并传入图形对象的根块
  PrepareForRemoveMutations(mr, graph->block());
  // 输出图形对象的状态，用于调试和分析
  GRAPH_DUMP("After PrepareForRemoveMutations: ", graph);
}

// findSubModuleAttr函数追踪getAttr链以定位子模块属性
// 例如：module M { attributes { A = <SubModule at ...> } %A = prim::GetAttr[name="A"](%self) %B = prim::GetAttr[name="B"](%A) %weight = prim::GetAttr[name="scale"](%B)
std::deque<std::string> findSubModuleAttr(
    Value* input,
    std::string& name,
    Module& attrModule,
    const std::shared_ptr<Graph>& graph) {
  // 获取输入值所属的节点
  Node* node = input->node();
  // 存储模块名的双向队列
  std::deque<std::string> moduleNames;

  // 从内部子模块开始追踪链条，直至达到顶层模块
  auto selfNode = graph->nodes().begin();
  auto n = *selfNode;
  while (node->outputs().at(0)->type() != n->output()->type()) {
    // 如果节点类型为prim::GetAttr，则将属性名称添加到moduleNames中
    if (node->kind() == prim::GetAttr) {
      moduleNames.push_front(node->s(attr::name));
      node = node->inputs()[0]->node();
    } else {
      break;
    }
  }
  // 将内部模块赋值给attrModule
  for (auto& moduleName : moduleNames) {
    attrModule = attrModule.attr(moduleName).toModule();
  }
  return moduleNames;
}

// findArgumentAsInputParam函数在图形对象的输入中查找参数名对应的输入值
Value* findArgumentAsInputParam(
    const std::shared_ptr<Graph>& graph,
    std::string& name,
    IValue& attr) {
  // 遍历图形对象的所有输入
  for (auto input : graph->inputs()) {
    // 如果输入值的调试名称与指定的名称相匹配，则返回该输入值
    if (input->debugName() == name)
      return input;
  }
  // 如果未找到匹配的输入值，则抛出运行时错误
  throw std::runtime_error(
      "Attribute is not part of model parameters. Cannot handle SetAttr and GetAttr nodes for : " +
      name);
}

// InplaceConverter::ValueTracker::init函数初始化ValueTracker对象
void InplaceConverter::ValueTracker::init(const std::shared_ptr<Graph>& graph) {
  // 初始化别名到值的映射为空
  alias_to_value_ = {};
  // 初始化值到已排序别名的映射为空
  value_to_sorted_aliases_ = {};
  // 将输入的图形对象指针保存到成员变量中
  graph_ = graph;
}

// InplaceConverter::ValueTracker::toString函数将ValueTracker对象转换为字符串表示
std::string InplaceConverter::ValueTracker::toString() const {
  std::stringstream ss;

  // 打印跟踪的图形对象的信息
  ss << "Tracking " << value_to_sorted_aliases_.size() << " individual values."
     << std::endl;
  // 打印值到已排序别名的映射信息
  ss << "value_to_sorted_aliases_: " << std::endl;
  size_t idx = 0;
  for (const auto& it : value_to_sorted_aliases_) {
    ss << "Value[" << idx << "]: " << it.first->debugName() << std::endl;
    ss << "  Mapping to ";
    for (auto v : it.second) {
      ss << v->debugName() << " ";
    }
    ss << std::endl;
    idx++;
  }

  // 打印别名到值的映射信息
  ss << "alias_to_value_: " << std::endl;
  for (auto it : alias_to_value_) {
    ss << "  Alias " << it.first->debugName();
    ss << " map to " << it.second->debugName() << std::endl;
  }

  return ss.str();
}

// InplaceConverter::ValueTracker::recordSetValue函数记录设置值操作
void InplaceConverter::ValueTracker::recordSetValue(
    Value* old_v,
    Value* new_v) {
  // 记录调用信息，包括旧值和新值的调试名称
  GRAPH_UPDATE(
      "Calling recordSetValue with old_v: ",
      old_v->debugName(),
      " new_v: ",
      new_v->debugName());
  // 更新图形状态
  GRAPH_UPDATE(this->toString());
  // 获取新值所属的节点
  auto* n = new_v->node();
  // 获取该节点所在的块
  auto* owning_block = n->owningBlock();

  // 如果旧值不在别名映射中，则将其添加
  if (alias_to_value_.find(old_v) == alias_to_value_.end()) {
    alias_to_value_[old_v] = old_v;
    value_to_sorted_aliases_[old_v] = {old_v};
  }

  // 获取旧值的根值
  auto root_v = alias_to_value_[old_v];
  // 将新值映射到旧值的根值
  alias_to_value_[new_v] = root_v;
  // 获取根值对应的排序别名列表的引用
  auto& sorted_alias = value_to_sorted_aliases_[root_v];
  // 将新值插入排序别名列表
  sorted_alias.insert(new_v);

  // 检查新值是否在 if 或循环子块内创建
  auto* owning_blocknode = owning_block->owningNode();
  if (nullptr == owning_blocknode) {
    return;
  }
  auto owning_block_nkind = owning_blocknode->kind();
  // 如果不是在 if 或循环块内创建，则返回
  if (owning_block_nkind != prim::Loop && owning_block_nkind != prim::If) {
    return;
  }

  // 检查是否有输出值是子块的输出
  bool registered = std::any_of(
      owning_block->outputs().begin(),
      owning_block->outputs().end(),
      [&sorted_alias](Value* out) {
        return std::any_of(
            sorted_alias.begin(), sorted_alias.end(), [&out](Value* alias) {
              return alias == out;
            });
      });

  // 检查该值是否来自外部块的别名
  bool from_outer_alias = std::any_of(
      sorted_alias.begin(),
      sorted_alias.end(),
      [&owning_blocknode](Value* alias) {
        return isAncestor(
            alias->node()->owningBlock(), owning_blocknode->owningBlock());
      });

  // 如果该值已经修改，并且来自外部块的别名，则需要注册为子块的输出
  // 如果该值的其他别名已经注册为子块输出，则此步骤可以跳过
  if (!registered && from_outer_alias) {
    if (owning_block_nkind == prim::Loop) {
      // 注册循环块的输出
      owning_block->registerOutput(new_v);
      // 添加新的块输入，并设置类型
      auto new_block_in = owning_block->addInput();
      new_block_in->setType(new_v->type());
      // 将新块输入插入排序别名列表
      sorted_alias.insert(new_block_in);
      // 将新块输入映射到根值
      alias_to_value_[new_block_in] = root_v;
      // 向块节点添加输入
      owning_blocknode->addInput(root_v);
    } else if (owning_block_nkind == prim::If) {
      // 遍历 if 块的子块
      for (auto* if_sub_block : owning_blocknode->blocks()) {
        if (owning_block == if_sub_block) {
          // 注册 if 块的输出
          if_sub_block->registerOutput(new_v);
        } else {
          // 注册根值作为 if 块的输出
          if_sub_block->registerOutput(root_v);
        }
      }
    }
    // 添加新块节点输出，并设置类型
    auto* new_blocknode_out = owning_blocknode->addOutput();
    new_blocknode_out->setType(new_v->type());
    // 记录设置
}

// 结束 InplaceConverter 类的成员函数定义

// 根据当前值别名记录，遍历图并纠正所有节点的别名引用
void InplaceConverter::correctAliasReferences() {
  correctAliasReferences(graph_->block());
}

// 根据给定的块遍历并纠正别名引用
void InplaceConverter::correctAliasReferences(Block* block) {
  for (auto it = block->nodes().begin(); it != block->nodes().end();) {
    Node* n = *it;
    it++; // 可能会销毁节点 n，因此提前递增迭代器

    correctAliasReferences(n); // 对节点 n 进行别名引用的纠正

    auto nkind = n->kind();
    // 如果节点类型为 prim::If 或 prim::Loop，则进一步遍历其子块
    if (nkind == prim::If || nkind == prim::Loop) {
      for (auto* sub_block : n->blocks()) {
        correctAliasReferences(sub_block);
      }
    }
  }
  correctAliasReferences(block->return_node()); // 纠正块的返回节点的别名引用
}

// 对节点 n 的每个输入找到正确的别名
void InplaceConverter::correctAliasReferences(Node* n) {
  for (size_t i = 0; i < n->inputs().size(); ++i) {
    auto* in = n->input(i);
    auto* alias = vt_.findAliasForValueAtNode(in, n);

    if (alias != in) {
      n->replaceInput(i, alias);
      GRAPH_UPDATE(
          "Replacing ",
          in->debugName(),
          " with ",
          alias->debugName(),
          " for ",
          *n); // 更新日志：用 alias 替换 in 作为节点 n 的输入
    }
  }
}

// 在节点 n 处找到表示值 v 的正确别名
Value* InplaceConverter::ValueTracker::findAliasForValueAtNode(
    Value* v,
    const Node* n) const {
  GRAPH_UPDATE("Finding alias for value:", v->debugName(), " at node ", *n);
  // 如果 v 没有被任何 inplace 操作影响，则直接返回 v
  if (alias_to_value_.find(v) == alias_to_value_.end()) {
    return v;
  }

  auto* root_v = alias_to_value_.find(v)->second;
  // 确保 root_v 的别名列表在 value_to_sorted_aliases_ 中存在
  TORCH_INTERNAL_ASSERT(
      value_to_sorted_aliases_.find(root_v) != value_to_sorted_aliases_.end());
  const auto& aliases = value_to_sorted_aliases_.find(root_v)->second;

  // 查找满足条件的最后一个别名：别名所在块是 n 的祖先且别名所在节点在 n 之前
  Value* found_alias = nullptr;
  for (auto* alias : aliases) {
    auto* alias_n = alias->node();
    if (alias_n->isBefore(n) &&
        isAncestor(alias_n->owningBlock(), n->owningBlock())) {
      found_alias = alias;
    }
  }

  // 确保找到了有效的别名
  TORCH_INTERNAL_ASSERT(
      nullptr != found_alias,
      "More details: \n",
      n->sourceRange().str(),
      "Input ",
      v->debugName(),
      " of node ",
      *n,
      " was modified by in-place operation, but we cannot find its updated value. ",
      "Please report a bug to PyTorch, and/or try to avoid using in-place operators on this value.");

  return found_alias;
}

// 遍历块，收集任何属性的初始值，并缓存每个 GetAttr/SetAttr 节点的完整属性名
void InplaceConverter::gatherAttrNameInitialValueMap(
    Block* block,
    std::unordered_map<std::string, Value*>& attr_name_value_map,
    std::unordered_map<Node*, std::string>& attr_node_fullname_map) {
  for (auto it = block->nodes().begin(); it != block->nodes().end();) {
    // 从迭代器中获取指向节点指针的引用，并将迭代器向前移动一步
    Node* n = *it;
    it++; // 节点 n 可能会被销毁

    // 遍历节点 n 的所有子块，并递归地收集属性名和初始值的映射关系
    for (auto* sub_block : n->blocks()) {
      gatherAttrNameInitialValueMap(
          sub_block, attr_name_value_map, attr_node_fullname_map);
    }

    // 如果节点 n 的类型不是 prim::GetAttr 或 prim::SetAttr，则继续下一个迭代
    if (n->kind() != prim::GetAttr && n->kind() != prim::SetAttr)
      continue;

    // 获取节点 n 的名称
    auto name = n->s(attr::name);
    // 复制模块指针
    auto attrModule = *module_;
    // 初始化参数常量指针为 nullptr
    Value* paramConst = nullptr;

    // 查找节点 n 输入的第一个子模块的属性名列表
    auto moduleNames =
        findSubModuleAttr(n->inputs().at(0), name, attrModule, graph_);

    // 构建完整的属性节点名称，包括所有父模块的名称
    std::string fullName("");
    for (auto& name : moduleNames) {
      fullName += name + '.';
    }
    fullName += name;

    // 将节点 n 与完整属性节点名称的映射关系插入到映射表中
    attr_node_fullname_map.insert({n, fullName});

    // 如果映射表中不包含完整属性节点名称，并且模块 attrModule 包含该属性名
    if (attr_name_value_map.find(fullName) == attr_name_value_map.end() &&
        attrModule.hasattr(name)) {
      // 获取属性的值
      auto attr = attrModule.attr(name);
      // 获取模块类型
      auto type = attrModule.type();
      // 查找属性在类型中的索引槽位
      auto slot = *type->findAttributeSlot(name);

      // 在图的首节点前插入当前插入点，并添加模型参数和模型缓冲区作为模型输入，保持顺序与图中出现的顺序一致
      WithInsertPoint guard(graph_->nodes().front());
      if (type->is_parameter(slot) || type->is_buffer(slot) ||
          (attr.isObject() && !attr.toObjectRef().type()->is_module())) {
        // 将找到的参数作为输入参数插入到图中，并将其与完整属性节点名称映射
        paramConst = findArgumentAsInputParam(graph_, fullName, attr);
        attr_name_value_map.insert({fullName, paramConst});
      } else if (auto attrVal = tryInsertConstant(*graph_, attr)) {
        // 尝试插入常量作为属性的值
        // TODO: 扩展对类型为 List[Tensor] 等的属性支持
        for (size_t i = 0; i < type->getAttributes().size(); i++) {
          if (type->getAttributeName(i) == name) {
            paramConst = *attrVal;
            attr_name_value_map.insert({fullName, paramConst});
          }
        }
      } else {
        // 如果属性是自定义类对象，而不是原始类型、Tensor 或 List/Tuple/Dict of Tensors，则记录调试信息
        GRAPH_DEBUG(
            attr.type()->cast<ClassType>() ? "" : "attribute: ",
            name,
            " is not materializable.");
      }
    }

    // 如果映射表中不包含完整属性节点名称，则创建一个虚拟的初始值为 None 的节点，并插入到图中
    if (attr_name_value_map.find(fullName) == attr_name_value_map.end()) {
      auto* noneNode = graph_->create(prim::Constant);
      noneNode->output()->setType(NoneType::get());
      noneNode->insertBefore(graph_->nodes().front());
      attr_name_value_map.insert({fullName, noneNode->output()});
    }
  }
void InplaceConverter::replaceAttrWithInplaceOps(
    Block* block,
    const std::unordered_map<std::string, Value*>& attr_name_value_map,
    const std::unordered_map<Node*, std::string>& attr_node_fullname_map) {
  // 遍历所有的节点和它们的完整名称映射
  for (const auto& pair : attr_node_fullname_map) {
    auto* n = pair.first;  // 获取节点指针
    auto fullName = pair.second;  // 获取节点的完整名称
    auto find_init_val = attr_name_value_map.find(fullName);  // 在映射中查找节点初始值
    TORCH_INTERNAL_ASSERT(find_init_val != attr_name_value_map.end());  // 断言确保找到初始值

    // 断言节点的类型为 prim::GetAttr 或 prim::SetAttr
    TORCH_INTERNAL_ASSERT(
        n->kind() == prim::GetAttr || n->kind() == prim::SetAttr);
    if (n->kind() == prim::SetAttr) {
      // 将 SetAttr 转换为 inplace 操作 aten::set_
      WithInsertPoint guard(n);  // 设置插入点
      auto* set_node = graph_->create(aten::set_, 1);  // 创建新的 set_ 节点
      set_node->addInput(find_init_val->second);  // 添加初始值作为输入
      set_node->addInput(n->input(1));  // 添加 SetAttr 的第二个输入作为输入
      set_node->copyMetadata(n);  // 复制元数据
      set_node->insertBefore(n);  // 在当前节点之前插入新节点
    } else if (n->kind() == prim::GetAttr) {
      // 如果节点的类型是 GetAttr
      // 将 GetAttr 的使用替换为第一次出现的别名（通常是初始值），
      // 在后续的处理中会发现并分配该节点位置正确的别名。
      n->output()->replaceAllUsesWith(find_init_val->second);
    }

    // 销毁当前节点
    n->destroy();
// 将 GetAttr 和 SetAttr 转换为原地操作的版本，并在 ValueTracker 中记录相关的新别名
void InplaceConverter::convertGetSetAttrToInplaceOps(Block* block) {
  // 初始化空的属性名到值的映射
  std::unordered_map<std::string, Value*> attr_name_value_map = {};
  // 初始化空的节点到完整属性名的映射
  std::unordered_map<Node*, std::string> attr_node_fullname_map = {};

  // 第一遍遍历图，收集所有属性名及其初始值。如果需要，为属性创建虚拟初始值。
  // 在此遍历结束时，这些虚拟初始值应该没有被使用，可以安全地移除。否则会意味着模型使用了未初始化的值。
  gatherAttrNameInitialValueMap(block, attr_name_value_map, attr_node_fullname_map);

  // 更新图后的日志输出，显示 gatherAttrNameInitialValueMap 后的图状态
  GRAPH_UPDATE("Graph after gatherAttrNameInitialValueMap", graph_->toString());

  // 第二遍遍历图，
  // 替换 GetAttr 为第一次见到的别名（通常是初始值），
  // 替换 SetAttr 为原地操作，更新新值到第一次见到的别名上。
  replaceAttrWithInplaceOps(block, attr_name_value_map, attr_node_fullname_map);
}

// 将原地操作转换为非原地版本，并在 ValueTracker 中记录相关的新别名
void InplaceConverter::convertInplaceOpsAndTrackAlias(Block* block) {
  // 遍历当前块中的所有节点
  for (auto it = block->nodes().begin(); it != block->nodes().end();) {
    Node* n = *it;
    it++; // 节点 n 可能被销毁

    auto nkind = n->kind();
    // 如果是条件或循环节点，递归处理其子块
    if (nkind == prim::If || nkind == prim::Loop) {
      for (Block* sub_block : n->blocks()) {
        convertInplaceOpsAndTrackAlias(sub_block);
      }
    } else {
      Value *orig_data = nullptr, *new_out = nullptr;
      // 根据节点类型选择合适的 ONNX 转换函数
      if (nkind == aten::copy_) {
        std::tie(orig_data, new_out) = PrepareCopyForONNX(n);
      } else if (nkind == aten::index_put || nkind == aten::index_put_) {
        std::tie(orig_data, new_out) = PrepareIndexPutForONNX(n);
        // 特殊情况，index_put 不是原地操作
        if (nkind == aten::index_put) {
          continue;
        }
      } else if (nkind == aten::insert || nkind == aten::append) {
        std::tie(orig_data, new_out) = PrepareListAppendAndInsertForONNX(n);
      } else if (nkind == aten::set_) {
        std::tie(orig_data, new_out) = PrepareSetForONNX(n);
      } else if (mr_->inplaceOpVariant(n)) {
        std::tie(orig_data, new_out) = PrepareInplaceOpsInBlocksForONNX(n);
      } else if (nkind == aten::pop) {
        std::tie(orig_data, new_out) = PrepareListPopForONNX(n);
      } else if (nkind == aten::Delete) {
        std::tie(orig_data, new_out) = PrepareListDeleteForONNX(n);
      } else if (nkind == aten::_set_item) {
        std::tie(orig_data, new_out) = PrepareSetItemForONNX(n);
      } else {
        // 不是原地操作的情况
        continue;
      }

      // 如果找到了有效的原始数据和新输出，则记录它们的关系
      if (nullptr != orig_data && nullptr != new_out) {
        vt_.recordSetValue(orig_data, new_out);
      }
    }
  }
}
void InplaceConverter::convertInplaceOpsAndTrackAlias() {
  // 调用 convertInplaceOpsAndTrackAlias 方法处理当前图块中的原地操作
  convertInplaceOpsAndTrackAlias(graph_->block());
  // 更新图的字符串表示，输出日志信息
  GRAPH_UPDATE(
      "Graph after convertInplaceOpsAndTrackAlias: ", graph_->toString());
  // 输出值追踪器的字符串表示
  GRAPH_UPDATE(vt_.toString());
}

void InplaceConverter::convertMutationForONNX() {
  // 第一遍扫描，将所有 prim::GetAttr 和 prim::SetAttr 转换为 ATen 的原地操作
  convertGetSetAttrToInplaceOps(graph_->block());
  // 更新图的字符串表示，输出日志信息
  GRAPH_UPDATE("Graph after convertGetSetAttrToInplaceOps", graph_->toString());
  // 初始化值追踪器
  vt_.init(graph_);
  // 第二遍扫描，将所有原地操作转换为非原地版本，并记录新的别名到值追踪器中
  convertInplaceOpsAndTrackAlias();
  // 第三遍扫描，检查并修正所有节点的别名引用
  correctAliasReferences();
}

} // namespace

void RemoveInplaceOpsForONNX(
    const std::shared_ptr<Graph>& graph,
    Module* model = nullptr) {
  // 隐式转换二元原地操作为非原地版本
  ImplicitCastForBinaryInplaceOps(graph->block());
  // 为移除突变操作做准备
  PrepareForRemoveMutations(graph);
  // 创建 MutationRemover 对象，并从图中移除张量突变
  MutationRemover mr(graph);
  mr.removeTensorMutation();
  // 从图中移除列表突变
  mr.removeListMutation();
  // 创建 InplaceConverter 对象，并执行 ONNX 转换
  InplaceConverter ic(graph, &mr, model);
  ic.convertMutationForONNX();
}

} // namespace jit
} // namespace torch


这段代码主要涉及图的操作和突变操作的移除，通过一系列的扫描和转换操作来确保图在进行转换后仍然保持正确的状态，并记录相关的别名信息。
```