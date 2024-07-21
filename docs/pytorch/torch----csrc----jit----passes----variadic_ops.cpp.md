# `.\pytorch\torch\csrc\jit\passes\variadic_ops.cpp`

```py
// 引入 Torch 库中的头文件，用于 JIT 编译的变参操作
#include <torch/csrc/jit/passes/variadic_ops.h>

// 引入 Torch 库中的别名分析相关头文件
#include <torch/csrc/jit/ir/alias_analysis.h>
// 引入 Torch 库中的 JIT 日志相关头文件
#include <torch/csrc/jit/jit_log.h>
// 引入 Torch 库中的常量池化相关头文件
#include <torch/csrc/jit/passes/constant_pooling.h>
// 引入 Torch 库中的移除变异相关头文件
#include <torch/csrc/jit/passes/remove_mutation.h>

// Torch 的命名空间
namespace torch {
namespace jit {

// 匿名命名空间，用于定义局部函数或变量
namespace {

// 根据函数模式识别列表类型的参数索引
std::vector<size_t> identifyListArgIndices(const c10::FunctionSchema& schema) {
  std::vector<size_t> list_indices;
  // 获取函数参数列表
  const auto& args = schema.arguments();
  // 遍历参数列表的索引
  for (const auto i : c10::irange(args.size())) {
    auto list_type = args[i].type()->castRaw<ListType>();
    // 检查参数是否为列表类型且元素类型为 Tensor
    if (list_type && list_type->getElementType()->castRaw<TensorType>()) {
      list_indices.push_back(i);
    }
  }
  return list_indices;
}

// 判断节点是否为张量列表构造
bool isTensorListConstruct(Node* node) {
  // 检查节点是否为 ListConstruct 类型
  if (node->kind() != prim::ListConstruct) {
    return false;
  }
  // 获取列表的类型信息
  const auto type = node->output()->type()->castRaw<ListType>();
  TORCH_CHECK(type != nullptr);
  const auto& elem_type = type->getElementType();
  // 判断列表元素类型是否为 Tensor
  return elem_type->castRaw<TensorType>();
}

// 类：变参更新器
class VariadicUpdater {
 public:
  // 构造函数，初始化图、操作类型和变参操作类型
  VariadicUpdater(
      std::shared_ptr<Graph> graph,
      NodeKind op,
      NodeKind variadic_op)
      : graph_(std::move(graph)),
        alias_db_(graph_),
        op_(op),
        variadic_op_(variadic_op) {}

  // 运行变参更新器
  bool run() {
    // 收集图中所有操作节点
    collectOpNodes(graph_->block());
    bool changed = false;
    // 替换操作节点中的非变参操作为变参操作
    for (auto n : op_nodes_) {
      changed |= replaceWithVariadicOp(n);
    }
    return changed;
  }

 private:
  // 记录操作节点的模式
  void recordSchema(Node* op_node) {
    // 获取操作节点的模式
    const auto& schema = op_node->schema();
    auto it = schema_to_list_indices_.find(schema.name());
    // 如果模式未记录，则记录模式及其列表类型参数索引
    if (it == schema_to_list_indices_.end()) {
      schema_to_list_indices_.emplace(
          schema.overload_name(), identifyListArgIndices(schema));
    }
  }

  // 获取操作节点中列表类型参数的索引
  const std::vector<size_t>& getListIndices(Node* op_node) const {
    const auto& schema = op_node->schema();
    auto it = schema_to_list_indices_.find(schema.overload_name());
    TORCH_CHECK(it != schema_to_list_indices_.end());
    return it->second;
  }

  // 收集图中的操作节点
  void collectOpNodes(Block* block) {
    // 遍历节点块中的所有节点
    for (auto node : block->nodes()) {
      // 检查节点是否为目标操作类型
      if (node->kind() == op_) {
        op_nodes_.push_back(node);
        // 记录操作节点的模式
        recordSchema(node);
      }
      // 递归处理节点块中的子块
      for (Block* b : node->blocks()) {
        collectOpNodes(b);
      }
    }
  }

  // 检查所有列表输入是否有效
  bool allListInputsAreValid(Node* op_node) {
    // 获取操作节点的输入数量
    const size_t num_inputs = op_node->inputs().size();
    // 遍历列表类型参数的索引
    for (const auto list_idx : getListIndices(op_node)) {
      TORCH_CHECK(list_idx < num_inputs);
      // 获取输入节点的列表
      const auto list = op_node->input(list_idx)->node();
      // 检查列表是否为张量列表构造，并且可以在操作节点之前移动
      if (!isTensorListConstruct(list) ||
          !alias_db_.couldMoveBeforeTopologically(list, op_node)) {
        return false;
      }
    }
    return true;
  }
  // 返回 true，表示操作成功完成
  return true;
}

// 将指定范围内的所有输入节点插入到输入向量中
void insertAllInputsBetween(
    std::vector<Value*>& inputs,    // 输入向量，将添加节点的输出值
    Node* node,                     // 节点，从它的输入中提取节点
    size_t start_idx,               // 起始索引，从该位置开始提取输入节点
    size_t end_idx) const {         // 结束索引，提取输入节点的终止位置
  const size_t num_inputs = node->inputs().size();  // 节点输入的总数
  TORCH_CHECK(start_idx <= end_idx && end_idx <= num_inputs);  // 检查索引的有效性
  inputs.insert(
      inputs.end(),
      node->inputs().begin() + start_idx,    // 将节点的输入从开始索引开始插入
      node->inputs().begin() + end_idx);     // 插入到结束索引的位置
}

// 向输入向量中插入整数类型的常量值节点
void insertIntegerInput(std::vector<Value*>& inputs, size_t input) {
  auto constant = graph_->create(prim::Constant);  // 创建常量节点
  constant->output()->setType(c10::IntType::get());  // 设置节点输出类型为整数
  constant->i_(attr::value, input);  // 设置常量节点的值为输入的整数值
  graph_->prependNode(constant);  // 在图中前置这个常量节点
  inputs.push_back(constant->output());  // 将常量节点的输出值添加到输入向量中
}

// 删除操作节点及其关联的列表节点
void deleteOpNodeAndLists(Node* op_node) {
  // 收集要销毁的列表节点
  std::vector<Node*> lists;
  const auto& list_indices = getListIndices(op_node);  // 获取操作节点关联的列表节点索引
  lists.reserve(list_indices.size());
  for (const size_t list_idx : list_indices) {
    auto* list = op_node->input(list_idx)->node();  // 获取列表节点
    lists.push_back(list);  // 将列表节点添加到列表中
  }

  GRAPH_UPDATE("Deleting\n", *op_node);  // 更新图，表示正在删除操作节点
  op_node->destroy();  // 销毁操作节点
  for (auto* list : lists) {
    if (!list->hasUses()) {  // 如果列表节点没有使用者
      GRAPH_UPDATE("Deleting\n", *list);  // 更新图，表示正在删除列表节点
      list->destroy();  // 销毁列表节点
    }
  }
}

// 使用变长操作节点替换原操作节点
bool replaceWithVariadicOp(Node* op_node) {
  if (!allListInputsAreValid(op_node)) {  // 检查所有列表输入是否有效
    return false;  // 若无效，则返回 false
  }

  std::vector<Value*> inputs;  // 输入值向量
  size_t cur_idx = 0;  // 当前索引
  std::vector<size_t> list_lens;  // 列表长度向量
  for (const size_t list_idx : getListIndices(op_node)) {
    insertAllInputsBetween(inputs, op_node, cur_idx, list_idx);  // 插入列表之前的输入节点
    const auto list = op_node->input(list_idx)->node();  // 获取列表节点
    const auto list_len = list->inputs().size();  // 获取列表节点的输入数量
    list_lens.push_back(list_len);  // 记录列表节点的输入数量
    insertAllInputsBetween(inputs, list, 0, list_len);  // 将列表节点的输入插入到输入向量中
    cur_idx = list_idx + 1;  // 更新当前索引
  }
  insertAllInputsBetween(inputs, op_node, cur_idx, op_node->inputs().size());  // 插入剩余的输入节点

  // 仅当有多个变长列表时，在参数列表末尾插入额外的整数
  if (list_lens.size() > 1) {
    for (const size_t list_len : list_lens) {
      insertIntegerInput(inputs, list_len);  // 插入列表长度的整数常量节点
    }
  }

  auto var_op_node = op_node->owningGraph()->create(variadic_op_, inputs);  // 创建变长操作节点
  var_op_node->output()->setType(op_node->output()->type());  // 设置变长操作节点的输出类型
  GRAPH_UPDATE("Adding\n", *var_op_node);  // 更新图，表示正在添加变长操作节点
  var_op_node->insertBefore(op_node);  // 在原操作节点之前插入变长操作节点
  GRAPH_UPDATE("Replacing\n", *op_node, "with\n", *var_op_node);  // 更新图，表示正在用变长操作节点替换原操作节点
  op_node->output()->replaceAllUsesWith(var_op_node->output());  // 替换原操作节点的所有使用者为变长操作节点的输出
  deleteOpNodeAndLists(op_node);  // 删除原操作节点及其关联的列表节点
  return true;  // 返回操作成功
}
};

} // namespace

// 判断是否使用变长操作来更新图中的节点
bool UseVariadicOp(
    const std::shared_ptr<Graph>& graph,  // 图的共享指针，表示要操作的图
    NodeKind op,                          // 节点的种类，表示要被替换的操作
    NodeKind variadic_op) {               // 节点的种类，表示要替换成的变长操作
  // 创建包含变长操作名称的通行证名字
  const std::string pass_name = std::string("variadic ") + op.toQualString();
  // 在应用变长操作前输出图的状态
  GRAPH_DUMP("Before " + pass_name, graph);
  // 使用变长操作更新图中的节点，返回是否有改变
  bool changed = VariadicUpdater(graph, op, variadic_op).run();
  if (changed) {
    // 如果有改变，则对图进行常量池化处理
    ConstantPooling(graph);
    // 在应用变长操作后输出图的状态
    GRAPH_DUMP("After " + pass_name, graph);
  }
  // 返回是否有改变
  return changed;
}

// 移除列表突变并使用变长操作来更新图中的节点
bool RemoveListMutationAndUseVariadicOp(
    const std::shared_ptr<Graph>& graph,  // 图的共享指针，表示要操作的图
    NodeKind op,                          // 节点的种类，表示要被替换的操作
    NodeKind variadic_op) {               // 节点的种类，表示要替换成的变长操作
  bool changed_in_last_iter = true;       // 标记上一轮迭代是否有改变
  bool changed = false;                   // 标记是否有改变
  while (changed_in_last_iter) {
    // 移除图中的列表突变
    changed_in_last_iter = RemoveListMutation(graph);
    // 使用变长操作更新图中的节点，并更新迭代改变标志
    changed_in_last_iter =
        UseVariadicOp(graph, op, variadic_op) || changed_in_last_iter;
    // 更新总的改变标志
    changed = changed || changed_in_last_iter;
  }
  // 返回是否有改变
  return changed;
}

// 使用变长操作at::cat来更新图中的节点
bool UseVariadicCat(const std::shared_ptr<Graph>& graph) {
  return UseVariadicOp(graph, aten::cat, prim::VarConcat);
}

// 移除列表突变并使用变长操作at::cat来更新图中的节点
bool RemoveListMutationAndUseVariadicCat(const std::shared_ptr<Graph>& graph) {
  return RemoveListMutationAndUseVariadicOp(graph, aten::cat, prim::VarConcat);
}

// 使用变长操作at::stack来更新图中的节点
bool UseVariadicStack(const std::shared_ptr<Graph>& graph) {
  return UseVariadicOp(graph, aten::stack, prim::VarStack);
}

// 移除列表突变并使用变长操作at::stack来更新图中的节点
bool RemoveListMutationAndUseVariadicStack(
    const std::shared_ptr<Graph>& graph) {
  return RemoveListMutationAndUseVariadicOp(graph, aten::stack, prim::VarStack);
}

} // namespace jit
} // namespace torch
```