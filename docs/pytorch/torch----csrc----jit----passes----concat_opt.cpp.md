# `.\pytorch\torch\csrc\jit\passes\concat_opt.cpp`

```py
// 包含 Torch 的 JIT 编译器优化相关头文件
#include <torch/csrc/jit/passes/concat_opt.h>

// C++ 标准库头文件
#include <algorithm>  // 包含算法库，用于标准算法操作
#include <deque>      // 包含双端队列，支持快速随机插入和删除
#include <unordered_map>  // 包含无序映射，实现哈希表
#include <unordered_set>  // 包含无序集合，实现哈希集合
#include <vector>      // 包含向量，用于动态数组操作

// Torch 的 C10 实用工具库
#include <c10/util/ssize.h>

// Torch JIT 编译器中的 IR（Intermediate Representation）相关头文件
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/named_value.h>

// Torch JIT 编译器日志相关
#include <torch/csrc/jit/jit_log.h>

// Torch JIT 编译器优化 passes 相关头文件
#include <torch/csrc/jit/passes/constant_pooling.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/remove_mutation.h>

// Torch JIT 运行时图遍历器
#include <torch/csrc/jit/runtime/graph_iterator.h>

// Torch 命名空间
namespace torch {
namespace jit {

// 匿名命名空间，内部实现细节隐藏
namespace {

// 从图中移除 cat 节点的函数
void removeCatNodeFromGraph(Node* n) {
  // 断言节点类型为 aten::cat
  TORCH_INTERNAL_ASSERT(n->kind() == aten::cat);
  // 获取输入列表
  auto inp_list = n->input(0);
  // 记录图更新日志
  GRAPH_UPDATE("Deleting\n", *n);
  // 销毁节点
  n->destroy();
  // 如果输入列表没有被使用，删除其节点
  if (!inp_list->hasUses()) {
    GRAPH_UPDATE("Deleting\n", *inp_list->node());
    inp_list->node()->destroy();
  }
}

// 比较两个值列表是否相等的函数
bool equal(at::ArrayRef<Value*> list1, at::ArrayRef<Value*> list2) {
  return list1.size() == list2.size() &&
      std::equal(list1.begin(), list1.end(), list2.begin());
}

// 消除共同输入的 concat 的类
class ConcatCommonInputsEliminator {
 public:
  // 构造函数，初始化使用图的智能指针
  explicit ConcatCommonInputsEliminator(std::shared_ptr<Graph> graph)
      : graph_(std::move(graph)) {}

  // 执行消除共同输入的主函数
  bool run() {
    // 处理整个图的块
    handleBlock(graph_->block());
    // 后处理，完成消除共同输入操作
    return postprocess();
  }

 private:
  // 处理图块的递归函数
  void handleBlock(Block* block) {
    // 遍历块中的每一个节点
    for (auto node : block->nodes()) {
      // 如果节点是 prim::VarConcat 类型，处理 cat 节点
      if (node->kind() == prim::VarConcat) {
        handleCat(node);
      }
      // 递归处理节点的子块
      for (Block* block : node->blocks()) {
        handleBlock(block);
      }
    }
  }

  // 处理 cat 节点的函数
  void handleCat(Node* node) {
    // 记录调试信息，表示正在考虑的 cat 节点
    GRAPH_DEBUG("Considering cat node for CSE opt: ", node);

    // 获取当前所有输入
    auto curr_all_inputs = node->inputs();
    // 获取当前张量输入列表
    auto curr_tensor_inputs =
        curr_all_inputs.slice(0, curr_all_inputs.size() - 1);
    // 获取当前维度输入
    auto curr_dim = curr_all_inputs.back();

    // 如果输出没有写操作，将该 cat 节点插入到共享列表中
    if (!getOrCreateAliasDb()->hasWriters(node->output())) {
      concated_outputs_.insert(node);
    }

    // 如果张量输入的数量小于等于 2，无法优化
    if (curr_tensor_inputs.size() <= 2) {
      return;
    }

    // 现在，检查前 N-1 个元素在任何先前的 cat 操作中是否出现
    // 示例:
    //    %11 = prim::VarConcat(%0, %1, <dim>)
    //    ...
    //    %13 = prim::VarConcat(%0, %1, %2, <dim>) // 前两个输入与 %11 相同
    //    ...
    //        = %13 ... // 使用 %13
    //
    // 在 CSE 优化后:
    // 将当前张量输入的前缀切片，排除最后一个元素
    auto curr_tensor_inputs_prefix =
        curr_tensor_inputs.slice(0, curr_tensor_inputs.size() - 1);
    // 遍历已连接输出的列表
    for (const auto& prev : concated_outputs_) {
      // 获取先前连接操作的所有输入
      auto prev_all_inputs = prev->inputs();
      // 对先前连接操作的张量输入进行切片，排除最后一个元素
      auto prev_tensor_inputs =
          prev_all_inputs.slice(0, prev_all_inputs.size() - 1);
      // 获取先前连接操作的维度参数
      auto prev_dim = prev_all_inputs.back();
      // 检查当前张量输入的前缀和先前连接操作的张量输入是否相等，并且维度相同
      if (equal(curr_tensor_inputs_prefix, prev_tensor_inputs) &&
          curr_dim == prev_dim) {
        // 如果当前节点不被先前的节点所支配，则跳过
        if (!node->isDominatedBy(prev)) {
          // 如果当前连接节点不被先前的连接节点支配，则不能使用先前的连接输出
          continue;
        }

        // 创建新的输入向量，替换第一个输入为先前连接操作的输出
        std::vector<Value*> new_inputs = {
            prev->output(), curr_tensor_inputs.back(), curr_dim};
        // 在当前节点所属图中创建一个新的 VarConcat 节点
        auto new_concat =
            node->owningGraph()->create(prim::VarConcat, new_inputs);
        // 设置新节点的输出类型与当前节点的输出类型相同
        new_concat->output()->setType(node->output()->type());
        // 将当前节点需要替换的连接操作映射到新的连接节点
        concats_to_replace_[node] = new_concat;
        // 返回，表示已经找到可以替换的连接操作
        return;
      }
    }

    // 现在，检查当前张量输入的后缀（排除第一个元素）是否出现在任何先前的连接操作中
    auto curr_tensor_inputs_suffix =
        curr_tensor_inputs.slice(1, curr_tensor_inputs.size() - 1);
    // 再次遍历已连接输出的列表
    for (const auto& prev : concated_outputs_) {
      // 获取先前连接操作的所有输入
      auto prev_all_inputs = prev->inputs();
      // 对先前连接操作的张量输入进行切片，排除最后一个元素
      auto prev_tensor_inputs =
          prev_all_inputs.slice(0, prev_all_inputs.size() - 1);
      // 获取先前连接操作的维度参数
      auto prev_dim = prev_all_inputs.back();
      // 检查当前张量输入的后缀和先前连接操作的张量输入是否相等，并且维度相同
      if (equal(curr_tensor_inputs_suffix, prev_tensor_inputs) &&
          curr_dim == prev_dim) {
        // 如果当前节点不被先前的节点所支配，则跳过
        if (!node->isDominatedBy(prev)) {
          // 如果当前连接节点不被先前的连接节点支配，则不能使用先前的连接列表
          continue;
        }

        // 创建新的输入向量，替换第二个输入为先前连接操作的输出
        std::vector<Value*> new_inputs = {
            curr_tensor_inputs.front(), prev->output(), curr_dim};
        // 在当前节点所属图中创建一个新的 VarConcat 节点
        auto new_concat =
            node->owningGraph()->create(prim::VarConcat, new_inputs);
        // 设置新节点的输出类型与当前节点的输出类型相同
        new_concat->output()->setType(node->output()->type());
        // 将当前节点需要替换的连接操作映射到新的连接节点
        concats_to_replace_[node] = new_concat;
        // 返回，表示已经找到可以替换的连接操作
        return;
      }
    }

    // 是否需要处理其他情况，例如 N-2 或更少元素的情况？
  // 检查之前的连接操作中是否出现了输入
  // 待办事项

  bool postprocess() {
    // 替换已标记的列表节点
    bool changed = false;
    for (auto it : concats_to_replace_) {
      auto curr_node = it.first;
      auto new_node = it.second;
      GRAPH_UPDATE("在", *new_node, "之前插入\n", *curr_node);
      new_node->insertBefore(curr_node);
      GRAPH_UPDATE("用", *new_node, "替换", *curr_node, "的使用");
      curr_node->output()->replaceAllUsesWith(new_node->output());
      GRAPH_UPDATE("删除", *curr_node);
      curr_node->destroy();
      changed = true;
    }
    return changed;
  }

  AliasDb* getOrCreateAliasDb() {
    if (!aliasDb_) {
      aliasDb_ = std::make_unique<AliasDb>(graph_);
    }
    return aliasDb_.get();
  }

  std::shared_ptr<Graph> graph_;
  std::unique_ptr<AliasDb> aliasDb_ = nullptr;

  std::unordered_set<Node*> concated_outputs_;
  std::unordered_map<Node*, Node*> concats_to_replace_;
};

} // namespace

// 在图中消除 Concat 节点的共同输入
bool EliminateConcatCommonInputs(const std::shared_ptr<Graph>& graph) {
  // 在消除 Concat 共同输入之前，对图进行转储
  GRAPH_DUMP("Before eliminating Concat common inputs", graph);
  // 运行 Concat 共同输入消除器，并检查是否有更改
  bool changed = ConcatCommonInputsEliminator(graph).run();
  if (changed) {
    // 如果有更改，则再次对图进行转储
    GRAPH_DUMP("After eliminating Concat common inputs", graph);
  }
  return changed;
}

namespace {

class ConcatExpander {
 public:
  explicit ConcatExpander(std::shared_ptr<Graph> graph)
      : graph_(std::move(graph)) {}

  // 执行 Concat 节点的扩展
  void run() {
    // 处理整个图的根块
    handleBlock(graph_->block());
    // 清理扩展后的 Concat 操作
    cleanupExpandedCatOps();
    // 转储图，显示重新使用复制缓冲区之前的状态
    GRAPH_DUMP("Before reusing copy buffers: ", graph_);
    // 在复制操作中重用缓冲区
    reuseBuffersInCopies();
  }

 private:
  // 递归处理块中的节点
  void handleBlock(Block* block) {
    for (auto node : block->nodes()) {
      // 如果节点是 aten::cat 操作，则进行扩展
      if (node->kind() == aten::cat) {
        expandCat(node);
      }
      // 递归处理节点的子块
      for (Block* block : node->blocks()) {
        handleBlock(block);
      }
    }
  }

  // 扩展 cat 节点为多个 copy 节点
  //
  // 示例：
  //     %2 = aten::clamp(%0, ...)
  //     %3 = aten::clamp(%1, ...)
  //     %10 = prim::ListConstruct(%2, %3)
  //     %11 = aten::cat(%10, ...)
  //     ...
  //         = %11 ... // 使用 %11
  //
  // 扩展后：
  //     %2 = aten::clamp(%0, ...)
  //     %3 = aten::clamp(%1, ...)
  //     %20 = aten::empty(...)          // cat 输出缓冲区
  //     %21 = aten::slice(%20, ...)     // %2 的切片
  //     %22 = aten::copy_(%21, %2)      // 复制 %2
  //     %23 = aten::slice(%20, ...)     // %3 的切片
  //     %24 = aten::copy_(%23, %3)      // 复制 %3
  //     ...
  //         = %20 ... // 使用 %20 替代 %11
  void expandCat(Node* node) {
    // 输出调试信息，显示考虑进行扩展的 cat 节点
    GRAPH_DEBUG("Considering cat node for expansion: ", node);
    // 如果 cat 节点的输入在图中被改变，则不优化该节点
    // TODO: 通过检查应用该优化的图区域来改进此处的优化判断逻辑
    if (getOrCreateAliasDb()->hasWriters(node->input(0))) {
      return;
    }
    // 如果 cat 节点的输入不是 prim::ListConstruct，则无法优化
    if (node->input(0)->node()->kind() != prim::ListConstruct) {
      // 输入到 `cat` 操作的未知形式
      return;
    }
    // 如果 `cat` 操作的形状未知，则无法扩展
    if (!allShapesAreKnown(node)) {
      return;
    }
    // 如果 `cat` 操作的输入形状未知，则无法扩展
    for (auto cat_inp : node->input(0)->node()->inputs()) {
      if (!shapeIsKnown(cat_inp)) {
        return;
      }
    }
    // TODO: 处理非连续的张量。
    // 例如，如何处理所有输入都是通道最后的情况？

    auto maybe_cat_dim = constant_as<int64_t>(node->input(1));
    // 如果 `cat` 维度不是常量，则无法扩展
    if (!maybe_cat_dim) {
      return;
    }
    auto cat_dim_value = maybe_cat_dim.value();
    auto cat_dim = node->input(1);

    // 将插入点设置为当前的 `cat` 节点
    WithInsertPoint guard(node);
    // 插入常量节点
    auto none = graph_->insertConstant(IValue());
    auto one = graph_->insertConstant(1);
    // 插入用于 `cat` 输出缓冲区大小的常量。
    auto tensortype = node->output()->type()->expect<TensorType>();
    TORCH_INTERNAL_ASSERT(tensortype);
    auto tensortype_sizes = tensortype->sizes();
    std::vector<Value*> cat_out_size;
    for (size_t i = 0; i < tensortype_sizes.size(); ++i) {
      cat_out_size.push_back(graph_->insertConstant(tensortype_sizes[i]));
    }
    
    // 创建一个整数列表，用于 `cat` 输出缓冲区大小。
    auto cat_out_size_list = graph_->createList(IntType::get(), cat_out_size);
    cat_out_size_list->insertBefore(node);
    
    // 创建一个空缓冲区，用作 `cat` 的输出缓冲区。
    // TODO: 处理具有不同 dtype、布局、设备、内存格式等的张量。
    auto cat_out_empty = graph_->create(
        aten::empty,
        {cat_out_size_list->output(), none, none, none, none, none});
    cat_out_empty->insertBefore(node);
    
    // 对于这个 `cat` 节点的每个输入：
    //   * 创建一个 `cat` 输出缓冲区的切片。
    auto cat_out_value = cat_out_empty->output();
    auto cat_inp_list = node->input(0)->node();
    int start_idx = 0;
    auto start = graph_->insertConstant(start_idx);
    for (auto cat_inp : cat_inp_list->inputs()) {
      // 创建与此输入大小和位置相对应的 `cat` 输出缓冲区的切片。
      auto cat_inp_tensor_type =
          dynamic_cast<TensorType*>(cat_inp->type().get());
      TORCH_INTERNAL_ASSERT(cat_inp_tensor_type);
      TORCH_INTERNAL_ASSERT(cat_inp_tensor_type->dim());
      auto cat_inp_tensortype_sizes = cat_inp_tensor_type->sizes();
      // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
      int end_idx = start_idx + *cat_inp_tensortype_sizes[cat_dim_value];
      auto end = graph_->insertConstant(end_idx);
    
      auto slice = graph_->create(
          aten::slice, {cat_out_value, cat_dim, start, end, one});
      GRAPH_UPDATE("Inserting\n", *slice, "before\n", *node);
      slice->insertBefore(node);
      slices_added_.push_back(slice);
    
      // 将此输入复制到输出切片。
      auto copy = graph_->create(aten::copy_, {slice->output(), cat_inp});
      GRAPH_UPDATE("Inserting\n", *copy, "before\n", *node);
      copy->insertBefore(node);
      copies_added_.push_back(copy);
    
      start_idx = end_idx;
      start = end;
    }
    
    // 用 `cat` 输出缓冲区替换 `cat` 节点的使用。
    replace_uses_with_[node->output()] = cat_out_value;
    nodes_to_remove_.insert(node);
    }
    
    // 检查值的形状是否已知。
    bool shapeIsKnown(Value* v) {
      if (v->type()->cast<TensorType>()) {
        if (!v->isCompleteTensor()) {
          return false;
        }
        if (*v->type()->castRaw<TensorType>()->dim() == 0) {
          return false;
        }
      }
      return true;
    }
    
    // 检查节点的所有形状是否已知。
    // TODO: 放宽检查以支持动态形状
    bool allShapesAreKnown(Node* node) {
    // 遍历节点的输入值列表
    for (Value* input : node->inputs()) {
      // 如果输入值的形状未知，则返回 false
      if (!shapeIsKnown(input)) {
        return false;
      }
    }
    // 遍历节点的输出值列表
    for (Value* output : node->outputs()) {
      // 如果输出值的形状未知，则返回 false
      if (!shapeIsKnown(output)) {
        return false;
      }
    }
    // 如果所有输入和输出值的形状都已知，则返回 true
    return true;
  }

  // 清理已展开的 cat 操作
  void cleanupExpandedCatOps() {
    // 遍历替换使用映射表中的元素
    for (auto it : replace_uses_with_) {
      // 更新图信息，显示正在替换的节点和替换后的节点
      GRAPH_UPDATE(
          "Replacing uses of\n",
          *it.first->node(),
          "with\n",
          *it.second->node());
      // 用替换后的节点替换所有使用到的原节点
      it.first->replaceAllUsesWith(it.second);
    }
    // 遍历需要从图中移除的节点列表
    for (auto n : nodes_to_remove_) {
      // 从图中移除 cat 节点
      removeCatNodeFromGraph(n);
    }
  }

  // 将节点移动到另一个节点之前
  void moveBefore(Node* node, Node* before) {
    // 若要将一个节点移动到另一个节点之前，需先移动所有它依赖的节点
    for (auto inp : node->inputs()) {
      // 递归调用，移动当前节点的输入节点
      moveBefore(inp->node(), before);
    }
    // 实际将当前节点移动到指定节点之前
    node->moveBefore(before);
  }

  // 在拷贝操作中重用缓冲区
  //
  // 例如，考虑以下操作序列：
  //     %10 = prim::ListConstruct(%0, %1)
  //     %11 = aten::cat(%10, ...)
  //     ...
  //     %12 = prim::ListConstruct(%11, %2)  // 使用上述 cat 的结果
  //     %13 = aten::cat(%12, ...)
  //
  // 一旦这些 cat 操作展开为拷贝操作，我们会得到两个缓冲区；一个是 %11 的，
  // 另一个是 %13 的。通过仅使用一个缓冲区来优化，可以将 %11 的缓冲区设为 %13 的视图（切片）。
  //
  // 如果之前添加的任何拷贝操作的源为 `aten::empty`，那么这些情况可以用单一缓冲区替代。
  //
  // 示例：
  //     %20 = aten::empty(...)          // cat.1 的输出缓冲区
  //     %21 = aten::slice(%20, ...)
  //     %22 = aten::copy_(%21, %2)
  //     %23 = aten::slice(%20, ...)
  //     %24 = aten::copy_(%23, %3)
  //     ...
  //     %30 = aten::empty(...)          // cat.2 的输出缓冲区
  //     %31 = aten::slice(%30, ...)
  //     %32 = aten::copy_(%31, %20)     // 拷贝的源为 aten::empty，因此重用此缓冲区
  //     %33 = aten::slice(%30, ...)
  //     %34 = aten::copy_(%33, %4)
  //
  // 重用拷贝操作中的缓冲区后：
  //     %30 = aten::empty(...)          // cat.2 的输出缓冲区
  //     %31 = aten::slice(%30, ...)     // 将 %31 和其输入移动到 %20 之前
  //     %21 = aten::slice(%31, ...)     // 在 %20 的位置使用 %31
  //     %22 = aten::copy_(%21, %2)
  //     %23 = aten::slice(%31, ...)     // 在 %20 的位置使用 %31
  //     %24 = aten::copy_(%23, %3)
  //     ...
  //     ...                             // 现在删除了对 %31 的拷贝
  //     %33 = aten::slice(%30, ...)
  //     %34 = aten::copy_(%33, %4)
  void reuseBuffersInCopies() {
    for (auto copy : copies_added_) {
      // 遍历所有添加的复制节点
      auto src = copy->input(1);
      // 获取当前复制节点的第二个输入作为源节点
      auto dst = copy->input(0);
      // 获取当前复制节点的第一个输入作为目标节点
      if (src->node()->kind() != aten::empty) {
        // 如果源节点的操作类型不是 aten::empty，则跳过当前循环
        continue;
      }

      // 将目标节点移动到源节点之前
      GRAPH_UPDATE("Moving\n", *dst->node(), "before\n", *src->node());
      moveBefore(dst->node(), src->node());

      // 用目标节点替换源节点的所有使用
      GRAPH_UPDATE("Replacing\n", *src->node(), "with\n", *dst->node());
      src->replaceAllUsesWith(dst);

      // 销毁源节点
      GRAPH_UPDATE("Deleting\n", *src->node());
      src->node()->destroy();

      // 销毁当前复制节点
      GRAPH_UPDATE("Deleting\n", *copy);
      copy->destroy();
    }
  }

  // 获取或创建别名分析数据库
  AliasDb* getOrCreateAliasDb() {
    if (!aliasDb_) {
      aliasDb_ = std::make_unique<AliasDb>(graph_);
    }
    return aliasDb_.get();
  }

  std::shared_ptr<Graph> graph_;
  std::unique_ptr<AliasDb> aliasDb_ = nullptr;

  std::unordered_set<Node*> nodes_to_remove_;
  std::unordered_map<Value*, Value*> replace_uses_with_;
  std::vector<Node*> copies_added_;
  std::vector<Node*> slices_added_;
};

} // namespace

// 对图中的 Concat 进行扩展、消除冗余操作
void ExpandConcatAndEliminateRedundancy(const std::shared_ptr<Graph>& graph) {
  // 创建 ConcatExpander 对象，并执行扩展操作
  ConcatExpander(graph).run();
  // 在图中记录操作后的状态
  GRAPH_DUMP("After expanding Concat and eliminating redundancy", graph);
}

namespace {

// 确定值在节点使用中的索引
size_t determineUsageIdx(Value* value, Node* user) {
  // 在用户节点的输入中查找值的索引
  const auto idx =
      std::find(user->inputs().begin(), user->inputs().end(), value) -
      user->inputs().begin();
  using c10::ssize;
  // 检查索引是否有效
  TORCH_CHECK(idx != ssize(user->inputs()));
  return idx;
}

// 获取 Concat 节点的输入值列表
std::vector<Value*> getConcatInputs(Node* concat) {
  // 检查节点是否为 aten::cat 类型
  TORCH_CHECK(concat->kind() == aten::cat);
  auto* list = concat->input(0);
  auto* list_construct = list->node();
  // 检查列表构造节点是否为 prim::ListConstruct 类型
  TORCH_CHECK(list_construct->kind() == prim::ListConstruct);
  // 返回列表构造节点的输入列表
  return list_construct->inputs().vec();
}

// ConcatCombiner 类，用于组合和优化 Concat 操作
class ConcatCombiner {
 public:
  explicit ConcatCombiner(std::shared_ptr<Graph> graph)
      : graph_(std::move(graph)), aliasDb_(graph_) {}

  // 运行组合和优化操作
  bool run() {
    // 收集可优化的 Concat 操作
    collectOptimizableConcats();
    // 组合可优化的 Concat 操作
    bool changed = combineConcats();
    // 如果有改变，则消除死代码
    if (changed) {
      EliminateDeadCode(graph_);
    }
    return changed;
  }

 private:
  // 处理单个 Concat 节点，判断是否可以优化
  void handleConcat(Node* node) {
    auto* list = node->input(0);
    auto* list_node = list->node();

    const auto dim_opt = toIValue(node->input(1));
    // 需要能够静态确定维度才能与另一个 Concat 匹配
    if (!dim_opt || !dim_opt->isInt()) {
      return;
    }
    const auto dim = dim_opt->toInt();

    // 检查此节点的输入是否为未修改的列表构造
    if (list_node->kind() != prim::ListConstruct ||
        !aliasDb_.couldMoveBeforeTopologically(list_node, node)) {
      return;
    }

    // 检查此节点的唯一输出是否在未修改的列表构造中使用
    const auto& concat_uses = node->output()->uses();
    if (concat_uses.size() != 1) {
      return;
    }

    auto* next_list = concat_uses[0].user;
    if (next_list->kind() != prim::ListConstruct) {
      return;
    }

    const auto& next_list_uses = next_list->output()->uses();
    if (next_list_uses.size() != 1) {
      return;
    }

    auto* next_concat = next_list_uses[0].user;

    if (next_concat->kind() == aten::cat) {
      // 维度必须能够静态确定，并且与之前看到的维度匹配
      const auto next_dim_opt = toIValue(next_concat->input(1));
      if (!next_dim_opt || next_dim_opt->toInt() != dim) {
        return;
      }
      // 将可组合的 Concat 对添加到 combinable_concats_ 中
      combinable_concats_.emplace_back(
          node, next_concat, determineUsageIdx(node->output(), next_list));
    }
  }

  // 收集可优化的 Concat 操作
  void collectOptimizableConcats() {
    DepthFirstGraphNodeIterator graph_it(graph_);
    for (auto* node = graph_it.next(); node != nullptr;
         node = graph_it.next()) {
      if (node->kind() == aten::cat) {
        handleConcat(node);
      }
    }
  }

  Node* createListConstruct(const std::deque<Value*>& inputs) {
    // 创建一个新的 ListConstruct 节点，并设置为当前图的输出节点
    auto* output = graph_->create(prim::ListConstruct);

    // 将输入列表中的每个节点作为输入添加到新创建的 ListConstruct 节点中
    for (auto* v : inputs) {
      output->addInput(v);
    }

    // 返回新创建的 ListConstruct 节点作为函数的输出
    return output;
  }

  // 定义一个别名，表示每个节点与其新列表输入之间的映射关系
  using ListConstructInputs = std::shared_ptr<std::deque<Value*>>;

  // 构建一个映射表，将每个可合并的连接节点映射到其新的列表输入
  // 使用 std::deque 以便能够在常数时间内在列表开头进行插入操作
  std::unordered_map<Node*, ListConstructInputs> getListConstructInputs() {
    // 当前连接节点的映射表
    std::unordered_map<Node*, ListConstructInputs> cur_list_construct_inputs;

    // 遍历所有可合并的连接节点
    for (const auto& combinable : combinable_concats_) {
      // 获取第二个连接节点的输入列表
      const auto& inputs_to_add = getConcatInputs(combinable.second_concat);

      // 查找第一个连接节点在映射表中的位置
      auto it = cur_list_construct_inputs.find(combinable.first_concat);
      std::shared_ptr<std::deque<Value*>> cur_list;
      if (it != cur_list_construct_inputs.end()) {
        cur_list = it->second;
        // 移除第一个连接节点的映射，因为其所有输入都将移动到第二个连接节点
        cur_list_construct_inputs.erase(combinable.first_concat);
      } else {
        cur_list = std::make_shared<std::deque<Value*>>();
      }

      // 将第二个连接节点和其输入列表添加到映射表中
      cur_list_construct_inputs.emplace(combinable.second_concat, cur_list);

      // 如果当前列表不为空，则保证它已经包含第一个连接节点的所有输入
      if (cur_list->empty()) {
        // 获取第一个连接节点的起始值，并将其添加到当前列表的末尾
        const auto& starting_values = getConcatInputs(combinable.first_concat);
        cur_list->insert(
            cur_list->end(), starting_values.begin(), starting_values.end());
      }

      // 在当前列表的开头插入第二个连接节点的部分输入
      cur_list->insert(
          cur_list->begin(),
          inputs_to_add.begin(),
          inputs_to_add.begin() + combinable.idx);

      // 在当前列表的末尾插入第二个连接节点的其余输入
      cur_list->insert(
          cur_list->end(),
          inputs_to_add.begin() + combinable.idx + 1,
          inputs_to_add.end());
    }

    // 返回包含所有连接节点及其新输入列表的映射表
    return cur_list_construct_inputs;
  }

  // 尝试合并可合并的连接节点，如果没有可合并的节点则返回 false
  bool combineConcats() {
    if (combinable_concats_.empty()) {
      return false;
    }

    // 获取所有连接节点及其新列表输入的映射表
    auto list_construct_inputs = getListConstructInputs();

    // 遍历映射表中的每个节点及其新输入列表
    for (const auto& node_and_new_list : list_construct_inputs) {
      auto* node = node_and_new_list.first;
      auto& inputs = node_and_new_list.second;

      // 创建一个新的 ListConstruct 节点，使用当前节点的新输入列表
      auto* new_list_construct = createListConstruct(*inputs);

      // 获取当前节点输入的原始 ListConstruct 节点
      auto* old_list_construct = node->input(0)->node();

      // 设置新 ListConstruct 节点的输出类型与原始节点相同
      new_list_construct->output()->setType(
          old_list_construct->output()->type());

      // 将新 ListConstruct 节点插入到当前节点之前
      new_list_construct->insertBefore(node);

      // 替换原始节点的所有使用情况为新创建的 ListConstruct 节点
      old_list_construct->replaceAllUsesWith(new_list_construct);
    }

    // 成功合并连接节点，返回 true
    return true;
  }
    return true;
  }

  // 表示可优化的连接节点对。
  // - first_concat 必须在 second_concat 之前出现
  // - idx 是 first_concat 的输入应该插入 second_concat 新输入的索引位置
  // 示例:
  //    %inputs.1 = prim::ListConstruct(%0, %0)
  //    %concat.1 = aten::cat(%inputs.1, %dim)
  //    %inputs.2 = prim::ListConstruct(%1, %concat.1, %1)
  //    %concat.2 = aten::cat(%inputs.2, %dim)
  // -> first_concat = &concat.1, second_concat = &concat.2, idx = 1
  struct CombinableConcat {
    // 构造函数，初始化可优化连接的信息
    CombinableConcat(Node* a, Node* b, size_t i)
        : first_concat(a), second_concat(b), idx(i) {}

    Node* first_concat;   // 第一个连接节点指针
    Node* second_concat;  // 第二个连接节点指针
    size_t idx;           // 插入位置索引
  };

  std::vector<CombinableConcat> combinable_concats_;  // 可优化连接节点的向量列表

  std::shared_ptr<Graph> graph_;  // 共享指针，指向图对象
  AliasDb aliasDb_;               // 别名数据库对象
};

} // namespace



// 结束了一个名为 namespace 的命名空间定义

bool CombineConcats(const std::shared_ptr<Graph>& graph) {
  // 创建一个 ConcatCombiner 对象，将图作为参数传入并执行其 run 方法，获取是否发生了变化的布尔值
  bool changed = ConcatCombiner(graph).run();
  // 在图执行连接操作合并后，打印图的状态信息，包括合并连接操作后的结果
  GRAPH_DUMP("After combining concats", graph);
  // 返回合并连接操作是否修改了图的布尔值结果
  return changed;
}

} // namespace jit
} // namespace torch



// 结束了名为 jit 和 torch 的命名空间定义
```