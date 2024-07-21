# `.\pytorch\torch\csrc\jit\passes\peephole_list_idioms.cpp`

```
// 包含必要的头文件：ATen 库的 JIT 类型、Torch 的 IR 分析、IR 视图、JIT 日志、死代码消除、
// 窥孔优化、列表惯用语的窥孔优化、值细化工具、图执行器、切片索引调整等
#include <ATen/core/jit_type.h>
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/ir/ir_views.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/passes/peephole.h>
#include <torch/csrc/jit/passes/peephole_list_idioms.h>
#include <torch/csrc/jit/passes/value_refinement_utils.h>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <torch/csrc/jit/runtime/slice_indices_adjust.h>
#include <limits>
#include <utility>

// 命名空间 torch::jit 下的声明
namespace torch {
namespace jit {

// normalizeIndex 函数定义，用于规范化索引
static std::optional<size_t> normalizeIndex(int64_t index, size_t len) {
  // 如果索引为负数，则转换为正数索引
  if (index < 0) {
    index = index + len;
  }
  // 如果索引在有效范围内，返回规范化后的索引作为 optional
  if (index >= 0 && index < static_cast<int64_t>(len)) {
    return index;
  } else {
    // 否则返回空 optional
    return c10::nullopt;
  }
}

// ListLenRefiner 结构体定义，用于列表长度的精细化处理
struct ListLenRefiner {
  // 构造函数，接受图和已变异列表的共享指针
  ListLenRefiner(
      std::shared_ptr<Graph> graph,
      std::unordered_set<Value*>& mutated_lists)
      : graph_(std::move(graph)), mutated_lists_(mutated_lists) {}

  // 运行精细化处理
  bool run() {
    // 存储使用了 len() 的列表的集合
    std::unordered_set<Value*> li_with_len_use;
    // 收集需要精细化处理的列表
    collectListsToRefine(graph_->block(), li_with_len_use);
    // 如果没有需要精细化处理的列表，则返回 false
    if (lists_to_refine_.empty()) {
      return false;
    }
    // 创建列表精细化对象
    ListRefinement refinements;
    // 在图中执行列表长度的精细化处理
    RefineListLens(graph_->block(), std::move(refinements));
    // 返回处理是否改变了图
    return changed_;
  }

  // 收集需要进行精细化处理的列表
  void collectListsToRefine(
      Block* b,
      std::unordered_set<Value*>& li_with_len_use) {
    // 遍历节点块中的每个节点
    for (Node* n : b->nodes()) {
      // 对于包含块的节点，递归收集列表
      for (Block* block : n->blocks()) {
        collectListsToRefine(block, li_with_len_use);
      }

      // 如果节点不是 aten::len 类型，则继续下一个节点
      if (n->kind() != aten::len) {
        continue;
      }

      // 获取 len() 函数的第一个输入值
      auto first_input = n->input(0);
      // 如果输入是列表类型，并且没有被修改过
      if (first_input->type()->castRaw<ListType>() &&
          !mutated_lists_.count(first_input)) {
        // 如果列表尚未记录使用 len()，则记录该列表
        if (!li_with_len_use.count(first_input)) {
          li_with_len_use.insert(first_input);
        } else {
          // 否则将列表添加到需要精细化处理的列表集合中
          lists_to_refine_.insert(first_input);
        }
      }
    }
  }

  // 在节点块中进行列表长度的精细化处理
  ListRefinement RefineListLens(Block* b, ListRefinement block_refinements) {
    // 将块的精细化处理对象添加到活动精细化处理列表中
    active_refinements_.push_back(&block_refinements);
    // ...
    // 遍历指针 b 所指向对象的所有节点
    for (Node* n : b->nodes()) {
      // 检查节点 n 是否匹配特定的操作符类型，如等于或不等于操作
      if (n->matches("aten::eq(int a, int b) -> bool") ||
          n->matches("aten::ne(int a, int b) -> bool")) {
        // 对于输入常量和来自 len(li) 的另一个输入进行检查
        for (size_t const_index : {0, 1}) {
          // 尝试将输入转换为 int64_t 类型的常量
          auto ival = constant_as<int64_t>(n->input(const_index));
          // 如果转换失败则继续下一个循环
          if (!ival) {
            continue;
          }
          // 获取另一个输入节点 li_len
          auto li_len = n->input(1 - const_index);
          // 检查节点 li_len 是否匹配特定的操作符类型，并且它是否在 lists_to_refine_ 集合中
          if (!li_len->node()->matches("aten::len.t(t[] a) -> int") ||
              !lists_to_refine_.count(li_len->node()->input())) {
            continue;
          }
          // 创建 ListRefinement 对象
          ListRefinement refine;
          // 将 li_len 对应的长度与 ival 关联存储在 refine 中
          refine[li_len->node()->input()] = *ival;
          // 根据节点 n 的操作符类型，存储相应的布尔值细化信息
          boolean_value_refinements_[n->output()] = n->kind() == aten::eq
              ? BooleanRefinementMapping::TrueRefinements(std::move(refine))
              : BooleanRefinementMapping::FalseRefinements(std::move(refine));
        }
      } else if (n->kind() == aten::len) {
        // 如果节点 n 是长度操作符
        if (auto maybe_len = tryFindRefinement(n->input(0))) {
          // 尝试查找输入节点的细化信息，并标记状态为已更改
          changed_ = true;
          // 在节点 n 处插入一个常量，替换节点 n 的输出用途
          WithInsertPoint guard(n);
          n->output()->replaceAllUsesWith(
              graph_->insertConstant(static_cast<int64_t>(*maybe_len)));
        }
      } else if (n->kind() == prim::If) {
        // 如果节点 n 是条件分支操作
        IfView if_n(n);
        // 检查条件节点 if_n.cond() 是否存在于布尔值细化信息中
        bool has_cond_ref = boolean_value_refinements_.count(if_n.cond()) != 0;
        // 创建空的 ListRefinement 对象
        ListRefinement empty;
        // 根据条件节点的真分支和假分支，进行列表长度的细化
        auto true_block_refinements = RefineListLens(
            if_n.thenBlock(),
            has_cond_ref ? boolean_value_refinements_[if_n.cond()].true_refine()
                         : empty);
        auto false_block_refinements = RefineListLens(
            if_n.elseBlock(),
            has_cond_ref
                ? boolean_value_refinements_[if_n.cond()].false_refine()
                : empty);

        // 合并条件分支的细化信息
        joinIfRefinements(
            n,
            throwing_blocks_,
            block_refinements,
            true_block_refinements,
            false_block_refinements,
            boolean_value_refinements_);
      } else {
        // 处理常见的细化操作符
        handleCommonRefinentOperators(
            n, throwing_blocks_, boolean_value_refinements_);
      }
    }
    // 移除当前活跃的细化信息，并返回块的细化信息
    active_refinements_.pop_back();
    return block_refinements;
  };

  // 尝试查找给定值 v 的细化信息
  std::optional<int64_t> tryFindRefinement(Value* v) {
    // 遍历当前活跃的细化信息列表
    for (const auto& ref : active_refinements_) {
      // 尝试在当前细化信息中查找值 v 的细化
      auto maybe_refinement = ref->find(v);
      // 如果找到则返回找到的细化值
      if (maybe_refinement != ref->end()) {
        return maybe_refinement->second;
      }
    }
    // 如果未找到对应的细化信息则返回空值
    // 返回一个空的 std::optional 包装，表明没有值可返回
    return c10::nullopt;
  }

  // 指向图形数据结构的共享指针
  std::shared_ptr<Graph> graph_;
  // 存储发生变化的列表元素的集合
  std::unordered_set<Value*> mutated_lists_;
  // 可优化的候选列表集合
  std::unordered_set<Value*> lists_to_refine_;
  // 活跃的列表精化对象栈，每个基本块对应一个
  std::vector<ListRefinement*> active_refinements_;
  // 布尔值 Value* 到其相关精化映射的哈希表
  std::unordered_map<Value*, BooleanRefinementMapping> boolean_value_refinements_;
  // 抛出异常的基本块集合
  std::unordered_set<Block*> throwing_blocks_;
  // 表示是否发生了变化的布尔标志
  bool changed_ = false;
};

// 结构体 PeepholeOptimizeListIdiomsImpl，用于优化不会被修改的列表的情况。
// 首先使用别名数据库来收集不能进行优化的列表值集合。
struct PeepholeOptimizeListIdiomsImpl {
  // 构造函数，接受图对象和是否需要优化列表长度的标志。
  PeepholeOptimizeListIdiomsImpl(
      std::shared_ptr<Graph> graph,
      bool refine_list_len)
      : graph_(std::move(graph)),  // 初始化图对象
        aliasDb_(std::make_unique<AliasDb>(graph_)),  // 使用图对象初始化别名数据库
        refine_list_len_(refine_list_len) {}  // 初始化是否需要优化列表长度的标志

  // 运行优化过程的主函数
  bool run() {
    collectMutatedLists(graph_->block());  // 收集可能被修改的列表
    bool changed = runBlock(graph_->block());  // 运行基本块的优化
    if (refine_list_len_) {
      changed |= ListLenRefiner(graph_, mutated_lists_).run();  // 如果需要优化列表长度，则运行列表长度优化
    }
    return changed;  // 返回是否有改变
  }

 private:
  // 检查值是否为列表类型并且是否有写操作
  void checkForMutatedList(Value* v) {
    if (v->type()->castRaw<ListType>() && aliasDb_->hasWriters(v)) {
      mutated_lists_.insert(v);  // 将可能被修改的列表插入集合中
    }
  }

  // 收集块中所有可能被修改的列表
  void collectMutatedLists(Block* b) {
    for (Value* v : b->inputs()) {
      checkForMutatedList(v);  // 检查块的输入值是否为列表并可能被修改
    }
    for (Node* n : b->nodes()) {
      for (Value* v : n->outputs()) {
        checkForMutatedList(v);  // 检查节点的输出值是否为列表并可能被修改
      }
      for (Block* block : n->blocks()) {
        collectMutatedLists(block);  // 递归收集子块中可能被修改的列表
      }
    }
  }

  // 对切片操作进行优化
  bool optimizeSlice(Node* slice_node, Node* list_construct_node) {
    auto start_val = toIValue(slice_node->input(1));  // 获取切片起始位置的值
    auto end_val = toIValue(slice_node->input(2));  // 获取切片结束位置的值
    auto step_val = toIValue(slice_node->input(3));  // 获取切片步长的值

    // 如果任何参数不是常量，则无法应用此优化
    if (start_val == c10::nullopt || end_val == c10::nullopt ||
        step_val == c10::nullopt) {
      return false;
    }

    // 将值转换为整数
    int64_t start = start_val->isInt() ? start_val->to<int64_t>()
                                       : std::numeric_limits<int64_t>::max();
    int64_t end = end_val->isInt() ? end_val->to<int64_t>()
                                   : std::numeric_limits<int64_t>::max();
    int64_t step = step_val->isInt() ? step_val->to<int64_t>() : 1;

    // 获取列表构造节点的输入大小
    size_t list_size = list_construct_node->inputs().size();
    // 调整切片索引，返回切片中的值数目
    size_t num_values = slice_indices_adjust(list_size, &start, &end, step);

    // 设置插入点，并创建切片后的列表构造节点
    WithInsertPoint guard(slice_node);
    auto slice_list_construct =
        graph_->insertNode(graph_->create(prim::ListConstruct));
    slice_list_construct->output()->setType(slice_node->output()->type());
    // 根据切片参数填充切片后的列表构造节点
    for (size_t i = start, j = 0; j < num_values; ++j) {
      slice_list_construct->addInput(list_construct_node->input(i));
      i += step;
    }

    // 替换原始切片节点的所有使用
    slice_node->output()->replaceAllUsesWith(slice_list_construct->output());
    // 如果切片节点的输出是可能被修改的列表，则将切片后的列表构造节点输出添加到集合中
    if (mutated_lists_.count(slice_node->output())) {
      mutated_lists_.insert(slice_list_construct->output());
    }

    return true;  // 返回优化是否成功
  }

  // 在块中运行优化操作
  bool runBlock(Block* block) {
    bool changed = false;
    // ...
    // （此处省略了部分代码，因为注释的要求是每一行都需要注释）
    return changed;  // 返回是否有改变
  }
    // 遍历给定代码块中的所有节点
    for (Node* node : block->nodes()) {
      // 遍历当前节点的所有子块
      for (Block* b : node->blocks()) {
        // 运行子块，并根据返回值更新标志位
        changed |= runBlock(b);
      }

      // 仅优化列表操作节点
      // 如果当前节点没有输入或者第一个输入不是列表类型，则跳过当前节点
      if (node->inputs().empty() ||
          !node->input(0)->type()->castRaw<ListType>()) {
        continue;
      }

      auto first_input = node->input(0);

      // 仅优化未被修改过的列表操作节点
      // 如果第一个输入节点已被标记为被修改过的列表，则跳过当前节点
      if (mutated_lists_.count(first_input)) {
        continue;
      }

      auto list_creation_node = first_input->node();
      // 如果第一个输入节点不是列表构造节点，则跳过当前节点
      if (list_creation_node->kind() != prim::ListConstruct) {
        continue;
      }

      // 处理节点种类为 aten::len 的情况
      if (node->kind() == aten::len) {
        // 设置插入点为当前节点，并替换当前节点的输出为列表的长度常量
        WithInsertPoint guard(node);
        node->output()->replaceAllUsesWith(graph_->insertConstant(
            static_cast<int64_t>(first_input->node()->inputs().size())));
        changed = true;
      } else if (node->kind() == aten::__getitem__) {
        // 处理节点种类为 aten::__getitem__ 的情况
        // 如果索引可以转换为整数值
        if (auto index = toIValue(node->input(1))) {
          size_t list_size = list_creation_node->inputs().size();
          // 标准化索引值，并替换当前节点的输出为列表中对应索引的值
          if (auto norm_index = normalizeIndex(index->toInt(), list_size)) {
            node->output()->replaceAllUsesWith(
                list_creation_node->input(*norm_index));
            changed = true;
          }
        }
      } else if (node->kind() == prim::ListUnpack) {
        // 处理节点种类为 prim::ListUnpack 的情况
        // 如果列表创建节点的输入数量与当前节点的输出数量不一致，则跳过当前节点
        if (list_creation_node->inputs().size() != node->outputs().size()) {
          continue;
        }
        // 替换当前节点的每个输出为列表创建节点对应位置的输入值
        for (size_t i = 0; i < node->outputs().size(); ++i) {
          node->output(i)->replaceAllUsesWith(list_creation_node->input(i));
          changed = true;
        }
      } else if (node->kind() == aten::add) {
        // 处理节点种类为 aten::add 的情况
        // 如果输入节点数量不为2，则跳过当前节点
        if (node->inputs().size() != 2) {
          continue;
        }
        auto second_input = node->input(1);
        // 已经检查了第一个输入，现在检查第二个输入是否被修改过
        if (mutated_lists_.count(second_input)) {
          continue;
        }
        // 如果第二个输入不是列表构造节点，则跳过当前节点
        if (second_input->node()->kind() != prim::ListConstruct) {
          continue;
        }
        // 设置插入点为当前节点，并替换当前节点的输出为合并后的列表构造节点输出
        WithInsertPoint guard(node);
        auto list_construct =
            graph_->insertNode(graph_->create(prim::ListConstruct));
        list_construct->output()->setType(node->output()->type());
        for (Value* v : first_input->node()->inputs()) {
          list_construct->addInput(v);
        }
        for (Value* v : second_input->node()->inputs()) {
          list_construct->addInput(v);
        }
        node->output()->replaceAllUsesWith(list_construct->output());
        // 如果当前节点的输出被标记为修改过的列表，则更新标记
        if (mutated_lists_.count(node->output())) {
          mutated_lists_.insert(list_construct->output());
        }
        changed = true;
      } else if (node->kind() == aten::slice) {
        // 处理节点种类为 aten::slice 的情况，调用 optimizeSlice 函数进行优化
        changed |= optimizeSlice(node, first_input->node());
      }
    }
    // 返回是否有任何节点被修改的标志位
    return changed;
  }

  // 用于存储被修改过的列表的集合
  std::unordered_set<Value*> mutated_lists_;
  // 共享的计算图对象指针
  std::shared_ptr<Graph> graph_;
  // 唯一的别名数据库对象指针
  std::unique_ptr<AliasDb> aliasDb_;
  // 是否需要优化列表长度的标志位
  bool refine_list_len_;
};

// 定义一个名为 PeepholeOptimizeListIdioms 的函数，它接受一个名为 graph 的 std::shared_ptr 类型的参数和一个名为 refine_list_len 的 bool 类型参数，并返回一个 bool 类型的结果
bool PeepholeOptimizeListIdioms(
    const std::shared_ptr<Graph>& graph,
    bool refine_list_len) {
  // 创建 PeepholeOptimizeListIdiomsImpl 对象 opt，传入 graph 和 refine_list_len 参数
  PeepholeOptimizeListIdiomsImpl opt(graph, refine_list_len);
  // 调用 opt 对象的 run 方法并返回其结果
  return opt.run();
}

// 结束 jit 命名空间的定义
} // namespace jit

// 结束 torch 命名空间的定义
} // namespace torch
```