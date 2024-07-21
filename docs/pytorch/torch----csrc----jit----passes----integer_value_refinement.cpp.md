# `.\pytorch\torch\csrc\jit\passes\integer_value_refinement.cpp`

```py
# 包含 ATen 核心的 JIT 类型头文件
#include <ATen/core/jit_type.h>
# 包含 Torch JIT IR 的头文件
#include <torch/csrc/jit/ir/ir.h>
# 包含 Torch JIT 日志的头文件
#include <torch/csrc/jit/jit_log.h>
# 包含 Torch JIT 整数值细化的头文件
#include <torch/csrc/jit/passes/integer_value_refinement.h>
# 包含 Torch JIT 值细化工具的头文件
#include <torch/csrc/jit/passes/value_refinement_utils.h>

# 包含 C++ 标准库中的实用工具
#include <utility>

# 定义 torch::jit 命名空间
namespace torch {
namespace jit {

# 使用 IntegerRefinement 别名表示 Value 指针到整数值的无序映射
using IntegerRefinement = std::unordered_map<Value*, int64_t>;

# 结构体 IntegerValueRefiner 实现
# 用于处理整数值细化的算法
struct IntegerValueRefiner {
  # 构造函数，接受一个共享指针指向图对象
  IntegerValueRefiner(std::shared_ptr<Graph> graph)
      : graph_(std::move(graph)) {}

  # 运行整数值细化的方法
  bool run() {
    # 如果图的基本块中不包含整数比较操作，直接返回 false
    if (!blockHasIntComparisons(graph_->block())) {
      return false;
    }
    # 创建整数值细化的映射对象
    IntegerRefinement refinements;
    # 对图的基本块进行整数值细化
    RefineIntegerValues(graph_->block(), std::move(refinements));
    # 返回是否有改变的标志
    return changed_;
  }

  # 检查基本块中是否包含整数比较操作
  bool blockHasIntComparisons(Block* b) {
    for (Node* n : b->nodes()) {
      # 检查节点是否匹配整数相等比较操作
      if (n->matches("aten::eq(int a, int b) -> bool") ||
          n->matches("aten::ne(int a, int b) -> bool")) {
        for (size_t const_index : {0, 1}) {
          auto non_const_index = 1 - const_index;
          # 如果一个输入是常量且另一个输入被多次使用，则返回 true
          if (n->inputs().at(const_index)->node()->kind() == prim::Constant &&
              n->inputs().at(non_const_index)->uses().size() > 1) {
            return true;
          }
        }
      }
      # 递归检查节点的所有子块
      for (Block* block : n->blocks()) {
        if (blockHasIntComparisons(block)) {
          return true;
        }
      }
    }
    # 如果没有找到整数比较操作，返回 false
    return false;
  }

  # 如果节点的输出带有细化值，则移除条件节点的输出
  void removeIfNodeOutputsWithRefinements(
      Node* if_node,
      IntegerRefinement& true_block_refinements,
      IntegerRefinement& false_block_refinements) {
    # 查找可以替换两个块输出为相同值的情况，以打开进一步的优化机会
    # 这里处理两个块输出被细化为相同常量值的情况
    # 这种情况下，可以安全地用一个块的输出替换另一个块的输出
    # 这对于符号形状分析非常重要
    // 对于 block_index 取值为 {0, 1} 中的每一个值，执行以下操作
    for (size_t block_index : {0, 1}) {
      // 获取 if_node 的第 block_index 个 block
      Block* if_block = if_node->blocks().at(block_index);
      // 获取 if_node 的另一个 block
      Block* other_if_block = if_node->blocks().at(1 - block_index);
      // 遍历 if_node 的输出值列表
      for (size_t i = 0; i < if_node->outputs().size(); ++i) {
        // 获取当前 if_block 的第 i 个输出值
        Value* block_output = if_block->outputs().at(i);
        // 如果该输出值不是 IntType 类型，则跳过当前循环
        if (!block_output->type()->cast<IntType>()) {
          continue;
        }
        // 确保 block_output 在两个 block 中均处于作用域内
        if (!if_node->isDominatedBy(block_output->node())) {
          continue;
        }
        // 获取 other_if_block 的第 i 个输出值
        auto other_output = other_if_block->outputs().at(i);
        // 如果 other_output 不是 IntType 类型，则跳过当前循环
        auto other_const_value = other_output->type()->cast<IntType>()
            ? constant_as<int64_t>(other_output)
            : c10::nullopt;
        // 如果 other_const_value 不存在，或者 block_output 是常量，则跳过当前循环
        if (!other_const_value ||
            block_output->node()->kind() == prim::Constant) {
          continue;
        }
        // 获取当前 block_index 对应的 block_refinements
        const auto& other_block_refinements =
            block_index == 0 ? false_block_refinements : true_block_refinements;
        // 如果 other_block_refinements 中不包含 block_output，则跳过当前循环
        if (!other_block_refinements.count(block_output)) {
          continue;
        }
        // 如果 other_block_refinements 中 block_output 对应的值等于 other_const_value
        if (other_block_refinements.at(block_output) == *other_const_value) {
          // 将 if_node 的第 i 个输出值替换为 block_output
          if_node->outputs().at(i)->replaceAllUsesWith(block_output);
          // 设置 changed_ 为 true
          changed_ = true;
        }
      }
    }
  }

  // 迭代地查找 block `b` 中可以进行细化的 refinements 或者可以细化的 Value 使用，
  // `block_refinements` 是从该 block 开始的 refinements（以及该 block 支配的所有 block）。
  IntegerRefinement RefineIntegerValues(
      Block* b,
      IntegerRefinement block_refinements) {
    // 将 block_refinements 添加到 active_refinements_ 中
    active_refinements_.push_back(&block_refinements);
    // 遍历基本块中的所有节点
    for (Node* n : b->nodes()) {
      // 如果节点匹配 "aten::eq(int a, int b) -> bool" 或 "aten::ne(int a, int b) -> bool"
      if (n->matches("aten::eq(int a, int b) -> bool") ||
          n->matches("aten::ne(int a, int b) -> bool")) {
        // 对于常数索引为 0 和 1 的情况
        for (size_t const_index : {0, 1}) {
          // 如果输入是常数，获取其值
          if (auto ival = constant_as<int64_t>(n->inputs().at(const_index))) {
            // 创建整数精化对象
            IntegerRefinement refine;
            // 将第二个输入与其值关联起来
            refine[n->inputs().at(1 - const_index)] = *ival;
            // 根据节点类型将结果存储在 info_ 中
            info_[n->output()] = n->kind() == aten::eq
                ? BooleanRefinementMapping::TrueRefinements(std::move(refine))
                : BooleanRefinementMapping::FalseRefinements(std::move(refine));
          }
        }
      }
      // 遍历节点的所有输入
      for (size_t input = 0; input < n->inputs().size(); ++input) {
        Value* input_v = n->inputs().at(input);
        // 如果输入不是整数类型，跳过
        if (!input_v->type()->cast<IntType>()) {
          continue;
        }
        // 尝试找到输入值的精化
        if (auto refine = tryFindRefinement(input_v)) {
          // 在当前节点插入常数值，并替换输入值
          WithInsertPoint guard(n);
          auto refine_constant =
              graph_->insertConstant(static_cast<int64_t>(*refine));
          n->replaceInputWith(input_v, refine_constant);
          // 标记修改状态为真
          changed_ = true;
        }
      }

      // 如果节点是 prim::If 类型
      if (n->kind() == prim::If) {
        // 创建 IfView 对象
        IfView if_n(n);
        // 检查条件是否已经有精化信息
        bool has_cond_ref = info_.count(if_n.cond()) != 0;
        // 创建空的整数精化对象
        IntegerRefinement empty;
        // 获取条件为真和条件为假时的块中的整数精化信息
        auto true_block_refinements = RefineIntegerValues(
            if_n.thenBlock(),
            has_cond_ref ? info_[if_n.cond()].true_refine() : empty);
        auto false_block_refinements = RefineIntegerValues(
            if_n.elseBlock(),
            has_cond_ref ? info_[if_n.cond()].false_refine() : empty);

        // 移除带有精化信息的 If 节点输出
        removeIfNodeOutputsWithRefinements(
            n, true_block_refinements, false_block_refinements);

        // 合并 If 节点的精化信息
        joinIfRefinements(
            n,
            throwing_blocks_,
            block_refinements,
            true_block_refinements,
            false_block_refinements,
            info_);
      } else {
        // 处理常见的精化操作符
        handleCommonRefinentOperators(n, throwing_blocks_, info_);
      }
    }

    // 遍历基本块中的所有输出值
    for (size_t i = 0; i < b->outputs().size(); ++i) {
      Value* output_v = b->outputs().at(i);
      // 如果输出值不是整数类型，跳过
      if (!output_v->type()->cast<IntType>()) {
        continue;
      }
      // 尝试找到输出值的精化
      if (auto refine = tryFindRefinement(output_v)) {
        // 在当前基本块中插入常数值，并替换输出值
        WithInsertPoint guard(b);
        auto refine_constant =
            graph_->insertConstant(static_cast<int64_t>(*refine));
        b->replaceOutput(i, refine_constant);
        // 标记修改状态为真
        changed_ = true;
      }
    }

    // 弹出当前活跃的精化信息
    active_refinements_.pop_back();
    // 返回基本块的精化信息
    return block_refinements;
  };

  // 尝试找到给定值的精化信息，返回一个可选的 int64_t 类型
  std::optional<int64_t> tryFindRefinement(Value* v) {
    // 遍历活动精化（refinement）的列表 active_refinements_
    for (const auto& ref : active_refinements_) {
      // 尝试在当前精化（refinement）中查找值 v
      auto maybe_refinement = ref->find(v);
      // 如果找到了值 v 对应的精化（refinement）
      if (maybe_refinement != ref->end()) {
        // 返回找到的精化（refinement）的值
        return maybe_refinement->second;
      }
    }
    // 如果未找到任何精化（refinement）对应于值 v，则返回空的 optional 对象
    return c10::nullopt;
  }

  // 一个指向 Graph 对象的共享指针
  std::shared_ptr<Graph> graph_;
  // 一组活动精化（refinement）对象的堆栈，每个块对应一个
  std::vector<IntegerRefinement*> active_refinements_;
  // 一个从 Boolean Value 指针到相关精化（refinement）的映射
  std::unordered_map<Value*, BooleanRefinementMapping> info_;
  // 一个包含抛出异常的块指针的集合
  std::unordered_set<Block*> throwing_blocks_;
  // 标志，表示 Graph 对象是否发生了变化，默认为 false
  bool changed_ = false;
};

bool RefineIntegerValues(const std::shared_ptr<Graph>& graph) {
    // 创建 IntegerValueRefiner 对象并使用传入的图对象运行 refine() 方法
    return IntegerValueRefiner(graph).run();
}

} // namespace jit
} // namespace torch
```