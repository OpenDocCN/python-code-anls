# `.\pytorch\torch\csrc\jit\frontend\canonicalize_modified_loop.cpp`

```py
// 包含必要的头文件和命名空间声明
#include <functional>
#include <memory>
#include <string>

#include <torch/csrc/Export.h>
#include <torch/csrc/jit/frontend/canonicalize_modified_loop.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/ir_views.h>

namespace torch::jit {

// 对一个节点中的修改后的循环进行规范化处理
static void canonicalizeModifiedLoop(Node* n) {
  // 使用 LoopView 对象来查看循环类型
  LoopView loop(n);
  // 如果循环类型不是修改后的循环，则直接返回
  if (loop.loopType() != LoopView::ModifiedLoop) {
    return;
  }

  auto g = n->owningGraph();
  // 在节点的插入点进行操作
  WithInsertPoint node_insert(n);
  // 插入常量 0 和 1 到图中
  auto zero = g->insertConstant(0);
  auto one = g->insertConstant(1);
  // 获取最大迭代次数和生成循环条件
  auto max_trip_count = loop.maxTripCount();
  auto condition = g->insert(aten::gt, {max_trip_count, zero});
  // 将最大迭代次数替换为 int64_t 类型的最大值
  loop.replaceMaxTripCount(
      g->insertConstant(std::numeric_limits<int64_t>::max()));

  // 获取循环输入条件的 IValue 表示
  auto inp_condition = toIValue(loop.inputCond());
  // 如果输入条件不存在或者为 false，则将 condition 与输入条件进行与操作
  if (inp_condition == c10::nullopt || inp_condition->toBool() == false) {
    condition = g->insert(aten::__and__, {condition, loop.inputCond()});
  }
  // 替换循环的输入条件
  loop.replaceInputCondition(condition);
  // 为节点添加一个输出，类型为 IntType
  n->addOutput()->setType(IntType::get());
  // 在循环体的插入点进行操作
  WithInsertPoint loop_insert(loop.bodyBlock());
  // 添加常量 0 到循环体的输入
  n->addInput(zero);
  // 在循环体中添加一个新的迭代器变量
  auto new_iter = loop.bodyBlock()->addInput()->setType(IntType::get());
  // 清除 jitter 的唯一名称，其替换不再具有名称
  loop.currentTripCount()->setDebugName("")->replaceAllUsesWith(new_iter);
  // 生成迭代器增加操作
  auto inc_iter = g->insert(aten::add, {new_iter, one});
  // 注册循环体的输出为迭代器增加结果
  loop.bodyBlock()->registerOutput(inc_iter);
  // 生成小于最大迭代次数的条件
  auto less_than_max_trip = g->insert(aten::lt, {inc_iter, max_trip_count});
  // 获取循环的继续条件
  auto loop_continue = loop.nextCond();
  // 生成新的循环条件
  auto new_condition =
      g->insert(aten::__and__, {less_than_max_trip, loop_continue});
  // 清除循环体的第一个输出
  loop.bodyBlock()->eraseOutput(0);
  // 在循环体中插入新生成的条件作为第一个输出
  loop.bodyBlock()->insertOutput(0, new_condition);
}

// 递归遍历一个基本块及其子块，对其中的修改后的循环进行规范化处理
static void canonicalizeModifiedLoops(Block* block) {
  for (Node* n : block->nodes()) {
    for (Block* b : n->blocks()) {
      canonicalizeModifiedLoops(b);
    }
    // 如果节点的类型是 prim::Loop，则对其进行规范化处理
    if (n->kind() == prim::Loop) {
      canonicalizeModifiedLoop(n);
    }
  }
}

// 对图中的所有修改后的循环进行规范化处理，使其可以表示为 Python 的 for 或 while 循环
TORCH_API void CanonicalizeModifiedLoops(std::shared_ptr<Graph>& graph) {
  canonicalizeModifiedLoops(graph->block());
}

} // namespace torch::jit
```