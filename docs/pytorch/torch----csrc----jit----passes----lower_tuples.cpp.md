# `.\pytorch\torch\csrc\jit\passes\lower_tuples.cpp`

```py
// 引入 Torch 库中的头文件，用于 JIT 编译时的元组降级处理
#include <torch/csrc/jit/passes/lower_tuples.h>

// 引入 ATen 库中的头文件，包括功能函数和异常处理
#include <ATen/core/functional.h>
#include <c10/util/Exception.h>
#include <c10/util/irange.h>

// 引入 Torch JIT 中的常量定义和日志记录功能
#include <torch/csrc/jit/ir/constants.h>
#include <torch/csrc/jit/jit_log.h>

// 引入 Torch JIT 中的死代码消除功能
#include <torch/csrc/jit/passes/dead_code_elimination.h>

// 引入 C++ 标准库中的实用工具
#include <utility>

// 定义 torch::jit 命名空间下的内部匿名命名空间
namespace torch {
namespace jit {

namespace {

// 支持期望输入/输出为元组的操作符集合
// 用于断言仅在可以展平元组时才进行修改
std::unordered_set<Symbol> supported_ops = {
    prim::If,
    prim::Loop,
    prim::Uninitialized,
    prim::TupleUnpack,
    prim::TupleConstruct,
    prim::TupleIndex,
    prim::TupleSlice,
    prim::Param,
    prim::Return,
    prim::PythonOp,
    aten::format,
    prim::Uninitialized,
    aten::__getitem__};

// 在循环节点参数中展平元组输入并插入元组构造节点
static void flattenTupleInLoopParams(Node* n, size_t index) {
  auto input = n->inputs().at(index);
  TupleTypePtr tt = input->type()->cast<TupleType>();
  TORCH_INTERNAL_ASSERT(tt);

  Block* block = n->blocks().at(0);
  Node* block_node = n;

  std::vector<Value*> new_node_inputs = {};
  auto new_construct_node =
      block->prependNode(block->owningGraph()->create(prim::TupleConstruct));
  for (size_t j = 0; j < tt->elements().size(); ++j) {
    auto new_block_in = block->insertInput(index + j);
    new_construct_node->addInput(new_block_in);
    block_node->insertInput(index + j + 1, input->node()->inputs().at(j));
  }
  new_construct_node->output()->setType(block->inputs().at(index - 1)->type());
  new_construct_node->copyMetadata(n);
  block->inputs().at(index - 1)->replaceAllUsesWith(
      new_construct_node->output());
  block->eraseInput(index - 1);
  block_node->removeInput(index);
}

// 展平块节点返回的元组输出，并在存在外部块的情况下在块节点后附加元组构造节点
static void flattenTupleInBlockReturn(Node* n, size_t index) {
  auto input = n->inputs().at(index);
  Block* block = n->owningBlock();
  Node* block_node = block->owningNode();
  Node* new_construct_node = nullptr;
  TupleTypePtr tt = input->type()->cast<TupleType>();
  TORCH_INTERNAL_ASSERT(tt);

  // 1- 添加展平后的元组到块输出
  for (size_t j = 0; j < tt->elements().size(); ++j) {
    block->insertOutput(index + j + 1, input->node()->inputs().at(j));
  }
  block->eraseOutput(index);

  // 如果块节点为空，则返回
  if (block_node == nullptr)
    return;

  // 2- 对外部块中使用块节点的地方，展平块节点的输出并插入元组构造节点进行替换
  // 循环块有额外的元素（迭代计数器）
  if (block_node->kind() == prim::Loop)
    index = index - 1;
  auto tuple_output = block_node->outputs().at(index);

  // 当节点有多个块时，不要重复在第二个块上展平输出
  if (!(tuple_output->type()->cast<TupleType>()))
    // 直接返回，不执行后续代码
    return;

  // 创建一个新的 TupleConstruct 节点，使用 block 所属图创建
  new_construct_node = block->owningGraph()->create(prim::TupleConstruct);
  // 将新节点插入到 block_node 之后
  new_construct_node->insertAfter(block_node);
  // 遍历 tt 的元素
  for (size_t j = 0; j < tt->elements().size(); ++j) {
    // 在 block_node 后插入一个新的输出
    auto new_block_out = block_node->insertOutput(index + j + 1);
    // 将新输出作为输入添加到 new_construct_node
    new_construct_node->addInput(new_block_out);
  }
  // 设置新 TupleConstruct 节点的输出类型与 tuple_output 相同
  new_construct_node->output()->setType(tuple_output->type());
  // 复制 block_node 的元数据到 new_construct_node
  new_construct_node->copyMetadata(block_node);
  // 用 new_construct_node 的输出替换 tuple_output 的所有使用
  tuple_output->replaceAllUsesWith(new_construct_node->output());
  // 删除 block_node 的指定输出
  block_node->eraseOutput(index);
// 匿名命名空间，用于限定符号的作用域，避免全局命名冲突
} // anonymous namespace

// 声明一个静态函数，将块中所有的元组降级处理
static void LowerAllTuples(Block* block);

// 移除常量节点中的元组常量
static void RemoveTupleConstants(Node* n) {
  // 如果节点不是常量或者输出不是元组类型，则直接返回
  if (!(n->kind() == prim::Constant &&
        n->output()->type()->cast<TupleType>())) {
    return;
  }

  auto g = n->owningGraph();
  // 将节点的输出值转换为元组类型的IValue
  auto tuple = toIValue(n->output()).value().toTuple();
  // 获取元组中的所有元素
  const auto& tuple_elements = tuple->elements();
  // 在当前节点插入新的插入点
  WithInsertPoint insert(n);
  std::vector<Value*> elements;
  // 遍历元组的每个元素，并将其作为常量插入到图中
  for (const auto& elem : tuple_elements) {
    auto constant = insertConstant(*n->owningGraph(), elem);
    elements.push_back(constant);
  }
  // 获取当前节点输出的元组类型
  auto tuple_type = n->output()->type()->expect<TupleType>();
  // 创建一个新的元组构造节点，并插入到图中
  auto tuple_construct = g->insertNode(n->owningGraph()->createTuple(
      elements, tuple_type->schema() ? std::move(tuple_type) : nullptr));
  tuple_construct->copyMetadata(n);

  // 在递归处理元素之前先插入新的元组节点，以便其元素有用
  for (Value* elem : elements) {
    RemoveTupleConstants(elem->node());
  }

  // 将当前节点的所有使用替换为新创建的元组构造节点
  n->replaceAllUsesWith(tuple_construct);
}
// 对输入节点进行扁平化处理，即将元组类型的输入展开成多个单独的输入
static void flattenInputs(Node* n, Node* insert_point) {
  // flatten the input list  op(a, tup, b) --> op(a, t0, t1, b)
  for (size_t i = 0; i < n->inputs().size();) {
    auto input = n->inputs()[i];
    // 检查当前输入是否为元组类型
    if (TupleTypePtr tt = input->type()->cast<TupleType>()) {
      TORCH_CHECK(
          (input->node()->kind() == prim::TupleConstruct),
          "tuple use not matched to tuple construct. Instead found: ",
          n->kind().toQualString());
      // 检查当前操作是否支持元组扁平化处理
      if (supported_ops.count(n->kind()) > 0) {
        if (n->kind() == prim::Loop) {
          // 对于循环操作，支持接受元组作为参数的块
          flattenTupleInLoopParams(n, i);
        } else if (n->kind() == prim::Return) {
          // 在返回操作中扁平化元组
          flattenTupleInBlockReturn(n, i);
        } else {
          // 将元组展开成单独的输入，并从原输入列表中移除元组
          for (size_t j = 0; j < tt->elements().size(); ++j) {
            n->insertInput(i + 1 + j, input->node()->inputs().at(j));
          }
          n->removeInput(i);
        }
        // 注意：不更新 i 的值，因为可能需要递归扫描新的扁平化输入
      } else {
        // 如果操作不支持元组扁平化，发出警告并继续处理下一个输入
        TORCH_WARN(
            "tuple appears in op inputs, but this op does not forward tuples, ",
            "unsupported kind: ",
            n->kind().toQualString());
        ++i;
      }
    } else {
      // 如果当前输入不是元组类型，则直接处理下一个输入
      ++i;
    }
  }
}

// 对输出节点进行扁平化处理
static void flattenOutputs(Node* n, Node* insert_point) {
  // flatten the outputs list
  auto& graph = *n->owningGraph();
  for (size_t i = 0; i < n->outputs().size();) {
    Value* output = n->outputs()[i];
    // 检查当前输出是否有使用者
    if (!output->hasUses()) {
      // 如果输出没有使用者，直接处理下一个输出
      ++i;
      continue;
    }

    // (a, b, tup, c) -> (a, b, t0, t1, c)
    // and:
    //    tup = (t0, t1)
    // is placed at the current insertion point
    // 检查当前输出是否为元组类型
    if (TupleTypePtr tt = output->type()->cast<TupleType>()) {
      // 检查当前操作是否支持元组扁平化处理
      if (supported_ops.count(n->kind()) > 0) {
        // 将元组展开成多个单独的输出
        for (const auto j : c10::irange(tt->elements().size())) {
          n->insertOutput(i + 1 + j)->setType(tt->elements()[j]);
        }
        // 创建新的元组节点，并在当前插入点之前插入
        auto new_tup =
            graph.createTuple(n->outputs().slice(i + 1, tt->elements().size()));
        new_tup->copyMetadata(n);
        new_tup->insertBefore(insert_point);
        insert_point = new_tup;
        // 替换所有使用当前输出的节点为新创建的元组节点
        output->replaceAllUsesWith(new_tup->output());
        n->eraseOutput(i);
        // 注意：不更新 i 的值以处理嵌套元组
      } else {
        // 如果操作不支持元组扁平化，发出警告并继续处理下一个输出
        TORCH_WARN(
            "tuple appears in the op outputs, but this op does not forward tuples, ",
            "unsupported kind: ",
            n->kind().toQualString());
        ++i;
      }
    } else {
      // 如果当前输出不是元组类型，则直接处理下一个输出
      ++i;
    }
  }
}

// 访问节点的函数，用于处理元组构造操作符
static void VisitNode(Node* n, Node* insert_point) {
  // tuple construction operators will become dead when the unpacks are replaced
  // 当解压操作替换时，元组构造操作符将被移除
  if (n->kind() == prim::TupleConstruct) {
    return;
  }


    # 结束当前函数，直接返回，不执行后续代码
    return;
  }



  // note: changing the second argument to false changes this pass from a
  // complete lowering pass to one that removes tuples when possible. When
  // tuples are first-class in the interpreter, we should still run this pass to
  // remove extraneous uses


  # 注意：将第二个参数更改为 false 将此处理从完全降级处理变为尽可能移除元组的处理。
  # 当元组在解释器中是一等公民时，仍应运行此处理以删除多余的使用情况。



  if (n->kind() == prim::TupleUnpack || n->kind() == prim::TupleIndex ||
      n->kind() == prim::TupleSlice) {
    removeTupleNodes(n, /*must_remove_tuples*/ true);
    return;
  }


  # 如果节点 n 的类型为元组解包、元组索引或元组切片
  if (n->kind() == prim::TupleUnpack || n->kind() == prim::TupleIndex ||
      n->kind() == prim::TupleSlice) {
    # 调用函数 removeTupleNodes，强制移除元组节点
    removeTupleNodes(n, /*must_remove_tuples*/ true);
    # 直接返回，不执行后续代码
    return;
  }



  flattenInputs(n, insert_point);
  for (auto b : n->blocks()) {
    LowerAllTuples(b);
  }
  flattenOutputs(n, insert_point);


  # 对节点 n 进行输入展平化处理，插入点为 insert_point
  flattenInputs(n, insert_point);
  # 对节点 n 的每个子块执行 LowerAllTuples 函数
  for (auto b : n->blocks()) {
    LowerAllTuples(b);
  }
  # 对节点 n 进行输出展平化处理，插入点为 insert_point
  flattenOutputs(n, insert_point);
}

// 下面是一个静态函数，用于将一个块中的所有元组转换为小写。元组在块的参数列表中的行为与普通指令的输出完全相同，因此我们可以通过简单访问节点来处理它。
static void LowerAllTuples(Block* block) {
  // 访问块的参数节点，它表示参数作为输出，我们可以通过访问节点来处理它
  VisitNode(block->param_node(), *block->nodes().begin());
  // 遍历块中的每个节点
  for (auto it = block->nodes().begin(), end = block->nodes().end();
       it != end;) {
    auto n = *it++;
    // 移除节点中的元组常量
    RemoveTupleConstants(n);
    // 继续访问节点，并移动到下一个节点
    VisitNode(n, *it);
  }
  // 访问块的返回节点，由于返回节点没有输出，因此可以将 insert_point 设置为 nullptr
  VisitNode(block->return_node(), nullptr);
}

// 确保给定值中不存在元组类型，否则抛出错误
static void EnsureNoTuples(ArrayRef<Value*> values) {
  for (Value* v : values) {
    TORCH_CHECK(
        v->type()->kind() != TypeKind::TupleType, "Couldn't lower all tuples.");
  }
}

// 确保一个块中不存在任何元组类型
static void EnsureNoTuples(Block* block) {
  // 遍历块中的每个节点
  for (Node* n : block->nodes()) {
    // 递归地确保节点中不存在元组
    for (Block* b : n->blocks()) {
      EnsureNoTuples(b);
    }
    // 确保节点的输出中不存在元组类型
    EnsureNoTuples(n->outputs());
  }
}

// 将图中的所有元组转换为小写
void LowerAllTuples(const std::shared_ptr<Graph>& graph) {
  // 将图的块传递给 LowerAllTuples 函数处理
  LowerAllTuples(graph->block());
  // 打印转换后的图
  GRAPH_DUMP("After LowerAllTuples: ", graph);
  // 消除死代码
  EliminateDeadCode(graph->block());
  // 确保图的块中不存在任何元组类型
  EnsureNoTuples(graph->block());
}

// 将一个块中的简单元组转换为小写
void LowerSimpleTuples(Block* block) {
  // 遍历块中的每个节点
  for (auto n : block->nodes()) {
    // 移除节点中的元组节点，但不强制移除元组
    removeTupleNodes(n, /*must_remove_tuples*/ false);
    // 递归处理节点中的每个块
    for (auto b : n->blocks()) {
      LowerSimpleTuples(b);
    }
  }
}

// 将图中的简单元组转换为小写
void LowerSimpleTuples(const std::shared_ptr<Graph>& graph) {
  // 将图的块传递给 LowerSimpleTuples 函数处理
  LowerSimpleTuples(graph->block());
  // 打印转换后的图
  GRAPH_DUMP("After LowerSimpleTuples: ", graph);
  // 消除死代码
  EliminateDeadCode(graph);
}
} // namespace jit
} // namespace torch
```