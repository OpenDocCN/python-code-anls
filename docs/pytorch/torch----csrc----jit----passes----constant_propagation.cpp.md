# `.\pytorch\torch\csrc\jit\passes\constant_propagation.cpp`

```
#include <torch/csrc/jit/passes/constant_propagation.h>

#include <ATen/core/functional.h>
#include <ATen/core/ivalue.h>
#include <c10/util/Exception.h>
#include <c10/util/irange.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/jit/ir/alias_analysis.h>
#include <torch/csrc/jit/ir/constants.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/node_hashing.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>
#include <torch/csrc/jit/runtime/operator.h>
#include <torch/csrc/jit/runtime/vararg_functions.h>

#include <utility>

namespace torch {
namespace jit {

// 根据节点的输入是否为常量，运行节点的操作，进行常量传播
std::optional<std::vector<IValue>> runNodeIfInputsAreConstant(
    const Node* n,
    bool ignore_custom_classes,
    AliasDb* db) {
  Stack stack; // 创建空的操作栈
  for (auto input : n->inputs()) { // 遍历节点的所有输入
    if (auto ival = toIValue(input)) { // 尝试将输入转换为 IValue 类型
      stack.push_back(*ival); // 将转换成功的输入值压入栈中
    } else {
      return c10::nullopt; // 如果有输入无法转换为 IValue，则返回空
    }
  }

  switch (n->kind()) { // 根据节点的操作类型进行不同的处理
    case prim::ListUnpack: { // 列表解包操作
      if (stack.back().toList().size() != n->outputs().size()) {
        return c10::nullopt; // 如果解包后的列表大小与输出数量不匹配，则返回空
      }
      listUnpack(stack, n->outputs().size()); // 执行列表解包操作
    } break;
    case prim::TupleConstruct: { // 元组构造操作
      auto tt = n->output()->type()->expect<TupleType>();
      if (tt->name()) {
        namedTupleConstruct(stack, std::move(tt), n->inputs().size());
      } else {
        tupleConstruct(stack, n->inputs().size());
      }
    } break;
    case prim::ListConstruct: { // 列表构造操作
      listConstruct(
          stack,
          n->output()->type()->expectRef<ListType>(),
          n->inputs().size());
    } break;
    case prim::DictConstruct: { // 字典构造操作
      dictConstruct(
          stack,
          n->output()->type()->expectRef<DictType>(),
          n->inputs().size());
    } break;
    case prim::CreateObject: { // 创建对象操作
      createObject(
          stack,
          n->output()->type()->expect<ClassType>(),
          /*use_weak_ref*/ true);
    } break;
    case prim::GetAttr: { // 获取属性操作
      auto attr = pop(stack).toObject()->getAttr(n->s(attr::name));
      push(stack, attr);
    } break;
    case prim::isinstance: { // 类型检查操作
      isinstance(stack, n->tys(attr::types));
    } break;
    default: { // 默认操作，执行节点的运算操作
      const auto maybe_schema = n->maybeSchema();
      if (maybe_schema && maybe_schema->is_vararg()) {
        // vararg schemas require the number of inputs at the top of the stack
        // but this is broken in other places in constant prop, so disable it
        // for now
        return c10::nullopt; // 如果操作类型不在预定义的范围内或是可变参数，返回空
      }

      try {
        auto op = n->getOperation(); // 获取节点的运算操作
        op(stack); // 执行节点的运算操作
      } catch (...) {
        return c10::nullopt; // 捕获异常，返回空
      }
    } break;
  }

  for (IValue& v : stack) { // 遍历操作栈中的所有值
    if (v.isTensor()) { // 如果值是 Tensor 类型
      const at::Tensor& t = v.toTensor();
      if (t.defined() && t.requires_grad()) {
        // requires grad tensors cannot be constants
        return c10::nullopt; // 如果 Tensor 需要梯度，则返回空
      }
    }
    // Weak form of const propagation
    // 如果忽略自定义类别
    if (ignore_custom_classes) {
      // 如果值是自定义类别，则返回空值
      if (v.isCustomClass()) {
        return c10::nullopt;
      }
    }
    // 查看 [Constant Object Weak CompilationUnit Reference] 注释
    if (v.isCustomClass()) {
      // 如果值是自定义类别并且是弱编译单元引用，则继续下一次循环
      if (v.toObject()->is_weak_compilation_ref()) {
        continue;
      }
      // 如果数据库不存在，则继续下一次循环
      if (!db) {
        continue;
      }
      // 允许修改非常量对象指针，这里使用了 NOLINTNEXTLINE 以忽略特定的Lint检查
      Node* n_non_const = const_cast<Node*>(n);
      // 如果数据库可能包含别名，则继续下一次循环
      if (db->mayContainAlias(
              n_non_const->inputs(), {n_non_const->outputs()})) {
        continue;
      }
      // 获取对象并将其标记为弱编译单元引用
      auto obj = v.toObject();
      obj->unsafe_make_weak_compilation_ref();
    }
    // 如果值是对象并且不是弱编译单元引用，则返回空值
    if (v.isObject()) {
      if (!v.toObject()->is_weak_compilation_ref()) {
        return c10::nullopt;
      }
    }
  }
  // 返回栈的当前状态
  return stack;
  // 匿名命名空间，定义跳过的符号集合，用于常量传播器
  std::unordered_set<Symbol> skip_list = {
      prim::If,                    // 跳过 If 符号
      prim::Loop,                  // 跳过 Loop 符号
      prim::Closure,               // 跳过 Closure 符号
      prim::Constant,              // 跳过 Constant 符号
      prim::AutogradZero,          // 跳过 AutogradZero 符号
      prim::Uninitialized,         // 跳过 Uninitialized 符号
      prim::Guard,                 // 跳过 Guard 符号
      prim::profile,               // 跳过 profile 符号
      prim::profile_ivalue,        // 跳过 profile_ivalue 符号
      prim::unchecked_unwrap_optional, // 跳过 unchecked_unwrap_optional 符号，TODO: 后续移除
      prim::awaitable,             // 跳过 awaitable 符号
      aten::dequantize             // 跳过 dequantize 符号
      // TODO (zach): 在某些情况下跳过张量工厂，当常量张量大但创建成本低时
  };

  // 常量传播器结构体，用于执行常量传播并检查图中可能被改变的输入或输出
  struct ConstantPropagator {
    // 使用别名数据库运行常量传播，检查图中输入或输出是否可能被改变
    static ConstantPropagator WithAliasDb(
        std::shared_ptr<Graph> graph,
        bool ignore_custom_classes) {
      return ConstantPropagator(std::move(graph), true, ignore_custom_classes);
    }

    // 仅在明确没有别名输入或输出的操作上运行常量传播，无需计算别名信息
    static ConstantPropagator NoAliasDb(std::shared_ptr<Graph> graph) {
      return ConstantPropagator(std::move(graph), false, false);
    }

    // 运行常量传播并返回是否进行了更改
    bool run() {
      ConstantPropagation(graph_->block());
      return made_change_;
    }

   private:
    ConstantPropagator(
        std::shared_ptr<Graph> graph,
        bool aliasing_types,
        bool ignore_custom_classes)
        : graph_(std::move(graph)),
          aliasing_types_(aliasing_types),
          ignore_custom_classes_(ignore_custom_classes) {}

    // 传播节点的常量值
    void propagateNode(Node* n) {
      std::vector<IValue> outputs;
      // 如果输入为常量，则尝试运行节点并获取输出
      if (auto outputs_opt =
              runNodeIfInputsAreConstant(n, ignore_custom_classes_)) {
        outputs = std::move(outputs_opt.value());
      } else {
        // 操作无法运行，无法继续常量传播
        return;
      }
      auto graph = n->owningGraph();
      WithInsertPoint guard(n);
      // 对每个输出尝试插入常量值
      for (const auto i : c10::irange(outputs.size())) {
        auto new_output = tryInsertConstant(*graph, outputs[i]);
        if (new_output) {
          made_change_ = true;
          GRAPH_UPDATE(
              "Folding %",
              n->outputs()[i]->debugName(),
              " with ",
              getHeader((*new_output)->node()));
          if (outputs[i].isNone()) {
            (*new_output)->setType(n->outputs()[i]->type());
          }
          n->outputs()[i]->replaceAllUsesWith(*new_output);
        }
        // 如果无法将 IValue 插入为常量，则放弃替换节点并让 DCE 移除它
      }
    }

    // 移除循环节点
    void removeLoopNode(Node* n) {
      auto loop_input_offset = 2; // 循环中的输入列表中的偏移量
      for (size_t i = 0; i < n->outputs().size(); ++i) {
        n->outputs().at(i)->replaceAllUsesWith(
            n->inputs().at(i + loop_input_offset));
      }
      made_change_ = true;
      n->destroy();
    }

    // 判断循环是否不会运行
    bool loopWillNotRun(Node* node) {
      Value* trip_count = node->inputs().at(0);
      int64_t iter_len = constant_as<int64_t>(trip_count).value_or(1);

      Value* start_cond = node->inputs().at(1);
      // ...
      // （此处省略了一部分代码）
      // ...
    }

    std::shared_ptr<Graph> graph_;
    bool aliasing_types_;
    bool ignore_custom_classes_;
    bool made_change_ = false;
  };
    // 从 start_cond 中获取常量 bool 值，如果无法获取则默认为 true
    bool cond_val = constant_as<bool>(start_cond).value_or(true);

    // 判断循环是否可能执行，条件为 cond_val 为真且 iter_len 大于 0
    bool loop_might_run = cond_val && iter_len > 0;
    if (!loop_might_run) {
      // 如果循环不会执行，则记录日志并删除未执行的循环
      GRAPH_UPDATE(
          "Removing unexecuted loop: ",
          *node,
          "\ntripcount: ",
          trip_count,
          " and start_cond: ",
          getHeader(start_cond->node()));
    }
    // 返回循环是否可能执行的结果的逻辑否定
    return !loop_might_run;
  }

  // 将条件为真的分支体内的节点移动到其所属的节点之前
  void inlineIfBody(Block* body) {
    Node* n = body->owningNode();
    for (auto it = body->nodes().begin(); it != body->nodes().end();) {
      Node* body_node = *it;
      // 在移动 body_node 之后，需要前进迭代器，因为其 next 指针将指向 n
      it++;
      body_node->moveBefore(n);
    }
    // 将节点 n 的输出替换为分支体的输出
    for (size_t i = 0; i < n->outputs().size(); ++i) {
      n->outputs().at(i)->replaceAllUsesWith(body->outputs().at(i));
    }
    // 注意：在此销毁节点 n，因为它可能包含诸如打印等副作用
    n->destroy();
  }

  // 折叠条件为真的分支体内的节点
  void inlineIf(Node* n) {
    auto input_bool = constant_as<bool>(n->input());
    AT_ASSERT(input_bool);
    // 更新图表，折叠条件为真的分支
    GRAPH_UPDATE(
        "Folding if ",
        getHeader(n->input()->node()),
        " where condition = ",
        *input_bool);
    // 根据条件的值（true 或 false）选择块的索引，并进行常量传播和内联操作
    size_t block_index = *input_bool ? 0 : 1;
    ConstantPropagation(n->blocks().at(block_index));
    inlineIfBody(n->blocks().at(block_index));
    // 标记已经进行了修改
    made_change_ = true;
  }

  // 替换并移除条件节点的输出
  void replaceAndRemoveIfOutput(Node* n, size_t i, Value* replacement) {
    n->outputs().at(i)->replaceAllUsesWith(replacement);
    n->eraseOutput(i);
    // 分别从两个分支块中移除指定位置的输出
    n->blocks().at(0)->eraseOutput(i);
    n->blocks().at(1)->eraseOutput(i);
  }

  // 移除条件节点的额外输出
  void removeExtraIfOutputs(Node* n) {
    // 断言节点类型为 prim::If，仅支持此类型的节点
    TORCH_CHECK(n->kind() == prim::If, "Only supported for If nodes");
    auto true_block = n->blocks()[0];
    auto false_block = n->blocks()[1];
    auto graph = n->owningGraph();
    auto initial_outputs = true_block->outputs().size();
    // 在当前插入点上执行操作
    WithInsertPoint guard(n);
    for (size_t i = 0; i < true_block->outputs().size();) {
      auto t_out = true_block->outputs().at(i);
      auto f_out = false_block->outputs().at(i);

      // 如果两个块的输出值相同，则替换并移除当前位置的输出
      if (true_block->outputs()[i] == false_block->outputs()[i]) {
        replaceAndRemoveIfOutput(n, i, true_block->outputs()[i]);
        continue;
      }

      // 如果真块的输出是常量且与假块的输出匹配，则插入新的常量并替换
      auto maybe_const = toIValue(t_out);
      auto eq = EqualNode();
      if (maybe_const && eq(t_out->node(), f_out->node())) {
        auto new_const = graph->insertConstant(*maybe_const);
        replaceAndRemoveIfOutput(n, i, new_const);
        continue;
      }

      i++; // 因为未移除当前索引，所以需要递增
    }
    // 标记是否进行了修改，比较初始输出与真块的输出大小
    made_change_ |= initial_outputs != true_block->outputs().size();
  }

  // 移除循环节点的额外输出
  void removeExtraLoopOutputs(Node* node) {
    auto initial_outputs = node->outputs().size();
    auto loop_body = node->blocks().at(0);
    auto loop_input_offset = 2; // 循环中携带的依赖项在输入列表中的偏移量
    auto loop_body_offset = 1; // 在块输入/输出中，循环中携带的依赖项的偏移量

    for (size_t i_1 = node->outputs().size(); i_1 > 0; --i_1) {
      size_t i = i_1 - 1;
      // 如果值不再改变，则移除输出
      if (loop_body->inputs().at(loop_body_offset + i) ==
          loop_body->outputs().at(loop_body_offset + i)) {
        auto node_input = node->inputs().at(loop_input_offset + i);
        node->outputs().at(i)->replaceAllUsesWith(node_input); // 用输入替换输出的所有使用
        loop_body->inputs()
            .at(loop_body_offset + i)
            ->replaceAllUsesWith(node_input); // 用输入替换循环体块的输入的所有使用
        node->eraseOutput(i); // 移除节点的输出
        node->removeInput(loop_input_offset + i); // 移除节点的输入
        loop_body->eraseInput(loop_body_offset + i); // 移除循环体块的输入
        loop_body->eraseOutput(loop_body_offset + i); // 移除循环体块的输出
      }
    }
    made_change_ |= initial_outputs != node->outputs().size(); // 标记是否进行了修改

  }

  bool noMutableValues(at::ArrayRef<Value*> values) {
    return std::none_of(values.begin(), values.end(), [](Value* v) {
      return AliasDb::isMutableType(v); // 检查值是否为可变类型
    });
  }

  AliasDb* getOrCreateAliasDb() {
    if (!aliasDb_) {
      aliasDb_ = std::make_unique<AliasDb>(graph_); // 如果别名数据库不存在则创建
    }
    return aliasDb_.get(); // 返回别名数据库指针
  }

  bool supportedNode(Node* n) {
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    bool no_mutation;
    if (aliasing_types_) {
      no_mutation = !getOrCreateAliasDb()->hasWriters(n); // 检查节点是否有写入者
    } else {
      no_mutation =
          noMutableValues(n->inputs()) && noMutableValues(n->outputs()); // 检查节点的输入和输出是否均为不可变
    }
    return no_mutation && !n->kind().is_onnx() &&
        skip_list.count(n->kind()) == 0 && !n->isNondeterministic() &&
        !n->hasSideEffects() && n->blocks().empty(); // 返回节点是否被支持
  }

  void ConstantPropagation(at::ArrayRef<Block*> blocks) {
    for (Block* block : blocks) {
      ConstantPropagation(block); // 对每个块进行常量传播
    }
  }

  void ConstantPropagation(Node* n) {
    bool constant_inputs =
        std::all_of(n->inputs().begin(), n->inputs().end(), [&](Value* v) {
          return v->node()->kind() == prim::Constant; // 检查节点的所有输入是否均为常量
        });
    if (n->kind() == prim::If) {
      // 如果可能，内联节点；否则检查是否可以简化输出
      if (constant_inputs) {
        inlineIf(n); // 内联 if 节点
      } else {
        ConstantPropagation(n->blocks()); // 对 if 节点的块进行常量传播
        removeExtraIfOutputs(n); // 移除额外的 if 节点输出
      }
    } else if (n->kind() == prim::Loop) {
      if (loopWillNotRun(n)) {
        removeLoopNode(n); // 移除不会运行的循环节点
      } else {
        ConstantPropagation(n->blocks()); // 对循环节点的块进行常量传播
        removeExtraLoopOutputs(n); // 移除额外的循环节点输出
      }
    } else if (constant_inputs && supportedNode(n)) {
      propagateNode(n); // 对支持的节点进行传播
    } else {
      ConstantPropagation(n->blocks()); // 对节点的块进行常量传播
    }
  }

  void ConstantPropagation(Block* block) {
    for (auto it = block->nodes().begin(); it != block->nodes().end();) {
      Node* n = *it;
      it++; // 前进迭代器，因为当前节点可能被销毁
      ConstantPropagation(n); // 对块中的每个节点进行常量传播
  }
}```

}

std::shared_ptr<Graph> graph_;
// 如果使用 aliasing_types，则延迟初始化，否则不初始化
std::unique_ptr<AliasDb> aliasDb_ = nullptr;
bool aliasing_types_;
bool made_change_ = false;
bool ignore_custom_classes_;
} // anonymous namespace
```