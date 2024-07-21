# `.\pytorch\torch\csrc\jit\frontend\exit_transforms.cpp`

```
// 引入头文件：用于退出转换的前端操作
#include <torch/csrc/jit/frontend/exit_transforms.h>

// 引入必要的头文件
#include <ATen/core/jit_type.h>
#include <c10/util/irange.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/ir_views.h>
#include <torch/csrc/jit/runtime/graph_iterator.h>

// torch::jit 命名空间开始
namespace torch::jit {

// 枚举类型，表示退出状态的可能性
enum class ExitStatus { WILL, MIGHT, WONT, THROWS };

// 枚举类型，表示转换类型
enum class Transform { Returns, LoopContinuations };

// 结构体 ExitPair 继承自 std::pair，用于表示退出相关的值对
struct ExitPair : public std::pair<Value*, std::vector<Value*>> {
  using pair::pair;

  // 构造函数，初始化 exit_v 和 exit_vals
  ExitPair(Value* exit_v, at::ArrayRef<Value*> exit_val_ref) {
    std::vector<Value*> exit_vals;
    for (Value* v : exit_val_ref) {
      exit_vals.push_back(v);
    }
    AT_ASSERT(exit_v->type() == BoolType::get());
    this->first = exit_v;
    this->second = std::move(exit_vals);
  }

  // 返回是否已经退出的值
  Value* hasExited() const {
    return this->first;
  }

  // 返回退出时传递的值
  std::vector<Value*> exitValues() const {
    return this->second;
  }
};

/**
 * ExitTransformer 结构体，用于转换图，将所有指向块位置的退出节点从图中移除并统一处理。
 * 用于循环继续的退出节点是 LoopContinuation，用于图和闭包的退出节点是 ReturnStmt。
 *
 * 一旦遇到退出节点，我们将不再执行后续指令，直到达到退出目标为止。
 *
 * 对于可能已经遇到退出语句的块和控制流节点，我们在一个布尔值上条件化所有执行，
 * 该布尔值指示我们是否已经遇到退出，即 hasExited()。
 *
 * 该 Pass 还跟踪总是抛出异常的块，以便构造更简单的图。例如，如果在 if 语句的一个块中返回，
 * 而在另一个块中抛出异常，我们可以将节点视为总是返回，而不是在余下的块中条件化执行。
 */
struct ExitTransformer {
  // 构造函数，初始化使用的图
  ExitTransformer(std::shared_ptr<Graph> graph) : graph_(std::move(graph)) {
    // 在图的节点列表的开头设置插入点
    WithInsertPoint guard(graph_->block()->nodes().front());
    // 插入常量 true 和 false
    true_val_ = graph_->insertConstant(true);
    false_val_ = graph_->insertConstant(false);
    // throws_val_ 目前未初始化，因为我们在抛出之前将总是抛出
    // 将布尔类型作为参数调用getUnitValue函数，并将返回值赋给throws_val_
    throws_val_ = getUnitValue(BoolType::get());
  };

  // 将当前退出类型设置为ReturnStmt，然后转换图的所有出口
  void transformReturnStmts() {
    current_exit_kind_ = prim::ReturnStmt;
    transformExits(graph_->block());
  }

  // 将当前退出类型设置为LoopContinuation，然后转换图的所有出口
  void transformLoopContinuations() {
    current_exit_kind_ = prim::LoopContinuation;
    transformExits(graph_->block());
  }

 private:
  // 构造一个表示抛出异常退出的ExitPair
  ExitPair constructThrowsExitPair() {
    return ExitPair(throws_val_, std::vector<Value*>({}));
  }

  // 构造一个表示不会退出的ExitPair
  ExitPair constructWontExitPair() {
    return ExitPair(false_val_, std::vector<Value*>({}));
  }

  // 构造一个表示会退出的ExitPair，使用给定的退出值数组
  ExitPair constructWillExitPair(at::ArrayRef<Value*> exit_val_ref) {
    return ExitPair(true_val_, exit_val_ref);
  }

  // 根据ExitPair确定退出状态（WILL、WONT、THROWS、MIGHT）
  ExitStatus getExitStatus(ExitPair& exit_pair) {
    Value* exit_v = exit_pair.hasExited();
    if (exit_v == true_val_) {
      return ExitStatus::WILL;
    } else if (exit_v == false_val_) {
      return ExitStatus::WONT;
    } else if (exit_v == throws_val_) {
      return ExitStatus::THROWS;
    } else {
      return ExitStatus::MIGHT;
    }
  }

  // 返回指定块的拥有节点的类型符号，如果不存在则返回空符号
  static Symbol owningNodeKind(Block* block) {
    if (block->owningNode()) {
      return block->owningNode()->kind();
    }
    return Symbol();
  }

  // 检查指定块是否为图或闭包块
  static bool isGraphOrClosureBlock(Block* block) {
    return block->owningNode() == nullptr ||
        owningNodeKind(block) == prim::Closure;
  }

  // 移除指定块的所有输出
  static void removeOutputs(Block* b) {
    while (!b->outputs().empty()) {
      b->eraseOutput(0);
    }
  }

  // 注册指定块的输出值数组
  static void registerBlockOutputs(Block* b, at::ArrayRef<Value*> outs) {
    for (Value* out : outs) {
      b->registerOutput(out);
    }
  }

  // 替换指定块的输出值数组，首先移除所有输出，然后注册新的输出
  static void replaceBlockOutputs(Block* b, at::ArrayRef<Value*> outs) {
    removeOutputs(b);
    registerBlockOutputs(b, outs);
  }

  // 为指定节点添加条件输出值到其分支块中
  static void addIfOutputs(
      Node* n,
      at::ArrayRef<Value*> true_outs,
      at::ArrayRef<Value*> false_outs) {
    IfView if_view(n);
    registerBlockOutputs(if_view.thenBlock(), true_outs);
    registerBlockOutputs(if_view.elseBlock(), false_outs);
    for (const auto i : c10::irange(true_outs.size())) {
      auto out_type = unifyTypes(
          true_outs.at(i)->type(),
          false_outs.at(i)->type(),
          /*default_to_union=*/true);
      n->addOutput()->setType(*out_type);
    }
  }

  // 使用与values_to_match中值相同类型的未初始化值，创建一个值数组
  std::vector<Value*> matchValuesWithUnitialized(
      at::ArrayRef<Value*> values_to_match) {
    std::vector<Value*> match_values;
    for (Value* val : values_to_match) {
      match_values.push_back(getUnitValue(val->type()));
    }
    return match_values;
  }

  // 转换循环节点，获取其主体块并进行退出转换
  ExitPair transformLoop(Node* node) {
    LoopView loop(node);
    Block* body = loop.bodyBlock();
    auto exit_pair = transformExits(body);
    // 如果没有到外部退出循环，则不需要任何操作，对于抛出异常的情况返回WONT。
    ```
    // 如果某个分支的退出状态为 WONT 或 THROWS，执行相应处理
    if (getExitStatus(exit_pair) == ExitStatus::WONT ||
        getExitStatus(exit_pair) == ExitStatus::THROWS) {
      // 返回一个表示不会退出的 ExitPair
      return constructWontExitPair();
    }

    // 更新循环继续条件，以便在遇到退出情况时退出循环
    // 同时将 hasExited() 和 exitValues() 传播到循环外部

    // 创建一个新的 If 节点来更新循环的条件
    WithInsertPoint insert(body);
    auto new_if = graph_->insertNode(graph_->create(prim::If, 0));
    new_if->addInput(exit_pair.hasExited());
    new_if->addBlock()->registerOutput(false_val_);
    new_if->addBlock()->registerOutput(loop.nextCond());
    auto new_condition = new_if->addOutput()->setType(BoolType::get());
    loop.bodyBlock()->eraseOutput(0);
    loop.bodyBlock()->insertOutput(0, new_condition);

    // 将 hasExited() 添加到循环的输出中，如果没有进入循环，则未退出
    node->addInput(false_val_);
    body->addInput()->setType(BoolType::get());
    body->registerOutput(exit_pair.hasExited());
    Value* new_has_exited = node->addOutput()->setType(BoolType::get());

    // 添加退出值
    for (Value* exit_value : exit_pair.exitValues()) {
      auto typ = exit_value->type();
      node->addInput(getUnitValue(typ));
      node->addOutput()->setType(typ);
      body->addInput()->setType(typ);
      body->registerOutput(exit_value);
    }

    // 提取退出值
    auto exit_vals = node->outputs().slice(
        node->outputs().size() - exit_pair.exitValues().size());

    // 返回更新后的 ExitPair
    return ExitPair(new_has_exited, exit_vals);
  }

  // 计算 If 节点的退出状态
  ExitStatus calcIfExitStatus(ExitStatus then_status, ExitStatus else_status) {
    // 如果其中一个分支抛出异常，返回另一个分支的状态
    if (then_status == ExitStatus::THROWS) {
      return else_status;
    } else if (else_status == ExitStatus::THROWS) {
      return then_status;
    }

    // 如果两个分支都不会退出，则返回 WONT
    if (then_status == ExitStatus::WONT && else_status == ExitStatus::WONT) {
      return ExitStatus::WONT;
    }

    // 如果两个分支都会退出，则返回 WILL
    if (then_status == ExitStatus::WILL && else_status == ExitStatus::WILL) {
      return ExitStatus::WILL;
    }

    // 其他情况返回 MIGHT
    return ExitStatus::MIGHT;
  }

  // 递归转换 If 节点
  ExitPair transformIf(Node* node) {
    // 获取 If 节点的两个分支块
    auto then_block = node->blocks().at(0);
    auto else_block = node->blocks().at(1);

    // 分别对两个分支块进行退出转换
    auto then_pair = transformExits(then_block);
    auto else_pair = transformExits(else_block);

    // 获取两个分支的退出状态
    auto then_status = getExitStatus(then_pair);
    // 获取 else 分支的退出状态
    auto else_status = getExitStatus(else_pair);

    // 计算 if 语句的退出状态
    auto if_status = calcIfExitStatus(then_status, else_status);

    // 如果 if 语句的退出状态为 THROWS，则返回一个抛出异常的退出对
    if (if_status == ExitStatus::THROWS) {
      return constructThrowsExitPair();
    }
    // 如果 if 语句的退出状态为 WONT，则返回一个不会退出的退出对
    if (if_status == ExitStatus::WONT) {
      return constructWontExitPair();
    }

    // 如果 then 分支的退出状态为 WONT 或 THROWS，则将 else 分支的退出值与未初始化的值匹配，并更新 then_pair
    if (then_status == ExitStatus::WONT || then_status == ExitStatus::THROWS) {
      std::vector<Value*> exit_vals =
          matchValuesWithUnitialized(else_pair.exitValues());
      then_pair = ExitPair(then_pair.hasExited(), exit_vals);
    } else if (
        // 如果 else 分支的退出状态为 WONT 或 THROWS，则将 then 分支的退出值与未初始化的值匹配，并更新 else_pair
        else_status == ExitStatus::WONT || else_status == ExitStatus::THROWS) {
      std::vector<Value*> exit_vals =
          matchValuesWithUnitialized(then_pair.exitValues());
      else_pair = ExitPair(else_pair.hasExited(), exit_vals);
    }

    // 创建一个未初始化的变量，根据 if 语句的退出状态确定其值
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    Value* has_exited;
    if (if_status == ExitStatus::WILL) {
      // 如果 if 语句会退出，则设置 has_exited 为 true_val_
      // 以维持这样的不变性：如果 hasExited() == true_val_，则表示已经退出
      has_exited = true_val_;
    } else {
      // 否则，调用 addIfOutputs 添加 if 语句的输出，并更新 has_exited 为最后一个输出
      addIfOutputs(node, {then_pair.hasExited()}, {else_pair.hasExited()});
      has_exited = node->outputs().at(node->outputs().size() - 1);
    }

    // 调用 addIfOutputs 添加 then 和 else 分支的退出值到输出中
    addIfOutputs(node, then_pair.exitValues(), else_pair.exitValues());
    size_t num_exit_vals = then_pair.exitValues().size();
    // 提取节点的输出，切片为退出值
    auto exit_vals =
        node->outputs().slice(node->outputs().size() - num_exit_vals);
    // 返回一个包含 has_exited 和 exit_vals 的退出对
    return ExitPair(has_exited, exit_vals);
  }

  // 递归转换 With 节点
  ExitPair transformWith(Node* node) {
    // 获取 With 节点的主体块
    auto body_block = node->blocks().at(0);
    // 对主体块执行退出转换
    auto body_pair = transformExits(body_block);
    // 返回转换后的退出对
    return body_pair;
  }

  // 保护块内其余节点的函数，使用一个 if 节点来保护其余节点，接受 hasExited 作为条件
  ExitPair guardBlockNodes(
      Block* block,
      const ExitPair& exit_pair,
      graph_node_list_iterator& iter) {
    // 在迭代器当前位置前创建一个新的 if 节点
    auto new_if = graph_->create(prim::If, 0)->insertBefore(*iter);
    // 将 hasExited 作为输入添加到新的 if 节点
    new_if->addInput(exit_pair.hasExited());

    // 添加 exit_block 和 guard_block 到新的 if 节点中
    auto exit_block = new_if->addBlock();
    auto guard_block = new_if->addBlock();

    // 将剩余节点移动到 guard_block 中
    while (iter != block->nodes().end()) {
      auto node = *iter++;
      node->moveBefore(guard_block->return_node());
    }

    std::vector<Value*> exit_block_vals;
    // 在退出后，只有 hasExited() 和 exitValues() 会被使用，因此将现有块输出与未初始化的值匹配
    exit_block_vals = matchValuesWithUnitialized(block->outputs());

    // 设置新的 if 节点具有原始块的相同输出，然后用新 if 的输出替换原始块的输出
    // 遍历当前块的输出
    for (size_t i = 0; i < block->outputs().size(); ++i) {
      // 将当前块的输出注册到退出块的输出值
      exit_block->registerOutput(exit_block_vals.at(i));
      // 将当前块的输出注册到守护块的输出值
      guard_block->registerOutput(block->outputs().at(i));
      // 将当前块的输出类型添加到新的条件语句块的输出
      new_if->addOutput()->setType(block->outputs().at(i)->type());
    }

    // 清空当前块的所有输出
    while (!block->outputs().empty()) {
      block->eraseOutput(0);
    }
    // 将新条件语句块的所有输出注册到当前块
    for (auto out : new_if->outputs()) {
      block->registerOutput(out);
    }

    // 在退出块的返回节点之前创建一个新的图节点
    graph_->create(current_exit_kind_, {exit_pair.exitValues()}, 0)
        ->insertBefore(exit_block->return_node());
    // 返回转换后的新条件语句块
    return transformIf(new_if);
  }

  // 这些节点可能有用途，
  // 例如在以下情况下：
  // if i == 1:
  //    break
  //    j = j + 1
  // 其中 j + 1 的值将成为块的输出，但由于它们永远不会被使用，将它们替换为未初始化的值是安全的
  void destroyNodeAfterExit(Node* n) {
    // 遍历节点的所有输出
    for (auto output : n->outputs()) {
      // 如果输出有使用，则用未初始化值替换所有使用
      if (!output->uses().empty()) {
        output->replaceAllUsesWith(getUnitValue(output->type()));
      }
    }
    // 销毁节点
    n->destroy();
  }

  // 在退出后删除节点
  void deleteAfterExitNodes(Block* block, graph_node_list_iterator& iter) {
    // 如果迭代器已经在块的节点末尾，则直接返回
    if (iter == block->nodes().end()) {
      return;
    }
    // 设置插入点为块的节点列表的开始
    WithInsertPoint insert(*block->nodes().begin());
    // 以相反顺序销毁节点，以便在销毁时节点没有使用
    for (auto it = block->nodes().reverse().begin(); it != iter;) {
      Node* n = *it++;
      // 如果节点不是返回节点，则销毁节点
      if (*it != block->return_node()) {
        destroyNodeAfterExit(n);
      }
    }
    // 销毁迭代器指向的节点
    destroyNodeAfterExit(*iter);
  }

  // 如果进入循环块并且正在转换循环继续，或者进入闭包/图块并且正在转换返回语句，则更新目标块为新块
  // 否则，目标块保持不变
  void updateTargetBlock(Block* block) {
    // 如果当前块的所有者节点类型是循环，并且当前退出类型是循环继续
    if (owningNodeKind(block) == prim::Loop &&
        current_exit_kind_ == prim::LoopContinuation) {
      // 更新目标块为当前块
      target_block_ = block;
    } else if (
        // 如果当前块是图块或闭包块，并且当前退出类型是返回语句
        isGraphOrClosureBlock(block) &&
        current_exit_kind_ == prim::ReturnStmt) {
      // 更新目标块为当前块
      target_block_ = block;
    }
  }

  // 转换块的退出
  ExitPair transformExits(Block* block) {
    // 保存之前的目标块
    Block* prev_target_block = target_block_;
    // 更新目标块为当前块
    updateTargetBlock(block);
    // 构造不会退出的退出对
    ExitPair exit_pair = constructWontExitPair();
    // 迭代遍历基本块中的节点列表
    for (auto it = block->nodes().begin(); it != block->nodes().end();) {
      // 获取当前节点并移动迭代器到下一个节点
      Node* node = *it;
      it++;
      // 根据节点类型进行不同的处理
      switch (node->kind()) {
        case prim::RaiseException: {
          // 如果是抛出异常节点，构造抛出异常的退出对
          exit_pair = constructThrowsExitPair();
        } break;
        case prim::ReturnStmt:
        case prim::LoopContinuation: {
          // 如果是返回语句或者循环继续节点，并且类型匹配当前退出类型
          if (node->kind() == current_exit_kind_) {
            // 构造即将退出的退出对，并销毁当前节点
            exit_pair = constructWillExitPair(node->inputs());
            node->destroy();
          }
        } break;
        case prim::If: {
          // 如果是条件语句节点，转换处理该节点
          exit_pair = transformIf(node);
        } break;
        case prim::With: {
          // 如果是with语句节点，转换处理该节点
          exit_pair = transformWith(node);
        } break;
        case prim::Closure: {
          // 如果是闭包声明节点，处理闭包内部的退出情况，不影响外部
          transformExits(node->blocks().at(0));
        } break;
        case prim::Loop: {
          // 如果是循环节点，转换处理该节点
          exit_pair = transformLoop(node);
        } break;
      }

      // 如果当前节点可能导致退出，需要条件性执行后续节点；如果当前节点将会退出，可以删除后续所有节点。
      ExitStatus status = getExitStatus(exit_pair);
      if (status == ExitStatus::WILL || status == ExitStatus::THROWS) {
        // 根据退出状态删除基本块中当前节点之后的所有节点并退出循环
        deleteAfterExitNodes(block, it);
        break;
      }
      if (status == ExitStatus::MIGHT) {
        // 如果可能会退出，并且还有后续节点，保护后续节点并退出循环
        if (it != block->nodes().end()) {
          exit_pair = guardBlockNodes(block, exit_pair, it);
        }
        break;
      }
    }

    // 如果目标块是当前处理的块，更新输出值为退出值；由于退出不会扩展到外部，将返回退出标志置为false，并重置目标块为之前的块。
    if (target_block_ == block) {
      // 如果可能已经退出，使用新的退出值；否则使用现有块的输出。
      if (getExitStatus(exit_pair) == ExitStatus::MIGHT) {
        // 创建一个新的条件语句节点，并将其插入到返回节点之前
        auto new_if =
            graph_->create(prim::If, 0)->insertBefore(block->return_node());
        new_if->addBlock();
        new_if->addBlock();
        new_if->addInput(exit_pair.hasExited());
        // 添加条件语句的输出
        addIfOutputs(new_if, exit_pair.exitValues(), block->outputs());
        // 替换基本块的输出为条件语句的输出
        replaceBlockOutputs(block, new_if->outputs());
      } else if (getExitStatus(exit_pair) == ExitStatus::WILL) {
        // 直接替换基本块的输出为退出值
        replaceBlockOutputs(block, exit_pair.exitValues());
      }

      // 重置退出状态，退出只能影响其目标块。
      exit_pair = constructWontExitPair();
    }
    // 重置目标块为之前的目标块
    target_block_ = prev_target_block;
    // 返回退出对
    return exit_pair;
  }

  // 获取给定类型的Unit值
  Value* getUnitValue(const TypePtr& type) {
    // 查找给定类型在 unit_values_ 中的缓存值
    auto maybe_val = unit_values_.find(type);
    // 如果找到，则返回缓存的值
    if (maybe_val != unit_values_.end()) {
      return maybe_val->second;
    }
    // 否则，创建一个未初始化的值，并插入到图中的参数节点之后，然后获取其输出
    auto unit = graph_->createUninitialized(type)
                    ->insertAfter(graph_->param_node())
                    ->output();
    // 将新创建的值缓存到 unit_values_ 中，以备后续重用
    unit_values_[type] = unit;
    // 返回新创建的值
    return unit;
  }

  // 存储不同类型的单元值的缓存
  std::unordered_map<TypePtr, Value*> unit_values_;

  // 当前退出类型，可能是 LoopContinuation/ReturnStmt
  Symbol current_exit_kind_;

  // 真值、假值和异常值
  Value* true_val_;
  Value* false_val_;
  Value* throws_val_;

  // 当我们看到 current_exit_kind_ 时，这是值将要退出的块
  // 例如，当我们转换循环继续时：
  // for i in range(5):
  //   while i < 3:
  //     continue
  //   break
  // 当我们转换 for 循环块时，target_block_ 将被设置为 for 块。
  // 然后，当我们进入 while 循环时，target_block_ 将成为 while 循环块。
  // 当我们完成转换 while 循环时，target_block_ 将被重新设置为 for 块。
  Block* target_block_ = nullptr;
  
  // 与该实例关联的共享图形对象
  std::shared_ptr<Graph> graph_;
};

// 检查连续的 if 语句是否可以内联优化
static bool inlineConsecutiveIfs(Node* node) {
  // 如果当前节点不是 if 语句或者下一个节点也不是 if 语句，则无法内联优化
  if (node->kind() != prim::If || node->next()->kind() != prim::If) {
    return false;
  }

  // 获取第一个和第二个 if 语句的视图
  IfView first_if(node);
  IfView second_if(node->next());

  // 第二个 if 语句的条件必须依赖于第一个 if 语句的输出值才能进行内联优化
  if (second_if.cond()->node() != node) {
    return false;
  }

  // 如果两个 if 语句的输出值不是常量或者值相同，则无法进行内联优化
  auto input_offset = second_if.cond()->offset();
  auto maybe_then_value = toIValue(first_if.thenOutputs().at(input_offset));
  auto maybe_else_value = toIValue(first_if.elseOutputs().at(input_offset));
  if (!maybe_then_value || !maybe_else_value ||
      maybe_then_value->toBool() == maybe_else_value->toBool()) {
    return false;
  }

  // 获取第一个和第二个 if 语句的输出布尔值
  bool then_value = maybe_then_value->toBool();
  bool else_value = maybe_else_value->toBool();

  // 对于两次循环（第一个和第二个块），分别处理 then 和 else 块的内联复制
  for (const auto i : c10::irange(2)) {
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    Block* first_if_block;
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    Block* second_if_block;

    if (i == 0) {
      first_if_block = first_if.thenBlock();
      second_if_block =
          then_value ? second_if.thenBlock() : second_if.elseBlock();
    } else {
      first_if_block = first_if.elseBlock();
      second_if_block =
          else_value ? second_if.thenBlock() : second_if.elseBlock();
      ;
    }

    // 替换第二个 if 语句中使用的来自第一个 if 语句的输出值为相应的块作用域内的等效值
    auto value_map = [&](Value* v) {
      if (v->node() != first_if.node()) {
        return v;
      }
      auto offset = v->offset();
      return first_if_block->outputs().at(offset);
    };

    // 从 second_if_block 复制到 first_if_block，并且将块输出从 second_if_block 复制到 first_if_block
    first_if_block->cloneFrom(second_if_block, value_map);
  }

  // 替换第二个 if 语句的输出值为第一个 if 语句的输出值，并销毁第二个 if 语句节点
  for (Value* output : second_if.outputs()) {
    auto new_out = first_if.node()->addOutput()->copyMetadata(output);
    output->replaceAllUsesWith(new_out);
  }
  second_if.node()->destroy();
  return true;
}

// 在早期返回后，将所有后续执行条件化
// 这意味着像下面这样的代码：
// if x:
//     return 1
// return 2
// 将被生成为一个检查 `if x` 的 if 语句，然后是一个将执行条件化的第二个 if 语句。
// 我们可以重写这样的情况，使得上述示例被重写为：if x:
//   return 1
// else:
//   return 2
static void inlineConsecutiveIfs(Block* block) {
  // 遍历块中的所有节点
  for (auto it = block->nodes().begin(), end = block->nodes().end();
       it != end;) {
    // 递归处理块中的每一个子块
    for (Block* b : it->blocks()) {
      inlineConsecutiveIfs(b);
    }
    // 如果我们合并了两个相邻的 if 语句块，需要检查当前节点和新的下一个节点
    if (!inlineConsecutiveIfs(*it)) {
      // 如果当前节点不符合合并条件，则将迭代器 it 向后移动到下一个节点
      it++;
    }
// Adds prim::With nodes to a graph to handle early exits between prim::Enter and prim::Exit nodes.
// More specifically, it transforms IR that looks for Enter and Exit pairs and encapsulates the intervening nodes.
static void convertEnterExitNodesToWithBlocks(std::shared_ptr<Graph>& graph) {
    // First, find all Enter-Exit pairs up front to avoid iterator invalidation
    // issues later when moving nodes around. Do this by iterating through the
    // nodes of the graph while keeping a stack of encountered Enter nodes. Each
    // time an Exit node is seen, its corresponding Enter node must be at the
    // top of the stack. Pop it and record the pair.
    std::vector<std::pair<Node*, Node*>> enter_exit_pairs;
    std::vector<Node*> enter_node_stack;

    DepthFirstGraphNodeIterator it(graph);
    Node* node = it.next();

    while (node) {
        if (node->kind() == prim::Enter) {
            enter_node_stack.emplace_back(node);
        } else if (node->kind() == prim::Exit) {
            // enter_node_stack should not be empty.
            TORCH_INTERNAL_ASSERT(!enter_node_stack.empty());
            // The input to this Exit node should be the same as that of the Enter
            // node on the top of the enter_node_stack.
            TORCH_INTERNAL_ASSERT(
                enter_node_stack.back()->input(0) == node->input(0));
            // Record the pair.
            enter_exit_pairs.emplace_back(enter_node_stack.back(), node);
            enter_node_stack.pop_back();
        }

        node = it.next();
    }

    // The stack should be empty; an Exit should have been found for every Enter.
    TORCH_INTERNAL_ASSERT(enter_node_stack.empty());

    // Now, add a With block for each Enter-Exit pair. The innermost pairs were
    // found first, so they will be converted first.
    for (auto& pair : enter_exit_pairs) {
        Node* enter = pair.first;
        Node* exit = pair.second;

        auto* with = graph->create(prim::With, /*num_outputs=*/0);
        auto* body_block = with->addBlock();
        auto* exit_block = with->addBlock();

        // Insert the With after the Enter.
        Node* cur = enter->next();
        Node* insert_point = body_block->param_node();

        // Move all of the nodes between the Enter and Exit into the body block.
        while (cur != exit) {
            auto* next = cur->next();
            cur->moveAfter(insert_point);
            insert_point = insert_point->next();
            cur = next;
        }

        // Move the Exit node into the exit block.
        exit->moveAfter(exit_block->param_node());
        with->insertAfter(enter);
    }
}
static void convertWithBlocksToEnterExitNodes(std::shared_ptr<Graph>& graph) {
  // 首先，查找所有 With 块，以避免稍后移动节点时迭代器失效的问题。
  std::vector<Node*> with_nodes;

  // 使用深度优先节点迭代器遍历图中的节点
  DepthFirstGraphNodeIterator it(graph);
  Node* node = it.next();

  while (node) {
    // 如果节点类型为 prim::With，则将其加入 with_nodes 中
    if (node->kind() == prim::With) {
      with_nodes.emplace_back(node);
    }
    // 获取下一个节点
    node = it.next();
  }

  // 对于每个 With 节点：
  for (auto& node : with_nodes) {
    // 获取 With 块和 Exit 块
    auto* body_block = node->blocks().at(0);
    auto* exit_block = node->blocks().at(1);

    std::vector<Node*> to_append;

    // 记录所有需要附加的节点，以确保在移动节点时不会因为迭代器失效而出现问题。
    for (auto body_node : body_block->nodes()) {
      to_append.emplace_back(body_node);
    }

    for (auto exit_node : exit_block->nodes()) {
      to_append.emplace_back(exit_node);
    }

    Node* cur = node->prev();

    // 将 With 块内的所有节点移出该块
    for (auto& node : to_append) {
      node->moveAfter(cur);
      cur = node;
    }
    // 销毁当前 With 节点
    node->destroy();
  }
}

// 此 pass 接受一个图，其中包含 LoopContinuation 和 ReturnStmts，并在图中擦除它们，
// 正确设置块的输出。
// prim::LoopContinuation(*vals) 表示这些值指向最近的循环块。
// prim::ReturnStmt(*vals) 表示这些值指向最近的闭包或图块。
// 一旦遇到退出节点，我们不会执行任何进一步的指令，直到块的退出达到其目的地。
// 如果遇到包含可能已经遇到退出节点的嵌套块的节点，例如在一个块中退出而在另一个块中不退出的 if 语句，
// 我们使用布尔值来指示是否已经遇到退出。然后，我们条件化进一步的执行。

// Python 示例：
// while i < 5:
//   if i == 3:
//     i += 1
//     continue
//   i += 2
//
// -> 转换为：
//
// continue_loop = i < 5
// while continue_loop:
//   if i == 3:
//     i = i + 1
//     continue_loop = i < 5
//     did_exit = True
//   if did_exit:
//     pass
//   else:
//     i = i + 2
//     continue_loop = i < 5
// IR 进入 pass 时的形式：
// %36 : bool = aten::lt(%i.1, %3)
// %i : int = prim::Loop(%1, %36, %i.1)
//   block0(%5 : int, %i.17 : int):
//     %8 : bool = aten::eq(%i.17, %7)
//     %i.16 : int = prim::If(%8)
//       block0():
//         %i.6 : int = aten::add(%i.17, %11)
//         %33 : bool = aten::lt(%i.6, %3)
//          = prim::LoopContinuation(%33, %i.6)
//         -> (%i.6)
//       block1():
//         -> (%i.17)
//     %i.13 : int = aten::add(%i.16, %19)
//     %4 : bool = aten::lt(%i.13, %3)
//     -> (%4, %i.13)
// return (%i)

// -> 转换为
//
// %false_val : bool = prim::Constant[value=0]()
// %true_val : bool = prim::Constant[value=1]()
// %40 : int = prim::Uninitialized()
// %39 : bool = prim::Uninitialized()
// %36 : bool = aten::lt(%i.1, %3)
// %i : int = prim::Loop(%1, %36, %i.1)
//   block0(%5 : int, %i.17 : int):
//     %8 : bool = aten::eq(%i.17, %7)
//     %did_exit : bool, %continue_loop : bool, %43 : int, %i.16 : int =
//     prim::If(%8)
//       block0():
//         %i.6 : int = aten::add(%i.17, %11)
//         %33 : bool = aten::lt(%i.6, %3)
//         -> (%true_val, %33, %i.6, %i.6)
//       block1():
//         -> (%false_val, %39, %40, %i.17)
//     %44 : bool, %i : int = prim::If(%did_exit)
//       block0():
//         -> (%continue_loop, %43)
//       block1():
//         %i.13 : int = aten::add(%i.16, %19)
//         %4 : bool = aten::lt(%i.13, %3)
//         -> (%4, %i.13)
//     -> (%44, %i)

// 将图中的进入和退出节点转换为带块的形式
void TransformExits(std::shared_ptr<Graph>& graph) {
  convertEnterExitNodesToWithBlocks(graph);
  // 使用 ExitTransformer 处理循环继续语句的转换
  ExitTransformer e_loop(graph);
  e_loop.transformLoopContinuations();
  // 使用 ExitTransformer 处理返回语句的转换
  ExitTransformer e_ret(graph);
  e_ret.transformReturnStmts();
  // 内联连续的条件语句块
  inlineConsecutiveIfs(graph->block());
  // 将带块的形式转换回进入和退出节点
  convertWithBlocksToEnterExitNodes(graph);
}
// namespace torch::jit
```