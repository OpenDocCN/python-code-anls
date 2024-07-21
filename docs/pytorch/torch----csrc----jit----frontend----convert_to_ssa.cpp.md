# `.\pytorch\torch\csrc\jit\frontend\convert_to_ssa.cpp`

```
// 引入头文件：将要使用的 Torch JIT 前端和 IR 相关的头文件包含进来
#include <torch/csrc/jit/frontend/convert_to_ssa.h>
#include <torch/csrc/jit/frontend/exit_transforms.h>
#include <torch/csrc/jit/frontend/inline_loop_condition.h>
#include <torch/csrc/jit/frontend/ir_emitter.h>
#include <torch/csrc/jit/frontend/mini_environment.h>
#include <torch/csrc/jit/ir/ir.h>

// 定义命名空间 torch::jit
namespace torch::jit {

// 该类的作用是在图形变换期间向控制流节点添加加载和存储操作
struct ControlFlowLoadStores {
  // 向块的输入添加加载和存储操作
  static void addBlockInput(
      Block* b,
      const TypePtr& type,
      const std::string& name) {
    auto g = b->owningGraph();
    // 创建并插入一个存储节点，用于将名称为 name 的变量值存储到块的输入中
    g->createStore(name, b->addInput(name)->setType(type))
        ->insertAfter(b->param_node());
  }

  // 向块的输出添加加载和存储操作
  static void addBlockOutput(
      Block* exit_block,
      const TypePtr& type,
      const std::string& name) {
    WithInsertPoint insert(exit_block);
    auto g = exit_block->owningGraph();
    // 在退出块中插入加载节点，用于从名称为 name 的变量中加载值作为块的输出
    auto block_exit = g->insertNode(g->createLoad(name, type))->output();
    exit_block->registerOutput(block_exit);
  }

  // 向节点的输出添加加载和存储操作
  static void addNodeOutput(
      Node* n,
      const TypePtr& type,
      const std::string& name) {
    auto out = n->addOutput()->setType(type);
    // 如果名称是有意义的，设置输出节点的调试名称
    if (meaningfulName(name)) {
      out->setDebugName(name);
    }
    auto g = n->owningGraph();
    // 创建并插入一个存储节点，用于将节点的输出值存储到名称为 name 的变量中
    g->createStore(name, out)->insertAfter(n);
  }

  // 向节点的输入添加加载和存储操作
  static void addNodeInput(
      Node* n,
      const TypePtr& type,
      const std::string& name) {
    auto g = n->owningGraph();
    // 创建加载节点，将名称为 name 的变量加载为节点 n 的输入
    auto inp = g->createLoad(name, type)->insertBefore(n)->output();
    n->addInput(inp);
  }

  // 向 If 节点添加加载和存储操作
  void addIfLoadStores(Node* n) {
    auto true_block = n->blocks().at(0);
    auto false_block = n->blocks().at(1);

    // 递归添加控制流加载和存储操作到 true_block 和 false_block 中
    auto true_vars = addControlFlowLoadStores(true_block);
    auto false_vars = addControlFlowLoadStores(false_block);
    std::set<std::string> mutated_variables;

    // 检查哪些变量在 true_block 和 false_block 中被修改
    for (auto& v : true_vars->definedVariables()) {
      if (false_vars->findInAnyFrame(v)) {
        mutated_variables.insert(v);
      }
    }
    for (auto& v : false_vars->definedVariables()) {
      if (true_vars->findInAnyFrame(v)) {
        mutated_variables.insert(v);
      }
    }

    // 根据 emitIfElseBlocks 函数的逻辑，在变量在每个块中都被定义且块的类型可以统一时，输出节点
    // 这里是基于 ir_emitter.cpp 中 emitIfElseBlocks 的逻辑


这段代码中，主要是在 Torch JIT 编译器的图形转换过程中，负责向控制流节点（如条件语句和循环）添加加载和存储操作，确保变量的正确处理和跟踪。
  for (const auto& x : mutated_variables) {
    // 遍历被修改的变量列表
    auto true_type = true_vars->findInAnyFrame(x);
    // 查找变量 x 在 true_vars 中的类型
    auto false_type = false_vars->findInAnyFrame(x);
    // 查找变量 x 在 false_vars 中的类型
    auto unified =
        unifyTypes(true_type, false_type, /*default_to_union=*/true);
    // 对 true_type 和 false_type 进行类型统一，采用默认的联合类型

    addBlockOutput(true_block, true_type, x);
    // 将 true_block 的输出设置为变量 x 的 true_type 类型
    addBlockOutput(false_block, false_type, x);
    // 将 false_block 的输出设置为变量 x 的 false_type 类型
    addNodeOutput(n, *unified, x);
    // 将节点 n 的输出设置为变量 x 的统一类型 unified
  }
}

// loop_carried_outputs* = Loop(max_trip_count, start_condition,
//                              loop_carried_inputs*)
//                    block0(loop_counter, loop_carried_block*) {
//                       <body>
//                       -> (continue_condition, loop_carried_block_outputs*)
//                    }
// all loop_carried_... lists are the same length and represent the value of
// loop-carried variables whose definitions are updated as the loop executes
// in a way that ensure single static assignment.
void addLoopLoadStores(Node* n) {
  auto body_block = n->blocks().at(0);
  // 获取节点 n 的第一个块作为循环体的块
  auto loop_vars = addControlFlowLoadStores(body_block);
  // 分析循环体的控制流，返回循环变量信息

  for (const auto& name : loop_vars->definedVariables()) {
    // 遍历循环变量中定义的变量名集合
    // if the variable local to the loop body, then
    // we do not need a loop carried variable for it
    // 如果变量是局部于循环体的，则无需为其创建循环传递的变量
    auto parent_type = environment_stack->findInAnyFrame(name);
    // 在环境栈中查找变量 name 的类型
    if (!parent_type) {
      // 如果找不到父环境中的类型，则继续下一个变量名的处理
      continue;
    }

    // since the loop may execute 0 or many times, the output types
    // of the loop and the input loop carried dependencies are conservatively
    // the union of the output of the body and the input to the loop
    // 由于循环可能执行0次或多次，循环的输出类型和输入的循环传递依赖关系是
    // 体输出和循环输入的保守联合
    auto block_type = loop_vars->findInThisFrame(name);
    // 在当前块中查找变量 name 的类型
    auto unified_type = unifyTypes(parent_type, block_type).value();
    // 统一父类型和块类型，得到统一后的类型

    // Insert a store at the beginning of the loop block, so that all
    // loads of the variable will use the loop carried value
    // 在循环块的开头插入一个存储，以便所有变量的加载都使用循环传递的值
    addNodeInput(n, parent_type, name);
    // 向节点 n 添加变量 name 的输入，使用父类型
    addBlockInput(body_block, unified_type, name);
    // 向循环体块添加变量 name 的输入，使用统一后的类型
    addBlockOutput(body_block, block_type, name);
    // 向循环体块添加变量 name 的输出，使用块类型
    addNodeOutput(n, unified_type, name);
    // 向节点 n 添加变量 name 的输出，使用统一后的类型
  }
}

std::shared_ptr<TypeEnvironment> addControlFlowLoadStores(Block* block) {
  pushFrame(block);
  // 压入一个新的环境帧，使用给定的块

  for (Node* n : block->nodes()) {
    // 遍历块中的所有节点
    switch (n->kind()) {
      case prim::If: {
        addIfLoadStores(n);
        // 如果是 If 节点，则添加其加载和存储逻辑
      } break;
      case prim::Loop: {
        addLoopLoadStores(n);
        // 如果是 Loop 节点，则添加其加载和存储逻辑
      } break;
      case prim::Closure: {
        for (auto b : n->blocks()) {
          addControlFlowLoadStores(b);
          // 如果是 Closure 节点，则递归添加其控制流的加载和存储逻辑
        }
      } break;
      case prim::Store: {
        environment_stack->setVar(n->s(attr::name), n->input()->type());
        // 如果是 Store 节点，则在环境栈中设置变量名和输入类型的对应关系
      } break;
      case prim::ComprehensionScope: {
        addControlFlowLoadStores(n->blocks().at(0));
        // 如果是 ComprehensionScope 节点，则添加其控制流的加载和存储逻辑
      } break;
    }
  }
  return popFrame();
  // 弹出当前环境帧，并返回上一环境帧的指针
}

void pushFrame(Block* b) {
  environment_stack = std::make_shared<TypeEnvironment>(b, environment_stack);
  // 创建一个新的环境帧，并将其推入环境栈中
}

std::shared_ptr<TypeEnvironment> popFrame() {
  auto old_frame = environment_stack;
  // 保存当前环境帧的指针
  environment_stack = environment_stack->previous();
  // 将环境栈的指针指向上一环境帧
  return old_frame;
  // 返回保存的旧环境帧指针
}
    // 将环境栈指针指向下一个环境，实现环境栈的退栈操作
    environment_stack = environment_stack->next;
    // 返回运行前的环境帧指针，用于维护环境栈的一致性
    return old_frame;
  }

  // 运行指定图形的控制流加载与存储操作
  void run(std::shared_ptr<Graph>& graph) {
    // 向图形的基本块添加控制流加载与存储操作
    addControlFlowLoadStores(graph->block());
  }

  // 指向当前环境的栈顶环境指针，初始值为 nullptr
  std::shared_ptr<TypeEnvironment> environment_stack = nullptr;
};

// 给定一个图，其中已经添加了控制流节点的输出，并且加载和存储操作在图中表示，删除所有加载和存储操作。
struct EraseLoadStores {
  void eraseBlockLoadStores(Block* block) {
    // 将当前块压入环境栈
    pushFrame(block);
    // 遍历当前块中的所有节点
    for (auto it = block->nodes().begin(); it != block->nodes().end();) {
      auto n = *it;
      it++;

      switch (n->kind()) {
        case prim::Store: {
          // 将变量存储到环境栈中
          environment_stack->setVar(n->s(attr::name), n->input());
          // 销毁当前节点
          n->destroy();
        } break;
        case prim::Load: {
          // 获取加载的变量名
          auto name = n->s(attr::name);
          // 在任意帧中查找变量
          auto var = environment_stack->findInAnyFrame(name);
          // 确保类型检查已经设置了变量名
          TORCH_INTERNAL_ASSERT(
              var, "Typechecking should ensure the variable name is set");
          // 替换加载节点的输出为变量值
          n->output()->replaceAllUsesWith(var);
          // 销毁当前节点
          n->destroy();
        } break;
        case prim::ComprehensionScope: {
          // 在局部变量作用域中的写入不会泄露到图的其他部分
          auto body = n->blocks().at(0);
          eraseBlockLoadStores(body);
          // 将局部变量作用域内联到图中
          for (auto it_cmpr = body->nodes().begin();
               it_cmpr != body->nodes().end();) {
            Node* body_node = *it_cmpr;
            it_cmpr++;
            body_node->moveBefore(n);
          }
          // 销毁当前节点
          n->destroy();
        } break;
        default: {
          // 递归处理节点的所有子块
          for (auto b : n->blocks()) {
            eraseBlockLoadStores(b);
          }
        } break;
      }
    }
    // 弹出当前块的环境栈
    popFrame();
  }

  // 将给定块推入环境栈
  void pushFrame(Block* b) {
    environment_stack =
        std::make_shared<ValueEnvironment>(b, environment_stack);
  }

  // 弹出当前环境栈的顶部块
  std::shared_ptr<ValueEnvironment> popFrame() {
    auto old_frame = environment_stack;
    environment_stack = environment_stack->next;
    return old_frame;
  }

  // 运行加载和存储操作删除的主函数
  void run(std::shared_ptr<Graph>& graph) {
    eraseBlockLoadStores(graph->block());
  }

  // 环境栈，用于跟踪变量环境
  std::shared_ptr<ValueEnvironment> environment_stack = nullptr;
};

// 此 Pass 将 Break 和 Continue 转换为 LoopContinuations，
// 其形式为 LoopContinuations(%loop_continue_condition, *loop_carried_vars)
// Break 语句的条件设置为 false，并且 Continue 语句内联循环条件作为第一个输入。
struct LoopContinuations {
 public:
  // 运行转换 Pass 的入口函数
  void run(std::shared_ptr<Graph>& graph) {
    run(graph->block());
  }

 private:
  // 为节点添加循环传递的输出
  void addLoopCarriedOutputs(Node* n) {
    auto g = n->owningGraph();
    WithInsertPoint insert(n);
    // NOLINTNEXTLINE(clang-analyzer-core.CallAndMessage)
    auto continuation = curr_loop_->blocks().at(0)->return_node();
    // 遍历返回节点的所有输出
    for (auto out : continuation->inputs()) {
      auto load_node = out->node();
      // 确保加载节点是正确的类型
      TORCH_INTERNAL_ASSERT(load_node->kind() == prim::Load);
      // 克隆加载节点，并将其添加为当前节点的输入
      auto new_load =
          g->insertNode(g->createClone(load_node, [](Value* v) { return v; }));
      n->addInput(new_load->output());
    }
  }

  // 为块分配退出继续的操作
  void assignExitContinuations(Block* block) {
    for (auto it = block->nodes().begin(); it != block->nodes().end();) {
      // 迭代器指向当前节点，遍历基本块中的所有节点
      Node* n = *it;
      // 获取当前节点，并将迭代器向前移动
      it++;
      // 根据当前节点的类型进行不同的处理
      switch (n->kind()) {
        case prim::If: {
          // 如果当前节点是 If 类型，则递归为其两个分支分配出口继续点
          assignExitContinuations(n->blocks().at(0));
          assignExitContinuations(n->blocks().at(1));
        } break;
        case prim::Closure: {
          // 如果当前节点是 Closure 类型，则为其闭包块运行赋值出口继续点
          LoopContinuations closure_block;
          closure_block.run(n->blocks().at(0));
        } break;
        case prim::Loop: {
          // 如果当前节点是 Loop 类型，则处理循环结构
          Node* prev_loop = curr_loop_;
          curr_loop_ = n;
          // 设置当前循环节点为当前节点，为循环体块分配出口继续点
          assignExitContinuations(n->blocks().at(0));
          curr_loop_ = prev_loop;
          // 恢复之前的循环节点
        } break;
        case prim::ContinueStmt: {
          // 如果当前节点是 ContinueStmt 类型，则处理循环继续语句
          auto loop_continuation =
              graph_->create(prim::LoopContinuation, 0)->insertAfter(n);
          auto header_block = loop_continuation->addBlock();
          // NOLINTNEXTLINE(clang-analyzer-core.CallAndMessage)
          auto pre_header = curr_loop_->blocks().at(1);
          // 克隆循环头部块，作为循环继续点的头部块
          header_block->cloneFrom(pre_header, [](Value* v) { return v; });
          InlineBlockBeforeNode(n, header_block);
          // 在当前节点之前插入头部块
          loop_continuation->addInput(header_block->outputs().at(0));
          loop_continuation->eraseBlock(0);
          // 将头部块的输出作为输入，移除原始块
          addLoopCarriedOutputs(loop_continuation);
          // 添加循环携带的输出
          n->destroy();
          // 销毁当前节点
        } break;
        case prim::BreakStmt: {
          // 如果当前节点是 BreakStmt 类型，则处理循环退出语句
          auto loop_exit =
              graph_->create(prim::LoopContinuation, 0)->insertAfter(n);
          // 第一个输入是循环继续条件 - BreakStmt 将其设置为 false
          loop_exit->addInput(false_val_);
          // 添加循环携带的输出
          addLoopCarriedOutputs(loop_exit);
          // 销毁当前节点
          n->destroy();
        } break;
      }
    }
  }

  void run(Block* b) {
    {
      // 在给定的基本块上运行，为当前对象分配图形
      graph_ = b->owningGraph();
      // 设置插入点为基本块的第一个节点之前
      WithInsertPoint guard(b->nodes().front());
      // 在当前图形中插入一个常量 false
      false_val_ = graph_->insertConstant(false);
    }
    // 为给定的基本块分配出口继续点
    assignExitContinuations(b);
  }

  // 图形指针，指向当前正在处理的图形
  Graph* graph_ = nullptr;
  // 指向常量 false 的值指针
  Value* false_val_ = nullptr;
  // 指向当前循环节点的指针
  Node* curr_loop_ = nullptr;
};

// 转换为静态单赋值形式（SSA）的过程分为多个步骤。
// 首先，向图中添加控制流的加载和存储操作。
// 现在控制流输出已设置，可以移除 Break 和 Continue，确保正确的块结束（LoopContinuation）。
// 然后将循环条件内联到图中。
// 接着，擦除加载和存储操作。
// 最后，从图中移除 LoopContinuation。

void ConvertToSSA(std::shared_ptr<Graph>& graph) {
  // 创建控制流加载和存储的对象
  ControlFlowLoadStores ctrl;
  // 运行控制流加载和存储的处理过程
  ctrl.run(graph);

  // 创建退出变量处理对象
  LoopContinuations exit_vars;
  // 运行退出变量处理过程
  exit_vars.run(graph);

  // 内联循环条件到图中
  InlineLoopCondition(graph);

  // 创建加载和存储擦除对象
  EraseLoadStores erase_loads_stores;
  // 运行加载和存储擦除过程
  erase_loads_stores.run(graph);

  // 转换退出处理
  TransformExits(graph);
}

} // namespace torch::jit
```