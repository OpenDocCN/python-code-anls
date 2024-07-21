# `.\pytorch\torch\csrc\jit\passes\inline_forked_closures.cpp`

```
// 包含 Torch 库中的头文件：用于 JIT 编译器的内联分叉闭包
#include <torch/csrc/jit/passes/inline_forked_closures.h>

// 包含 Torch 库中的头文件：用于 JIT 前端的 IR 发射器
#include <torch/csrc/jit/frontend/ir_emitter.h>

// Torch 的命名空间开始
namespace torch {
namespace jit {

// 闭包节点被发射为 (函数 %, 上下文元组 %) 的元组
// 在闭包内部，闭包被解包以设置所有闭合值
// 一个闭合了 a 和 b 的函数看起来像：
// def foo(context):
//  a, b = context
//
// 要分叉闭包，需要将上下文元组中的每个值设置为分叉节点的显式输入
// 然后在闭包子图中，用新的图输入替换上下文解包值
// fork(foo) ->
// def foo(a, b):
static void inlineForkedClosure(Node* fork_closure, NodeKind genKind) {
  // 获取函数上下文节点
  Node* function_context_node = fork_closure->input()->node();

  // 检查上下文节点的输入数量及类型是否符合闭包分叉的要求
  if (function_context_node->inputs().size() != 2 ||
      function_context_node->inputs().at(0)->node()->kind() != prim::Closure ||
      function_context_node->inputs().at(1)->node()->kind() !=
          prim::TupleConstruct) {
    throw ErrorReport(fork_closure->sourceRange()) << "Cannot fork this value";
  }

  // 获取闭包函数节点和上下文元组节点
  Node* function = function_context_node->inputs().at(0)->node();
  Node* context = function_context_node->inputs().at(1)->node();

  // 复制闭包函数的子图以备分叉使用，并获取当前图
  auto fork_graph = function->g(attr::Subgraph)->copy();
  auto g = fork_closure->owningGraph();

  // 创建新的分叉节点，设置其类型和源代码范围
  Node* fork_node = g->create(genKind, 1)
                        ->insertAfter(fork_closure)
                        ->setSourceRange(fork_closure->sourceRange());

  // 检查复制的子图是否有且仅有一个输入，并且其类型是元组类型
  if (fork_graph->inputs().size() != 1 ||
      !fork_graph->inputs().at(0)->type()->cast<TupleType>()) {
    throw ErrorReport(fork_node->sourceRange())
        << "Cannot fork lambda with parameters";
  }

  // 获取分叉子图的上下文输入，并确认其只被一个用户使用
  auto fork_graph_context = fork_graph->inputs().at(0);
  AT_ASSERT(fork_graph_context->uses().size() == 1);
  auto fork_graph_unpack = fork_graph_context->uses().at(0).user;

  // 遍历上下文元组的输入，将其作为分叉节点的输入，并替换子图中的解包值
  for (size_t i = 0; i < context->inputs().size(); ++i) {
    auto cont_input = context->inputs().at(i);
    fork_node->addInput(cont_input);
    auto inp = fork_graph->insertInput(i)->copyMetadata(cont_input);
    fork_graph_unpack->outputs().at(i)->replaceAllUsesWith(inp);
  }

  // 销毁解包节点并从子图中移除多余的输入
  fork_graph_unpack->destroy();
  fork_graph->eraseInput(fork_graph->inputs().size() - 1);

  // 复制分叉节点的元数据到闭包节点输出，并替换所有使用场景
  fork_node->output()->copyMetadata(fork_closure->output());
  fork_closure->output()->replaceAllUsesWith(fork_node->output());

  // 销毁原始闭包节点并将分叉子图设置到新的分叉节点中
  fork_closure->destroy();
  fork_node->g_(attr::Subgraph, fork_graph);

  // 运行清理 passes 来确保子图结构的完整性
  runCleanupPasses(fork_graph);
}

// 在一个块中内联分叉闭包节点的函数
static void inlineForkedClosures(Block* block) {
  // 遍历块中的每个节点
  for (auto it = block->nodes().begin(); it != block->nodes().end();) {
    Node* n = *it;
    it++;

    // 根据节点的种类执行不同的处理
    switch (n->kind()) {
      case prim::forkClosure: {
        inlineForkedClosure(n, prim::fork);
      } break;
      case prim::awaitableClosure: {
        inlineForkedClosure(n, prim::awaitable);
      } break;
      default: {
        // 对于非闭包分叉节点，递归处理其子块
        for (Block* b : n->blocks()) {
          inlineForkedClosures(b);
        }
      } break;
    }
  }
}
// Torch 的命名空间结束
} // namespace jit
} // namespace torch
void inlineForkedClosures(std::shared_ptr<Graph>& to_clean) {
  // 调用 inlineForkedClosures 函数，传入参数为 to_clean 所指向的 Graph 对象的 block() 方法返回的值
  inlineForkedClosures(to_clean->block());
}

} // namespace jit
} // namespace torch
```