# `.\pytorch\torch\csrc\jit\passes\inliner.cpp`

```
// 包含 Torch JIT 中的内联器头文件
#include <torch/csrc/jit/passes/inliner.h>

// 包含 ATen 核心模块中的内部字符串头文件
#include <ATen/core/interned_strings.h>

// 包含 Torch JIT 中的函数实现头文件
#include <torch/csrc/jit/api/function_impl.h>

// 包含 Torch JIT 中的模块头文件
#include <torch/csrc/jit/api/module.h>

// 包含 Torch JIT 中的前端错误报告头文件
#include <torch/csrc/jit/frontend/error_report.h>

// 包含 Torch JIT 中的日志记录头文件
#include <torch/csrc/jit/jit_log.h>

// Torch 命名空间开始
namespace torch {
namespace jit {

// prim 命名空间引入 C10 prim 命名空间
namespace prim {
using namespace ::c10::prim;
}

// 尝试将节点转换为图函数对象
GraphFunction* tryToGraphFunction(Node* n) {
  // 若节点类型为 CallFunction
  if (n->kind() == prim::CallFunction) {
    // 断言输入节点为常量节点
    AT_ASSERT(n->input(0)->node()->kind() == prim::Constant);
    // 获取函数常量节点
    auto function_constant = n->input(0)->node();
    // 获取函数类型
    auto fun_type = function_constant->output()->type()->expect<FunctionType>();
    // 尝试将函数类型转换为图函数对象
    return tryToGraphFunction(*fun_type->function());
  }
  // 若节点类型为 CallMethod
  if (n->kind() == prim::CallMethod) {
    // 获取方法名
    const std::string& name = n->s(attr::name);
    // 若输入类型为类类型
    if (auto class_type = n->input(0)->type()->cast<ClassType>()) {
      // 获取类方法
      Function& function = class_type->getMethod(name);
      // 尝试将函数对象转换为图函数对象
      return tryToGraphFunction(function);
    }
  }
  // 默认返回空指针
  return nullptr;
}

// 静态函数：内联调用
static void inlineCalls(Block* block) {
  // 遍历块中的节点
  for (auto it = block->nodes().begin(), end = block->nodes().end();
       it != end;) {
    // 当前节点
    Node* cur = *it++;
    // 根据当前节点的类型进行分支处理
    switch (cur->kind()) {
      case prim::CallFunction: {
        // 如果当前节点是函数调用
        if (auto graphFunction = tryToGraphFunction(cur)) {
          // 尝试将当前函数调用优化为图函数对象
          auto function_constant = cur->input(0)->node();
          auto fun_type =
              function_constant->output()->type()->expect<FunctionType>();

          // 移除函数调用节点的第一个输入节点
          cur->removeInput(0);

          // 更新图形信息，指示正在将函数 'fun_type->function()->name()' 内联到当前节点 'cur' 中
          GRAPH_UPDATE(
              "Inlining function '",
              fun_type->function()->name(),
              "' to ",
              *cur);

          std::shared_ptr<Graph> g = nullptr;
          // 仅在调试/测试目的下为 JIT 优化图内联优化图形。
          // 我们仅在执行时插入回退函数，而不是在用于序列化的图形上
          bool fallback =
              function_constant->hasAttribute(Symbol::attr("fallback"));
          if (fallback && graphFunction->get_executor().isOptimized()) {
            auto exec_plans =
                graphFunction->get_executor().getDebugState().execution_plans;
            if (!exec_plans.empty()) {
              // 获取第一个执行计划的图形
              g = exec_plans.begin()->second.graph;
              // optimized_graph() 调用 Inline，因此我们只需在 JIT 优化图上显式调用内联，其中递归回退函数调用
              Inline(*g.get());
            }
          }
          // 如果 g 仍然为空，则使用优化后的图形
          if (g == nullptr) {
            g = graphFunction->optimized_graph();
          }

          // 更新图形信息，显示当前函数体为 'g'
          GRAPH_UPDATE("Function body: ", g);
          // 执行函数内联到当前节点 'cur'
          inlineCallTo(cur, graphFunction, g.get());
        }
      } break;
      case prim::CallMethod: {
        // 如果当前节点是方法调用
        if (auto graphFunction = tryToGraphFunction(cur)) {
          // 尝试将当前方法调用优化为图函数对象
          // 更新图形信息，指示正在将方法 'cur->s(attr::name)' 内联到当前节点 'cur' 中
          GRAPH_UPDATE("Inlining method '", cur->s(attr::name), "' to ", *cur);
          // 更新图形信息，显示当前函数体为优化后的方法图形
          GRAPH_UPDATE("Function body: ", graphFunction->optimized_graph());
          // 执行方法内联到当前节点 'cur'
          inlineCallTo(cur, graphFunction);
        }
      } break;
      default: {
        // 对于其他类型的节点，递归处理其包含的所有块
        for (auto b : cur->blocks()) {
          inlineCalls(b);
        }
      } break;
    }
}

void Inline(Graph& graph) {
  // 输出调用内联操作前的图形状态，用于调试和分析
  GRAPH_DUMP("Before Inlining: ", &graph);
  // 执行内联调用操作，修改图中的代码块
  inlineCalls(graph.block());
  // 输出调用内联操作后的图形状态，用于调试和分析
  GRAPH_DUMP("After Inlining: ", &graph);
}

} // namespace jit
} // namespace torch
```