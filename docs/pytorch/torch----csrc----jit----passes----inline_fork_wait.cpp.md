# `.\pytorch\torch\csrc\jit\passes\inline_fork_wait.cpp`

```py
// 包含 Torch 中的 JIT 模块头文件
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/inline_fork_wait.h>

// Torch 的命名空间
namespace torch {
namespace jit {

// 内部静态函数：将 prim::fork 节点内联展开
static void InlineForkWait(
    Block* b,  // 输入参数：表示一个代码块
    std::unordered_map<Value*, Value*>& future_remap) {  // 输入输出参数：映射未来值的哈希表

  // 获取当前代码块中的所有节点
  auto nodes = b->nodes();

  // 遍历所有节点，跟踪由 prim::fork 返回的未来值
  for (auto it = nodes.begin(); it != nodes.end(); it++) {
    auto node = *it;
    // 如果节点的类型不是 prim::fork，则继续下一个节点
    if (node->kind() != prim::fork) {
      continue;
    }
    // 设置插入点为当前节点的位置
    WithInsertPoint insert_guard(node);
    // 获取当前节点所属的图和其子图
    auto graph = b->owningGraph();
    auto subgraph = node->g(attr::Subgraph);

    // 将子图插入到当前图中，并用节点的输入替换图中的输入
    auto output = insertGraph(*graph, *subgraph, node->inputs());

    // 将 prim::fork 节点的输出映射到展开后的输出
    future_remap[node->output()] = output.at(0);
  }

  // 反向遍历节点，处理 prim::fork 返回的未来值是否应删除对应的 aten::wait 调用
  auto reversed = b->nodes().reverse();
  for (auto it = reversed.begin(); it != reversed.end(); it++) {
    auto node = *it;
    if (node->kind() == prim::fork) {
      // 如果当前节点是 prim::fork，则用其输出替换所有使用该输出的节点
      node->output()->replaceAllUsesWith(future_remap.at(node->output()));
      // 销毁当前节点
      it.destroyCurrent();
    } else if (node->kind() == aten::wait) {
      // 如果当前节点是 aten::wait，则检查其输入是否在未来值映射中
      AT_ASSERT(node->inputs().size() == 1);
      AT_ASSERT(node->outputs().size() == 1);
      // 如果输入是 prim::fork 的输出，则替换该节点的输出
      if (future_remap.count(node->input())) {
        node->output()->replaceAllUsesWith(future_remap.at(node->input()));
        // 销毁当前节点
        it.destroyCurrent();
      }
    }
  }

  // 递归内联处理所有节点中的代码块
  for (auto it = nodes.begin(); it != nodes.end(); it++) {
    auto node = *it;
    // 对于每个节点中的所有子块，递归调用内联展开函数
    for (auto sub_b : node->blocks()) {
      InlineForkWait(sub_b, future_remap);
    }
  }
}

// 公共函数：在给定的图中进行 prim::fork 和 aten::wait 的内联展开
void InlineForkWait(const std::shared_ptr<Graph>& graph) {
  // 创建用于映射未来值的哈希表
  std::unordered_map<Value*, Value*> future_remap;
  // 调用内部函数，开始内联展开处理
  InlineForkWait(graph->block(), future_remap);
  // 打印内联展开后的图结构
  GRAPH_DUMP("After InlineForkWait: ", graph);
}

} // namespace jit
} // namespace torch
```