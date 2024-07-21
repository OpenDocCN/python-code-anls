# `.\pytorch\torch\csrc\jit\passes\onnx\pattern_conversion\autograd_function_process.cpp`

```py
// 包含 Torch 的头文件：autograd_function_process.h 提供了处理自动求导函数的相关功能
#include <torch/csrc/jit/passes/onnx/pattern_conversion/autograd_function_process.h>

// 包含 Torch 的日志和辅助功能头文件
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/onnx/helper.h>
#include <torch/csrc/jit/passes/utils/subgraph_utils.h>

// Torch 命名空间开始
namespace torch {
namespace jit {

// 将子图转换为子块的函数定义
void convertSubgraphToSubBlock(Block* block) {
  // 迭代当前块的所有节点
  for (auto it = block->nodes().begin(), end = block->nodes().end();
       it != end;) {
    Node* node = *it++;
    // 如果节点的类型是 PythonOp
    if (node->kind() == prim::PythonOp) {
      // 构造子块
      auto subblock = node->addBlock();
      auto graph = subblock->owningGraph();

      std::unordered_map<Value*, Value*> env;
      // 填充子块中的子图节点
      auto subgraph = node->g(attr::Subgraph);
      // 处理子图的输入节点
      for (const auto i : c10::irange(subgraph->inputs().size())) {
        subblock->addInput()->copyMetadata(subgraph->inputs()[i]);
        env[subgraph->inputs()[i]] = subblock->inputs()[i];
      }
      // 处理子图的所有节点
      for (auto* n : subgraph->nodes()) {
        // 克隆节点到子块中，并建立环境映射
        auto cloned_n =
            subblock->appendNode(graph->createClone(n, [&](Value* v) {
              return env.find(v) != env.end() ? env[v] : v;
            }));
        // 处理节点的输出
        for (size_t i = 0; i < n->outputs().size(); ++i) {
          env[n->outputs().at(i)] = cloned_n->outputs().at(i);
          // 如果节点输出被子图输出使用，则注册为子块的输出
          auto it = std::find(
              subgraph->outputs().begin(),
              subgraph->outputs().end(),
              n->outputs()[i]);
          if (it != subgraph->outputs().end()) {
            subblock->registerOutput(cloned_n->outputs()[i]);
          }
        }
      }
      // 从 PythonOp 节点中移除 subgraph 属性，并递归处理子块
      node->removeAttribute(attr::Subgraph);
    }
    // 递归处理节点的所有子块
    for (auto block : node->blocks()) {
      convertSubgraphToSubBlock(block);
    }
  }
}

// 该函数仅用于 ONNX 转换
void ONNXAutogradFunctionProcess(std::shared_ptr<Graph>& graph) {
  // 转换主图的块为子块
  convertSubgraphToSubBlock(graph->block());
}

} // namespace jit
} // namespace torch
```