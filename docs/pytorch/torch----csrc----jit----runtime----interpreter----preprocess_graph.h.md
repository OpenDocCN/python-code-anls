# `.\pytorch\torch\csrc\jit\runtime\interpreter\preprocess_graph.h`

```
#pragma once

# 预处理指令，确保头文件只被包含一次


#include <memory>
#include <unordered_map>
#include <torch/csrc/jit/ir/ir.h>

# 包含必要的标准库头文件和Torch库中的IR头文件


namespace torch::jit::interpreter {

# 进入命名空间 torch::jit::interpreter


// pre-processing that happens once per graph

# 描述这段代码实现的功能：每个图形仅进行一次的预处理


struct PreprocessGraph {
  explicit PreprocessGraph(Graph& g);

# 定义一个结构体 PreprocessGraph，用于进行图形的预处理，接受一个 Graph 引用作为参数


  // Outputs of the preprocessing:
  std::shared_ptr<Graph> graph;
  std::unordered_map<Node*, bool> can_emit_inline;
};

# 结构体的成员变量：
# - graph：预处理后的图形对象，使用智能指针进行管理
# - can_emit_inline：存储每个节点是否可以内联的无序映射表


} // namespace torch::jit::interpreter

# 结束命名空间 torch::jit::interpreter
```