# `.\pytorch\torch\csrc\jit\passes\loop_unrolling.h`

```
#pragma once
// 预处理指令，确保头文件只被包含一次

#include <torch/csrc/jit/ir/ir.h>
// 包含 Torch 框架的图形表示头文件

namespace torch {
namespace jit {

// 如果图被修改则返回 true
TORCH_API bool UnrollLoops(std::shared_ptr<Graph>& graph);
// 展开所有循环的函数，接受一个图的共享指针作为参数

// 只展开常量循环。将不考虑循环块大小而展开它们
TORCH_API bool UnrollConstantLoops(std::shared_ptr<Graph>& graph);
// 只展开常量循环的函数，接受一个图的共享指针作为参数

TORCH_API Node* PeelLoop(Node* n, size_t times);
// 将指定的循环节点 n 迭代展开指定的次数 times

// 如果图被修改则返回 true
TORCH_API bool PeelProfilingLoops(const std::shared_ptr<Graph>& graph);
// 展开具有性能分析的循环，接受一个图的共享指针作为参数

struct TORCH_API LoopsPeeler {
  LoopsPeeler(std::function<bool(Node* n)> callback, size_t num_iterations = 1)
      : callback_(std::move(callback)), num_iterations_(num_iterations) {}
  // 构造函数，接受一个回调函数和迭代次数作为参数

  bool run(const std::shared_ptr<Graph>& graph);
  // 运行循环展开器的函数，接受一个图的共享指针作为参数

 private:
  void collectLoop(Node* n);
  // 收集指定节点的循环信息的私有方法

  void collectLoops(Block* block);
  // 收集指定块中的所有循环信息的私有方法

  void peelLoops();
  // 对收集到的循环进行展开的私有方法

  std::function<bool(Node* n)> callback_ = nullptr;
  // 回调函数，用于判断是否要展开指定的节点

  Node* in_loop_ = nullptr;
  // 当前正在处理的循环节点

  std::list<Node*> loops_to_peel_;
  // 待展开的循环节点列表

  size_t num_iterations_ = 1;
  // 展开循环的迭代次数
};

} // namespace jit
} // namespace torch
```