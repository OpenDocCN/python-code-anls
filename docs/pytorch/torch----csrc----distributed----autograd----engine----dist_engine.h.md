# `.\pytorch\torch\csrc\distributed\autograd\engine\dist_engine.h`

```
#pragma once

#include <mutex> // 包含互斥量（mutex）的头文件
#include <unordered_set> // 包含无序集合（unordered_set）的头文件

#include <torch/csrc/autograd/engine.h> // 包含 Torch 自动求导引擎的头文件
#include <torch/csrc/autograd/function.h> // 包含 Torch 自动求导函数的头文件
#include <torch/csrc/autograd/functions/basic_ops.h> // 包含 Torch 自动求导基本操作函数的头文件
#include <torch/csrc/distributed/autograd/context/context.h> // 包含 Torch 分布式自动求导上下文的头文件

namespace torch {
namespace distributed {
namespace autograd {

// Forward declaration.
class BackwardPassCleanupGuard; // 前向声明类 BackwardPassCleanupGuard

// This is a singleton class responsible for running distributed backward
// passes. This engine relies heavily on the vanilla autograd engine and tries
// to re-use it as much as possible. This class is mostly responsible for the
// distributed aspects of autograd and tries to hook into the autograd engine
// where convenient.
// 这是一个单例类，负责运行分布式的反向传播过程。该引擎在很大程度上依赖于基本的自动求导引擎，并尽可能地重用它。该类主要负责自动求导的分布式方面，并尝试在方便的地方接入自动求导引擎。

// Unlike the vanilla autograd engine, the distributed autograd engine
// accumulates the gradients in the appropriate DistAutogradContext. This avoids
// multiple trainer nodes stomping on each others gradients.
// 与基本的自动求导引擎不同，分布式自动求导引擎在适当的 DistAutogradContext 中累积梯度。这样可以避免多个训练节点互相覆盖梯度的问题。

};

// Guard to clean up resources once the backward pass is done.
// 用于在反向传播完成后清理资源的保护器
class BackwardPassCleanupGuard {
 public:
  explicit BackwardPassCleanupGuard(ContextPtr autogradContext)
      : autogradContext_(std::move(autogradContext)) {}

  ~BackwardPassCleanupGuard() {
    // 获取 DistEngine 的实例并调用其清理反向传播方法，传入自动求导上下文
    DistEngine::getInstance().cleanupBackwardPass(autogradContext_);
  }

 private:
  ContextPtr autogradContext_; // 自动求导上下文的智能指针
};

} // namespace autograd
} // namespace distributed
} // namespace torch
```