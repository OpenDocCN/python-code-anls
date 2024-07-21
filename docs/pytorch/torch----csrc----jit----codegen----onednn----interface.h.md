# `.\pytorch\torch\csrc\jit\codegen\onednn\interface.h`

```py
#pragma once
#include <ATen/Config.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/passes/pass_manager.h>

// Torch 框架命名空间
namespace torch {
// JIT 模块命名空间
namespace jit {
// 图优化相关命名空间
namespace fuser {
// 用于一DNN的命名空间
namespace onednn {

// 声明一个静态的原子布尔变量 onednn_enabled，并初始化为 false
static std::atomic<bool> onednn_enabled{false};

// 返回 onednn_enabled 的引用
static std::atomic<bool>& getLlgaEnabled() {
  return onednn_enabled;
}

// 导出的函数声明，用于将图进行融合处理
C10_EXPORT void fuseGraph(std::shared_ptr<Graph>& g);

} // namespace onednn
} // namespace fuser

// RegisterLlgaFuseGraph 结构体声明
struct C10_EXPORT RegisterLlgaFuseGraph
    : public PassManager<RegisterLlgaFuseGraph> {
  // 设置是否启用融合图功能的静态函数
  static bool setEnabled(bool enabled) {
    // 检查 MKLDNN 是否已启用
    TORCH_CHECK(
        AT_MKLDNN_ENABLED(),
        "Running oneDNN Graph fuser is only supported with MKLDNN builds.");
    
    // 获取旧的状态
    bool oldState = fuser::onednn::getLlgaEnabled();
    
    // 设置新的状态
    fuser::onednn::getLlgaEnabled() = enabled;
    
    // 如果启用了融合图功能
    if (enabled) {
      // 注册融合图的处理函数
      registerPass(fuser::onednn::fuseGraph);
    } else {
      // 清除注册的处理函数
      clearPass();
    }
    
    return oldState;
  }

  // 返回融合图功能是否启用的静态函数
  static bool isEnabled() {
    return fuser::onednn::getLlgaEnabled();
  }

  // 重写 PassManager::registerPass 函数来注册预处理过程
  static bool registerPass(GraphPass p) {
    if (!isRegistered()) {
      // 注册预处理过程并返回 passID
      passID(registerPrePass(std::move(p)), true);
      // 设置已注册标志为 true
      isRegistered(true);
      return false;
    }
    return true;
  }

  // 重写 PassManager::clearPass 函数来清除预处理过程
  static void clearPass() {
    if (isRegistered()) {
      // 清除预处理过程
      clearPrePass(passID());
      // 设置已注册标志为 true
      isRegistered(true);
    }
  }
};

} // namespace jit
} // namespace torch
```