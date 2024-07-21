# `.\pytorch\torch\csrc\jit\passes\onednn_graph_fuser.h`

```py
#pragma once

#include <torch/csrc/jit/ir/ir.h> // 引入 Torch 的 IR 类相关头文件
#include <torch/csrc/jit/passes/pass_manager.h> // 引入 Torch 的 Pass 管理器相关头文件

#include <ATen/Config.h> // 引入 ATen 库的配置头文件

namespace torch {
namespace jit {
namespace fuser {
namespace onednn {

static std::atomic<bool> onednn_enabled{true}; // 静态原子布尔变量，表示是否启用 oneDNN 加速，默认为 true

static std::atomic<bool>& getLlgaEnabled() { // 返回静态原子布尔变量 onednn_enabled 的引用
  return onednn_enabled;
}

TORCH_API void fuseGraph(std::shared_ptr<Graph>& g); // 声明一个 Torch API 函数，用于融合图形

} // namespace onednn
} // namespace fuser

struct C10_EXPORT RegisterLlgaFuseGraph // 定义一个结构体 RegisterLlgaFuseGraph，导出到 C10
    : public PassManager<RegisterLlgaFuseGraph> { // 继承 PassManager<RegisterLlgaFuseGraph> 类模板
  static bool setEnabled(bool enabled) { // 设置是否启用融合图形
    TORCH_CHECK(
        AT_MKLDNN_ENABLED(), // 断言检查是否启用了 MKLDNN 构建
        "Running oneDNN Graph fuser is only supported with MKLDNN builds."); // 如果未启用 MKLDNN 构建，则报错
    bool oldState = fuser::onednn::getLlgaEnabled(); // 获取旧的 oneDNN 加速状态
    fuser::onednn::getLlgaEnabled() = enabled; // 设置新的 oneDNN 加速状态
    if (enabled) {
      registerPass(fuser::onednn::fuseGraph); // 如果启用了加速，则注册融合图形的 Pass
    } else {
      clearPass(); // 如果禁用加速，则清除 Pass
    }
    return oldState; // 返回旧的加速状态
  }

  static bool isEnabled() { // 检查是否启用了 oneDNN 加速
    return fuser::onednn::getLlgaEnabled();
  }

  // override PassManager::registerPass to register pre-pass
  static bool registerPass(GraphPass p) { // 注册预处理 Pass
    if (!isRegistered()) { // 如果未注册
      passID(registerPrePass(std::move(p)), true); // 注册预处理 Pass，并设置 passID
      isRegistered(true); // 标记为已注册
      return false; // 返回 false
    }
    return true; // 如果已注册，则返回 true
  }

  // override PassManager::clearPass to clear pre-pass
  static void clearPass() { // 清除预处理 Pass
    if (isRegistered()) { // 如果已注册
      clearPrePass(passID()); // 清除指定的预处理 Pass
      isRegistered(true); // 标记为已注册
    }
  }
};

} // namespace jit
} // namespace torch
```