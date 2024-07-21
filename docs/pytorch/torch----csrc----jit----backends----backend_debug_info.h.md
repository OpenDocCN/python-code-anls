# `.\pytorch\torch\csrc\jit\backends\backend_debug_info.h`

```
#pragma once

#ifndef BUILD_LITE_INTERPRETER
// 如果不是轻量级解释器，包含调试处理的头文件
#include <torch/csrc/jit/backends/backend_debug_handler.h>
#endif
// 包含自定义类的头文件
#include <torch/custom_class.h>

// Torch 的命名空间和 JIT 模块的命名空间
namespace torch {
namespace jit {

// 常量表达式，定义后端工具的命名空间和后端调试信息类名
constexpr static auto kBackendUtilsNamespace = "backendutils";
constexpr static auto kBackendDebugInfoClass = "BackendDebugInfo";

#ifndef BUILD_LITE_INTERPRETER
/*
 * 用于在降低模块中保存调试信息的自定义类，仅用于将这些信息保存以便稍后在降低模块之外进行序列化。
 * 使用模式如下：
 * 1. LoweredModule 在 __backend_debug_info 中声明此类的实例
 * 2. 在序列化期间，使用 __backend_debug_info 获取调试信息
 * 3. LoweredModule.__backend_debug_info 的内容不会在 LoweredModule 本身中被序列化
 */
class TORCH_API PyTorchBackendDebugInfo : public torch::CustomClassHolder {
 public:
  // 默认构造函数
  PyTorchBackendDebugInfo() = default;

  // 返回调试信息映射的可选引用
  std::optional<BackendDebugInfoMapType>& getDebugInfoMap() {
    return debug_info_map_;
  }

  // 设置调试信息映射
  void setDebugInfoMap(BackendDebugInfoMapType&& debug_info_map) {
    debug_info_map_ = std::move(debug_info_map);
  }

 private:
  // 调试信息映射的可选类型
  std::optional<BackendDebugInfoMapType> debug_info_map_;
};

#else

/*
 * Dummy 实例存在的原因：
 * __backend_debug_info 的类型为 BackendDebugInfo，它是由 cpp 类 PyTorchBackendDebugInfo 支持的 torchbind 类。
 * PyTorchBackendDebugInfo 依赖于 ir.h、scope.h、source_range 等。在轻量级解释器中我们不包括这些。
 * 因此在轻量级解释器中，我们不能有有效的 PyTorchBackendDebugInfo 定义。
 * 无论如何，我们在轻量级解释器中不需要 LoweredModule 的 __backend_debug_info 的有效实例，因为如前所述我们不会将此信息作为 LoweredModule 的一部分序列化。
 * 但是，由于 LoweredModule 注册了 __backend_debug_info 属性，我们仍然需要确保 BackendDebugInfo 已经注册到 TorchScript 中。
 * 然而，在此示例中，它不必由 PyTorchBackendDebugInfo 支持，因此我们创建了一个名为 PyTorchBackendDebugInfoDummy 的虚拟实例。
 */
class PyTorchBackendDebugInfoDummy : public torch::CustomClassHolder {
 public:
  // 默认构造函数
  PyTorchBackendDebugInfoDummy() = default;
};
#endif

} // namespace jit
} // namespace torch
```