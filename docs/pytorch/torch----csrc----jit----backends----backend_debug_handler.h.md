# `.\pytorch\torch\csrc\jit\backends\backend_debug_handler.h`

```
#pragma once
#include <ATen/core/ivalue.h>

#include <torch/csrc/jit/backends/backend_detail.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/scope.h>

#include <atomic>

namespace torch {
namespace jit {

using BackendDebugInfoMapType =
    std::unordered_map<torch::jit::DebugHandleType, DebugInfoTuple>;

/*
 * This class is used to generate debug info map.
 * backend's preprocess will call generate_debug_handles (see
 * backend_detail.cpp), which uses debug_handle_manager to generate debug
 * handles. When lowering process finishes, calling stopRecording will
 * return debug info map from debug_handle_manager
 */
class TORCH_API BackendDebugInfoRecorder {
 public:
  // 默认构造函数
  BackendDebugInfoRecorder() = default;

  // 获取下一个调试句柄的方法，给定节点的调试句柄
  int64_t getNextDebugHandle(const Node* node);

  // 停止记录调试信息并返回调试信息映射
  // 未使用 RAII 的原因在于 stopRecording 可能抛出异常，
  // 使用析构函数处理异常将调用 terminate，而这将取消在更高层次捕获的任何异常。
  BackendDebugInfoMapType stopRecording();

  // 生成调试句柄，返回节点到调试句柄的映射
  NodeToDebugHandle generate_debug_handles(const std::shared_ptr<Graph>& graph);

 private:
  // 静态原子类型变量，用于生成唯一的调试句柄
  static std::atomic<DebugHandleType> unique_debug_handle_;

  // 调试句柄到内联调用堆栈指针的映射
  BackendDebugInfoMapType handles_to_inlined_callstack_ptrs_;
};

} // namespace jit
} // namespace torch
```