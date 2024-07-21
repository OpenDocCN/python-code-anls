# `.\pytorch\torch\csrc\jit\backends\backend_debug_handler.cpp`

```
// 引入 Torch 后端调试处理器的头文件
#include <torch/csrc/jit/backends/backend_debug_handler.h>

// 引入标准库中的堆栈容器
#include <stack>

// Torch 命名空间开始
namespace torch {
namespace jit {

// 定义静态原子类型变量 unique_debug_handle_，初始化为 0
std::atomic<DebugHandleType> BackendDebugInfoRecorder::unique_debug_handle_{0};

// 获取下一个调试句柄的方法，根据节点信息返回调试句柄
int64_t BackendDebugInfoRecorder::getNextDebugHandle(const Node* node) {
  // 定义内联调用堆栈指针 cs_ptr
  InlinedCallStackPtr cs_ptr;
  // 如果节点的调用堆栈有值，则赋给 cs_ptr
  if (node->callstack().has_value()) {
    cs_ptr = node->callstack().value();
  } else {
    // 否则将空的内联调用堆栈指针赋给 cs_ptr
    cs_ptr = c10::intrusive_ptr<InlinedCallStack>();
  }
  // 获取当前唯一调试句柄值
  DebugHandleType debug_handle = unique_debug_handle_;
  // 获取节点的源代码范围
  const SourceRange& range = node->sourceRange();
  // 将调试句柄、源代码范围、节点类型转换为限定字符串、内联调用堆栈指针的元组存入映射中
  handles_to_inlined_callstack_ptrs_[debug_handle] =
      std::make_tuple(range, node->kind().toQualString(), cs_ptr);
  // 原子方式增加唯一调试句柄的值，使用顺序内存顺序保证
  // 目前不优化性能
  unique_debug_handle_++;
  // 返回调试句柄
  return debug_handle;
}

// 停止记录调试信息的方法，返回存储映射的副本
BackendDebugInfoMapType BackendDebugInfoRecorder::stopRecording() {
  // 注意此处返回的是副本，由于 InlinedCallStackPtr 是侵入式指针，会增加引用计数
  // 不高效，但不用于性能关键路径
  // 可选的替代方案是移动操作，但会破坏原始数据
  return handles_to_inlined_callstack_ptrs_;
}

} // 命名空间 jit 结束
} // 命名空间 torch 结束
```