# `.\pytorch\torch\csrc\jit\runtime\interpreter\frame.h`

```py
#pragma once
// 预处理指令，确保此头文件只包含一次

#include <atomic>
#include <memory>

#include <torch/csrc/jit/runtime/interpreter/code_impl.h>
#include <torch/csrc/jit/runtime/profiling_record.h>

namespace torch::jit::interpreter {

// 命名空间 torch::jit::interpreter 中定义的结构体 Frame
// 表示捕获函数状态的帧
// （例如 `pc` 和 `base_pointer`）
// 每个 Frame 对应于对 `Frame::function` 的调用
// 该调用尚未返回
// `Frame::function` 的参数位于 [base_pointer + arg_number] 处
struct Frame {
  // 指向 CodeImpl 对象的共享指针，表示当前函数
  std::shared_ptr<CodeImpl> function;
  // 程序计数器，对应当前执行指令的索引
  size_t pc;
  // 标记帧的起始索引
  // base_pointer 被 TAIL_CALL 使用，
  // 用于将当前帧替换为回退图的帧
  size_t base_pointer;

  // 对于每个帧而言是唯一的，跨所有线程的 prim::profile
  std::optional<size_t> id;

  // 与此帧关联的 RecordFunction 对象的唯一指针
  std::unique_ptr<at::RecordFunction> record_function;

  // 帧的符号表
  ShapeSymbolTable symbols2dims;

  // 静态方法，生成唯一的帧 ID
  static size_t genId();
};

} // namespace torch::jit::interpreter
```