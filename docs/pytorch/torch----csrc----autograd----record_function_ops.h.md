# `.\pytorch\torch\csrc\autograd\record_function_ops.h`

```
#pragma once
// 预处理指令，确保此头文件仅被包含一次

#include <ATen/record_function.h>
// 包含 ATen 库中的 record_function.h 头文件

#include <c10/util/Optional.h>
// 包含 c10 库中的 Optional.h 头文件

#include <torch/custom_class.h>
// 包含 torch 库中的 custom_class.h 头文件

namespace torch::autograd::profiler {
// 定义命名空间 torch::autograd::profiler

struct PythonRecordFunction : public torch::CustomClassHolder {
  // 定义结构体 PythonRecordFunction，继承自 torch::CustomClassHolder

  at::RecordFunction record;
  // 成员变量，用于记录函数调用信息的 RecordFunction 对象

  explicit PythonRecordFunction(
      at::RecordScope scope = at::RecordScope::FUNCTION)
      : record(scope) {}
  // 构造函数，创建一个 PythonRecordFunction 对象，并初始化其 record 成员

};

// 创建一个新的性能分析范围，使用 RecordFunction，并调用其起始回调函数。
TORCH_API c10::intrusive_ptr<PythonRecordFunction> record_function_enter_new(
    const std::string& name,
    const std::optional<std::string>& args = c10::nullopt);
// 函数声明，用于创建一个新的性能分析范围，返回一个指向 PythonRecordFunction 对象的智能指针

// 在未来任务完成时，安排运行 RecordFunction 的结束回调。
TORCH_API c10::intrusive_ptr<c10::ivalue::Future> _call_end_callbacks_on_fut_new(
    const c10::intrusive_ptr<PythonRecordFunction>& record,
    const c10::intrusive_ptr<c10::ivalue::Future>& fut);
// 函数声明，用于在未来任务完成时，调度运行 RecordFunction 的结束回调

} // namespace torch::autograd::profiler
// 命名空间结束
```