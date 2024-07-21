# `.\pytorch\torch\csrc\itt_wrapper.cpp`

```py
#include <ittnotify.h>
#include <torch/csrc/itt_wrapper.h>
#include <torch/csrc/profiler/stubs/base.h>

namespace torch::profiler {
// 创建一个 ITT 域对象，并命名为 "PyTorch"
__itt_domain* _itt_domain = __itt_domain_create("PyTorch");

// 检查 ITT 是否可用的函数
bool itt_is_available() {
  // 调用 ITT 框架提供的实现来检查是否启用了 ITT
  return torch::profiler::impl::ittStubs()->enabled();
}

// 开始一个 ITT 范围，用给定的消息
void itt_range_push(const char* msg) {
  // 创建一个 ITT 字符串句柄，用于表示给定的消息
  __itt_string_handle* hsMsg = __itt_string_handle_create(msg);
  // 开始一个 ITT 任务，在指定的 ITT 域中，使用给定的消息句柄
  __itt_task_begin(_itt_domain, __itt_null, __itt_null, hsMsg);
}

// 结束当前的 ITT 范围
void itt_range_pop() {
  // 结束当前的 ITT 任务
  __itt_task_end(_itt_domain);
}

// 在 ITT 中标记一个事件，用给定的消息
void itt_mark(const char* msg) {
  // 创建一个 ITT 字符串句柄，用于表示给定的消息
  __itt_string_handle* hsMsg = __itt_string_handle_create(msg);
  // 开始一个 ITT 任务，在指定的 ITT 域中，使用给定的消息句柄
  __itt_task_begin(_itt_domain, __itt_null, __itt_null, hsMsg);
  // 结束当前的 ITT 任务
  __itt_task_end(_itt_domain);
}
} // namespace torch::profiler
```