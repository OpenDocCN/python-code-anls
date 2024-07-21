# `.\pytorch\torch\csrc\autograd\utils\warnings.cpp`

```py
// 包含 Torch 自动微分工具的警告处理工具头文件
#include <torch/csrc/autograd/utils/warnings.h>

// 定义 torch 命名空间
namespace torch {
// 定义自动微分工具的命名空间
namespace autograd {
// 定义工具命名空间
namespace utils {

// 处理延迟警告的处理器，将警告添加到警告列表中
void DelayWarningHandler::process(const c10::Warning& warning) {
  // 使用互斥锁保护临界区
  std::lock_guard<std::mutex> lock(mutex_);
  // 将警告添加到警告列表中
  warnings_.push_back(warning);
}

// 重放警告，将延迟的警告重新输出
void DelayWarningHandler::replay_warnings() {
  // 使用互斥锁保护临界区
  std::lock_guard<std::mutex> lock(mutex_);
  // 断言当前警告处理器不是 DelayWarningHandler 本身，否则会导致死锁
  TORCH_INTERNAL_ASSERT(
      c10::WarningUtils::get_warning_handler() != this,
      "DelayWarningHandler cannot replay warnings into itself, this will cause a deadlock");
  // 遍历警告列表，输出每个警告
  for (const auto& warning : warnings_) {
    c10::warn(warning);
  }
}

} // namespace utils
} // namespace autograd
} // namespace torch
```