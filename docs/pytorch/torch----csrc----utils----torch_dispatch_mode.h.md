# `.\pytorch\torch\csrc\utils\torch_dispatch_mode.h`

```
#pragma once
// 引入 TorchDispatchModeTLS 头文件
#include <c10/core/impl/TorchDispatchModeTLS.h>

// 定义命名空间 torch::torch_dispatch_mode
namespace torch::torch_dispatch_mode {

// 定义结构体 StashTorchDispatchModeGuard
struct StashTorchDispatchModeGuard {
 public:
  // 构造函数
  StashTorchDispatchModeGuard() {
    // 如果存在任何模式设置，跳过基础设施模式
    if (c10::impl::TorchDispatchModeTLS::any_modes_set(
            /*skip_infra_modes=*/true)) {
      // 保存当前堆栈顶部的模式，并将其弹出堆栈
      saved_mode_ = c10::impl::TorchDispatchModeTLS::pop_stack();
    } else {
      // 弹出最高基础设施模式并保存
      auto mode_and_key =
          c10::impl::TorchDispatchModeTLS::pop_highest_infra_mode();
      saved_mode_ = std::move(std::get<0>(mode_and_key));
      saved_mode_key_ = std::get<1>(mode_and_key);
    }
  }

  // 析构函数
  ~StashTorchDispatchModeGuard() {
    // 如果保存的模式键不为空
    if (saved_mode_key_ != c10::nullopt) {
      // 将保存的模式和键设置回 TLS 中
      c10::impl::TorchDispatchModeTLS::set_mode(
          saved_mode_, saved_mode_key_.value());
    } else {
      // 将非基础设施模式推送回堆栈
      c10::impl::TorchDispatchModeTLS::push_non_infra_mode_onto_stack(
          std::move(saved_mode_));
    }
  }

  // 返回当前保存的模式
  const std::shared_ptr<c10::impl::PyObject_TorchDispatchMode>& get_cur_mode() {
    return saved_mode_;
  }

 private:
  // 成员变量，保存当前模式和模式键
  std::shared_ptr<c10::impl::PyObject_TorchDispatchMode> saved_mode_;
  std::optional<c10::impl::TorchDispatchModeKey> saved_mode_key_;
};

// 定义结构体 StashTorchDispatchStackGuard
struct StashTorchDispatchStackGuard {
 public:
  // 构造函数
  StashTorchDispatchStackGuard() {
    // 保存当前 TLS 状态并设置为之前保存的状态
    auto old = c10::impl::TorchDispatchModeTLS::get_state();
    c10::impl::TorchDispatchModeTLS::set_state(std::move(saved_state_));
    saved_state_ = std::move(old);
  }

  // 析构函数
  ~StashTorchDispatchStackGuard() {
    // 恢复之前保存的 TLS 状态
    c10::impl::TorchDispatchModeTLS::set_state(std::move(saved_state_));
  }

 private:
  // 成员变量，保存当前 TLS 状态
  c10::impl::TorchDispatchModeTLS saved_state_;
};

} // namespace torch::torch_dispatch_mode
// 命名空间结束
```