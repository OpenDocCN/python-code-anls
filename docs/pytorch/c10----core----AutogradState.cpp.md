# `.\pytorch\c10\core\AutogradState.cpp`

```py
#include <c10/core/AutogradState.h>

namespace c10 {

namespace {
// 默认情况下，梯度模式和多线程均启用，推断模式禁用，
// 线程局部变量，存储自动求导状态对象
thread_local AutogradState autograd_state_tls = AutogradState(
    /* grad_mode */ true,               // 启用梯度模式
    /* inference_mode */ false,         // 禁用推断模式
    /* fw_grad_mode */ true,            // 启用前向梯度模式
    /* multithreading_enabled */ true); // 启用多线程

} // namespace

// 获取线程局部自动求导状态对象的引用
AutogradState& AutogradState::get_tls_state() {
  return autograd_state_tls;
}

// 设置线程局部自动求导状态对象
void AutogradState::set_tls_state(AutogradState state) {
  autograd_state_tls = state;
}

} // namespace c10
```