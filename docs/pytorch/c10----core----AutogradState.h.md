# `.\pytorch\c10\core\AutogradState.h`

```py
#pragma once

#include <c10/macros/Export.h>

namespace c10 {

// Structure used to pack all the thread local boolean
// flags used by autograd
// 用于打包自动求导中所有线程局部布尔标志的结构体

struct C10_API AutogradState {
  // Returns the thread-local instance of AutogradState
  // 返回 AutogradState 的线程局部实例
  static AutogradState& get_tls_state();

  // Sets the thread-local instance of AutogradState
  // 设置 AutogradState 的线程局部实例
  static void set_tls_state(AutogradState state);

  // Constructor initializing AutogradState with boolean flags
  // 根据给定的布尔标志初始化 AutogradState 的构造函数
  AutogradState(
      bool grad_mode,
      bool inference_mode,
      bool fw_grad_mode,
      bool multithreading_enabled)
      : grad_mode_(grad_mode),
        inference_mode_(inference_mode),
        fw_grad_mode_(fw_grad_mode),
        multithreading_enabled_(multithreading_enabled),
        view_replay_enabled_(false) {}

  // Setter for grad_mode flag
  // 设置 grad_mode 标志的方法
  void set_grad_mode(bool enabled) {
    grad_mode_ = enabled;
  }

  // Setter for fw_grad_mode flag
  // 设置 fw_grad_mode 标志的方法
  void set_fw_grad_mode(bool enabled) {
    fw_grad_mode_ = enabled;
  }

  // Setter for inference_mode flag
  // 设置 inference_mode 标志的方法
  void set_inference_mode(bool enabled) {
    inference_mode_ = enabled;
  }

  // Setter for multithreading_enabled flag
  // 设置 multithreading_enabled 标志的方法
  void set_multithreading_enabled(bool multithreading_enabled) {
    multithreading_enabled_ = multithreading_enabled;
  }

  // Setter for view_replay_enabled flag
  // 设置 view_replay_enabled 标志的方法
  void set_view_replay_enabled(bool view_replay_enabled) {
    view_replay_enabled_ = view_replay_enabled;
  }

  // Getter for grad_mode flag
  // 获取 grad_mode 标志的方法
  bool get_grad_mode() const {
    return grad_mode_;
  }

  // Getter for fw_grad_mode flag
  // 获取 fw_grad_mode 标志的方法
  bool get_fw_grad_mode() const {
    return fw_grad_mode_;
  }

  // Getter for inference_mode flag
  // 获取 inference_mode 标志的方法
  bool get_inference_mode() const {
    return inference_mode_;
  }

  // Getter for multithreading_enabled flag
  // 获取 multithreading_enabled 标志的方法
  bool get_multithreading_enabled() const {
    return multithreading_enabled_;
  }

  // Getter for view_replay_enabled flag
  // 获取 view_replay_enabled 标志的方法
  bool get_view_replay_enabled() const {
    return view_replay_enabled_;
  }

 private:
  bool grad_mode_ : 1;               // 1-bit field for grad_mode flag
  bool inference_mode_ : 1;          // 1-bit field for inference_mode flag
  bool fw_grad_mode_ : 1;            // 1-bit field for fw_grad_mode flag
  bool multithreading_enabled_ : 1;  // 1-bit field for multithreading_enabled flag
  bool view_replay_enabled_ : 1;     // 1-bit field for view_replay_enabled flag
};

} // namespace c10
```