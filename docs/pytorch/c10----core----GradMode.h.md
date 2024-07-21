# `.\pytorch\c10\core\GradMode.h`

```
#pragma once

#include <c10/core/AutogradState.h>
#include <c10/macros/Export.h>

namespace c10 {

// 定义了 GradMode 结构体，包含了控制梯度计算开关的两个静态方法
struct C10_API GradMode {
  // 返回当前梯度计算是否启用的状态
  static bool is_enabled();
  // 设置是否启用梯度计算
  static void set_enabled(bool enabled);
};

// AutoGradMode 类是一个 RAII 风格的对象，用于在构造时启用或禁用梯度计算模式，在析构时恢复原有模式
struct C10_API AutoGradMode {
  // 构造函数，根据传入的 enabled 参数设置梯度计算模式，并保存当前模式
  AutoGradMode(bool enabled) : prev_mode(GradMode::is_enabled()) {
    GradMode::set_enabled(enabled);
  }
  // 析构函数，恢复之前保存的梯度计算模式
  ~AutoGradMode() {
    GradMode::set_enabled(prev_mode);
  }
  // 保存先前的梯度计算模式状态
  bool prev_mode;
};

// NoGradGuard 类是 AutoGradMode 的派生类，用于禁用梯度计算的 RAII 风格对象
struct C10_API NoGradGuard : public AutoGradMode {
  // 构造函数，调用基类的构造函数来禁用梯度计算
  NoGradGuard() : AutoGradMode(/*enabled=*/false) {}
};

// AutoFwGradMode 类是一个 RAII 风格的对象，用于在构造时设置前向梯度计算模式，并在析构时恢复原有模式
struct C10_API AutoFwGradMode {
  // 构造函数，根据传入的 enabled 参数设置前向梯度计算模式，并保存当前模式
  AutoFwGradMode(bool enabled)
      : prev_mode(AutogradState::get_tls_state().get_fw_grad_mode()) {
    AutogradState::get_tls_state().set_fw_grad_mode(enabled);
  }
  // 析构函数，恢复之前保存的前向梯度计算模式
  ~AutoFwGradMode() {
    AutogradState::get_tls_state().set_fw_grad_mode(prev_mode);
  }
  // 保存先前的前向梯度计算模式状态
  bool prev_mode;
};

} // namespace c10
```