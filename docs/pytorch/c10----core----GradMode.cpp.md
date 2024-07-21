# `.\pytorch\c10\core\GradMode.cpp`

```
# 包含 C10 核心库中的 GradMode 头文件
#include <c10/core/GradMode.h>

# 进入 c10 命名空间
namespace c10 {

# 定义 GradMode 类的成员函数 is_enabled()
bool GradMode::is_enabled() {
  # 调用 AutogradState::get_tls_state() 获取线程本地状态，再获取梯度模式的状态
  return AutogradState::get_tls_state().get_grad_mode();
}

# 定义 GradMode 类的成员函数 set_enabled()
void GradMode::set_enabled(bool enabled) {
  # 调用 AutogradState::get_tls_state() 获取线程本地状态，设置梯度模式状态为 enabled
  AutogradState::get_tls_state().set_grad_mode(enabled);
}

} // 结束 c10 命名空间
```