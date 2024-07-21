# `.\pytorch\c10\core\InferenceMode.cpp`

```py
// 包含 C10 库中的 InferenceMode 头文件
#include <c10/core/InferenceMode.h>

// 进入 c10 命名空间
namespace c10 {

// 不变性：
//   is_enabled() ==
//   !c10::impl::tls_is_dispatch_key_included(DispatchKey::ADInplaceOrView);
// InferenceMode::is_enabled() 在性能关键路径上（TensorImpl 构造函数），
// 所以值得使用单独的 TLS 来跳过 DispatchKeySet 检查。
// 检查推断模式是否启用，通过获取 TLS 状态中的推断模式状态来判断
bool InferenceMode::is_enabled() {
  return AutogradState::get_tls_state().get_inference_mode();
}

} // namespace c10
```