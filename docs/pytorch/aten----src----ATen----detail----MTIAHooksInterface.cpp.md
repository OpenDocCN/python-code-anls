# `.\pytorch\aten\src\ATen\detail\MTIAHooksInterface.cpp`

```py
#include <ATen/detail/MTIAHooksInterface.h>

#include <c10/util/CallOnce.h>

#include <memory>

namespace at {
namespace detail {

// 返回全局的 MTIAHooksInterface 实例
const MTIAHooksInterface& getMTIAHooks() {
  // 使用静态指针保证 mtia_hooks 只被初始化一次
  static std::unique_ptr<MTIAHooksInterface> mtia_hooks = nullptr;
  // 创建一个只执行一次的标志对象
  static c10::once_flag once;
  // 使用 call_once 函数保证以下 lambda 函数只会被调用一次
  c10::call_once(once, [] {
    // 从 MTIAHooksRegistry 中创建名为 "MTIAHooks" 的实例
    mtia_hooks = MTIAHooksRegistry()->Create("MTIAHooks", MTIAHooksArgs{});
    // 如果未成功创建实例，则创建一个默认的 MTIAHooksInterface 实例
    if (!mtia_hooks) {
      mtia_hooks = std::make_unique<MTIAHooksInterface>();
    }
  });
  // 返回全局的 MTIAHooksInterface 实例引用
  return *mtia_hooks;
}

// 检查是否已经构建了 MTIAHooks
bool isMTIAHooksBuilt() {
  // 使用 MTIAHooksRegistry 检查是否存在名为 "MTIAHooks" 的实例
  return MTIAHooksRegistry()->Has("MTIAHooks");
}

} // namespace detail

// 定义 MTIAHooksRegistry 的全局注册表
C10_DEFINE_REGISTRY(MTIAHooksRegistry, MTIAHooksInterface, MTIAHooksArgs)

} // namespace at
```