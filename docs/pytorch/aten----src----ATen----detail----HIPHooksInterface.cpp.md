# `.\pytorch\aten\src\ATen\detail\HIPHooksInterface.cpp`

```py
// 包含 ATen 库中的 HIPHooksInterface 头文件
#include <ATen/detail/HIPHooksInterface.h>

// 包含 C10 库中的 CallOnce 和 Registry 头文件
#include <c10/util/CallOnce.h>
#include <c10/util/Registry.h>

// 引入标准库中的内存管理工具
#include <memory>

// 声明命名空间 at 和 detail
namespace at {
namespace detail {

// 获取 HIP 钩子函数接口的实例
const HIPHooksInterface& getHIPHooks() {
  // 静态唯一指针，用于存储 HIP 钩子函数接口的实例
  static std::unique_ptr<HIPHooksInterface> hip_hooks;

  // 如果不是在移动设备上编译
  #if !defined C10_MOBILE
    // 静态标志，确保初始化过程只执行一次
    static c10::once_flag once;
    // 使用 c10::call_once 确保 HIP 钩子实例只初始化一次
    c10::call_once(once, [] {
      // 从 HIPHooksRegistry 中创建名为 "HIPHooks" 的实例，使用空的 HIPHooksArgs
      hip_hooks = HIPHooksRegistry()->Create("HIPHooks", HIPHooksArgs{});
      // 如果创建失败，则创建一个默认的 HIPHooksInterface 实例
      if (!hip_hooks) {
        hip_hooks = std::make_unique<HIPHooksInterface>();
      }
    });
  // 如果是在移动设备上，直接创建一个默认的 HIPHooksInterface 实例
  #else
    if (hip_hooks == nullptr) {
      hip_hooks = std::make_unique<HIPHooksInterface>();
    }
  #endif

  // 返回 HIP 钩子函数接口的实例
  return *hip_hooks;
}

} // namespace detail

// 定义 HIPHooksRegistry，注册 HIPHooksInterface 类型和 HIPHooksArgs 参数
C10_DEFINE_REGISTRY(HIPHooksRegistry, HIPHooksInterface, HIPHooksArgs)

} // namespace at
```