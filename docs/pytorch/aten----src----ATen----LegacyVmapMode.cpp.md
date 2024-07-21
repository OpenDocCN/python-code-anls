# `.\pytorch\aten\src\ATen\LegacyVmapMode.cpp`

```py
#include <ATen/LegacyVmapMode.h>  // 包含 ATen 库中的 LegacyVmapMode 头文件

namespace at::impl {  // 进入 at::impl 命名空间

thread_local int64_t VmapMode_current_vmap_level = 0;  // 定义线程局部变量 VmapMode_current_vmap_level，初始值为0

int64_t VmapMode::current_vmap_level() {  // 定义 VmapMode 类的成员函数 current_vmap_level()
  return VmapMode_current_vmap_level;  // 返回当前的 VmapMode 层级
}

int64_t VmapMode::increment_nesting() {  // 定义 VmapMode 类的成员函数 increment_nesting()
  VmapMode_current_vmap_level++;  // 增加 VmapMode 当前层级计数
  if (VmapMode_current_vmap_level == 1) {  // 如果当前层级为1
    c10::impl::tls_set_dispatch_key_included(DispatchKey::VmapMode, true);  // 设置 DispatchKey::VmapMode 的调度键包含状态为真
  }
  return VmapMode_current_vmap_level;  // 返回增加后的 VmapMode 层级计数
}

int64_t VmapMode::decrement_nesting() {  // 定义 VmapMode 类的成员函数 decrement_nesting()
  VmapMode_current_vmap_level--;  // 减少 VmapMode 当前层级计数
  if (VmapMode_current_vmap_level == 0) {  // 如果当前层级为0
    c10::impl::tls_set_dispatch_key_included(DispatchKey::VmapMode, false);  // 设置 DispatchKey::VmapMode 的调度键包含状态为假
  }
  return VmapMode_current_vmap_level;  // 返回减少后的 VmapMode 层级计数
}

} // namespace at::impl  // 结束 at::impl 命名空间
```