# `.\pytorch\aten\src\ATen\detail\XPUHooksInterface.cpp`

```
// 包含 ATen 库中的 XPUHooksInterface 头文件
#include <ATen/detail/XPUHooksInterface.h>

// 包含 C10 库中的 CallOnce 实用工具
#include <c10/util/CallOnce.h>

// 声明 at 命名空间
namespace at {
namespace detail {

// 返回 XPUHooksInterface 的全局对象引用
const XPUHooksInterface& getXPUHooks() {
  // 静态变量，存储 XPUHooksInterface 指针，默认为 nullptr
  static XPUHooksInterface* xpu_hooks = nullptr;
  // 静态变量，用于确保初始化只进行一次
  static c10::once_flag once;
  // 调用 c10::call_once 函数，保证以下 Lambda 表达式只执行一次
  c10::call_once(once, [] {
    // 使用 XPUHooksRegistry 创建 XPUHooksInterface 对象，释放所有权
    xpu_hooks =
        XPUHooksRegistry()->Create("XPUHooks", XPUHooksArgs{}).release();
    // 如果创建失败，则新建一个 XPUHooksInterface 对象
    if (!xpu_hooks) {
      xpu_hooks = new XPUHooksInterface();
    }
  });
  // 返回 XPUHooksInterface 对象的引用
  return *xpu_hooks;
}

} // namespace detail

// 定义 XPUHooksRegistry，注册 XPUHooksInterface 类型对象和参数 XPUHooksArgs
C10_DEFINE_REGISTRY(XPUHooksRegistry, XPUHooksInterface, XPUHooksArgs)

} // namespace at
```