# `.\pytorch\aten\src\ATen\detail\IPUHooksInterface.cpp`

```
// 包含 ATen 库中的 IPUHooksInterface.h 文件
#include <ATen/detail/IPUHooksInterface.h>

// 包含 C++ 标准库中的 CallOnce 头文件
#include <c10/util/CallOnce.h>

// 使用 ATen 命名空间
namespace at {
namespace detail {

// 定义一个静态函数，返回一个常量引用类型的 IPUHooksInterface 对象
const IPUHooksInterface& getIPUHooks() {
  // 声明静态变量 hooks，用于保存 IPUHooksInterface 对象的唯一指针
  static std::unique_ptr<IPUHooksInterface> hooks;
  // 声明静态标志 once，确保初始化操作只执行一次
  static c10::once_flag once;
  
  // 使用 c10::call_once 函数确保以下 lambda 表达式只被调用一次
  c10::call_once(once, [] {
    // 尝试从 IPUHooksRegistry 注册表中创建名为 "IPUHooks" 的 hooks 对象
    hooks = IPUHooksRegistry()->Create("IPUHooks", IPUHooksArgs{});
    // 如果创建失败，则手动构造一个默认的 IPUHooksInterface 对象
    if (!hooks) {
      hooks = std::make_unique<IPUHooksInterface>();
    }
  });
  
  // 返回 hooks 指针所指向的 IPUHooksInterface 对象的引用
  return *hooks;
}

} // namespace detail

// 定义一个名为 IPUHooksRegistry 的注册表，注册类型为 IPUHooksInterface，参数类型为 IPUHooksArgs
C10_DEFINE_REGISTRY(IPUHooksRegistry, IPUHooksInterface, IPUHooksArgs)

} // namespace at
```