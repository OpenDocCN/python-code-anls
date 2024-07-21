# `.\pytorch\aten\src\ATen\detail\MAIAHooksInterface.cpp`

```
# 引入 ATen 库中的 MAIAHooksInterface 头文件
#include <ATen/detail/MAIAHooksInterface.h>

# 引入 C10 工具库中的 CallOnce 和 Registry 头文件
#include <c10/util/CallOnce.h>
#include <c10/util/Registry.h>

# 引入标准库中的常用头文件
#include <cstddef>
#include <memory>

# 定义 ATen 命名空间下的 detail 命名空间
namespace at {
namespace detail {

# 从 getCUDAHooks 中获取更多评论
# 定义函数 getMAIAHooks，返回一个常量引用 MAIAHooksInterface 对象
const MAIAHooksInterface& getMAIAHooks() {
  # 静态局部变量，存储 MAIAHooksInterface 的唯一指针
  static std::unique_ptr<MAIAHooksInterface> maia_hooks;
  # 静态局部变量，控制初始化操作的标志
  static c10::once_flag once;
  # 调用 c10::call_once 函数确保以下 Lambda 表达式只被执行一次
  c10::call_once(once, [] {
    # 通过 MAIAHooksRegistry() 的 Create 方法创建 MAIAHooksInterface 对象
    maia_hooks = MAIAHooksRegistry()->Create("MAIAHooks", {});
    # 若未成功创建对象，则创建一个新的 MAIAHooksInterface 对象
    if (!maia_hooks) {
      maia_hooks = std::make_unique<MAIAHooksInterface>();
    }
  });
  # 返回 MAIAHooksInterface 对象的引用
  return *maia_hooks;
}
} // namespace detail

# 定义 ATen 命名空间
} // namespace at

# 禁止 lint 工具对下一行进行检查（不建议使用非 const 全局变量）
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
# 使用 C10 宏定义注册 MAIAHooksRegistry，注册 MAIAHooksInterface 类型和 MAIAHooksArgs 参数
C10_DEFINE_REGISTRY(MAIAHooksRegistry, MAIAHooksInterface, MAIAHooksArgs)

# 定义 ATen 命名空间
} // namespace at
```