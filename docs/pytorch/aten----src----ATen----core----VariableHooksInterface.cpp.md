# `.\pytorch\aten\src\ATen\core\VariableHooksInterface.cpp`

```py
#include <ATen/core/VariableHooksInterface.h>  // 包含 VariableHooksInterface 的头文件

namespace at::impl {

namespace {
VariableHooksInterface* hooks = nullptr;  // 定义静态变量 hooks，用于存储 VariableHooksInterface 对象的指针
}

void SetVariableHooks(VariableHooksInterface* h) {
  hooks = h;  // 设置全局变量 hooks 的值为传入的参数 h
}

VariableHooksInterface* GetVariableHooks() {
  TORCH_CHECK(hooks, "Support for autograd has not been loaded; have you linked against libtorch.so?")  // 检查 hooks 是否为空，如果为空则输出错误信息
  return hooks;  // 返回全局变量 hooks 的值，即 VariableHooksInterface 对象的指针
}

bool HasVariableHooks() {
  return hooks != nullptr;  // 检查 hooks 是否为空，返回其是否非空的布尔值
}

} // namespace at::impl  // 结束命名空间 at::impl
```