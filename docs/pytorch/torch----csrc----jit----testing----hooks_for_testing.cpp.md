# `.\pytorch\torch\csrc\jit\testing\hooks_for_testing.cpp`

```py
// 引入 Torch 的 JIT 测试钩子函数头文件
#include <torch/csrc/jit/testing/hooks_for_testing.h>

// 引入 Torch 的 JIT 模块 API 头文件
#include <torch/csrc/jit/api/module.h>

// Torch JIT 命名空间
namespace torch {
namespace jit {

// 静态变量，用于保存模块发射后的回调函数
static ModuleHook emit_module_callback;

// 模块发射完成后的回调函数，接受一个 Module 对象作为参数
void didFinishEmitModule(Module module) {
  // 如果存在模块发射后的回调函数，则调用之，并传递当前模块对象作为参数
  if (emit_module_callback) {
    emit_module_callback(module);
  }
}

// 静态变量，用于保存函数发射后的回调函数
static FunctionHook emit_function_callback;

// 函数发射完成后的回调函数，接受一个 StrongFunctionPtr 对象作为参数
void didFinishEmitFunction(StrongFunctionPtr fn) {
  // 如果存在函数发射后的回调函数，则调用之，并传递当前函数对象作为参数
  if (emit_function_callback) {
    emit_function_callback(fn);
  }
}

// 设置模块和函数发射的回调函数
void setEmitHooks(ModuleHook for_mod, FunctionHook for_fn) {
  // 将传入的模块和函数发射回调函数移动赋值给静态变量
  emit_module_callback = std::move(for_mod);
  emit_function_callback = std::move(for_fn);
}

// 获取当前设置的模块和函数发射的回调函数
std::pair<ModuleHook, FunctionHook> getEmitHooks() {
  // 返回当前模块和函数发射的回调函数的 pair 对象
  return std::make_pair(emit_module_callback, emit_function_callback);
}

} // namespace jit
} // namespace torch
```