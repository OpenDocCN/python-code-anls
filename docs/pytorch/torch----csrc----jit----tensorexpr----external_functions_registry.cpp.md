# `.\pytorch\torch\csrc\jit\tensorexpr\external_functions_registry.cpp`

```py
#include <torch/csrc/jit/tensorexpr/external_functions_registry.h>

namespace torch::jit::tensorexpr {

// 定义静态函数，返回一个全局的、线程安全的外部函数注册表
std::unordered_map<std::string, NNCExternalFunction>& getNNCFunctionRegistry() {
  // 声明并定义静态的、只在当前编译单元可见的函数注册表
  static std::unordered_map<std::string, NNCExternalFunction> func_registry_;
  // 返回函数注册表的引用
  return func_registry_;
}

} // namespace torch::jit::tensorexpr
```