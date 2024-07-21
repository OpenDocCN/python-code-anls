# `.\pytorch\torch\csrc\jit\runtime\custom_operator.h`

```py
#pragma once

# 声明指令，确保本文件只被编译一次，避免重复包含。


#include <ATen/core/op_registration/op_registration.h>
#include <ATen/core/stack.h>
#include <torch/csrc/jit/runtime/operator.h>

# 包含头文件，引入必要的库和模块，用于操作注册和运行时操作的支持。


namespace torch::jit {

# 进入命名空间 `torch::jit`，包含下面的类和函数在这个命名空间内。


/// Registration class for new operators. Effectively calls
/// `torch::jit::registerOperator` for every supplied operator, but allows doing
/// so in the global scope when a `RegisterOperators` object is assigned to a
/// static variable.
/// Note: This is *not* the custom operator API. If you want to register custom
/// operators, take a look at torch::RegisterOperators.
struct TORCH_API RegisterOperators {
  RegisterOperators() = default;

  /// Registers a vector of already created `Operator`s.
  /// The operator element is now optional to filter null ops. It's backward
  /// compatible and works for selective operator registration.
  explicit RegisterOperators(std::vector<std::optional<Operator>> operators) {
    for (std::optional<Operator>& o : operators) {
      if (o) {
        registerOperator(std::move(o.value()));
      }
    }
  }
};

# 定义结构体 `RegisterOperators`，用于注册新的运算符。通过将 `registerOperator` 调用包装在 `RegisterOperators` 对象的静态变量分配中，允许在全局范围内注册操作符。
# 注释中指出，这不是自定义运算符 API。如果需要注册自定义运算符，请查看 `torch::RegisterOperators`。
# 构造函数 `explicit RegisterOperators(std::vector<std::optional<Operator>> operators)` 注册已创建的 `Operator` 向量。其中操作符元素现在是可选的，以便过滤空操作符。这种做法与选择性操作符注册兼容。


} // namespace torch::jit

# 退出 `torch::jit` 命名空间，结束命名空间声明。
```