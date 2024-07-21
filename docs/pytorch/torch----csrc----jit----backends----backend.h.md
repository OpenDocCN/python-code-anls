# `.\pytorch\torch\csrc\jit\backends\backend.h`

```py
#pragma once

#include <ATen/core/builtin_function.h>
#include <ATen/core/stack.h>
#include <torch/csrc/jit/backends/backend_interface.h>
#include <torch/custom_class.h>

namespace torch {
namespace jit {
namespace {
// NOLINTNEXTLINE(clang-diagnostic-unneeded-internal-declaration)
// 返回一个函数模式对象，用于描述is_available函数的参数和返回值
inline c10::FunctionSchema getIsAvailableSchema() {
  c10::Argument self("self", c10::AnyType::get());
  c10::Argument available("available", c10::BoolType::get());
  c10::FunctionSchema preprocessor_schema(
      "is_available",
      /*overload_name=*/"",
      /*arguments=*/{self},
      /*returns=*/{available});
  return preprocessor_schema;
}

constexpr static auto kBackendsNamespace = "__backends__";

// NOLINTNEXTLINE(clang-diagnostic-unneeded-internal-declaration)
// 返回一个函数模式对象，用于描述compile函数的参数和返回值
inline c10::FunctionSchema getCompileSchema() {
  c10::Argument self("self", c10::AnyType::get());
  c10::Argument mod("processed", c10::AnyType::get());
  auto any_dict_ty =
      c10::DictType::create(c10::StringType::get(), c10::AnyType::get());
  c10::Argument method_compile_spec("method_compile_spec", any_dict_ty);
  c10::Argument handles("handles", any_dict_ty);

  c10::FunctionSchema compile_schema(
      "compile",
      /*overload_name=*/"",
      /*arguments=*/{self, mod, method_compile_spec},
      /*returns=*/{handles});
  return compile_schema;
}

// NOLINTNEXTLINE(clang-diagnostic-unneeded-internal-declaration)
// 返回一个函数模式对象，用于描述execute函数的参数和返回值
inline c10::FunctionSchema getExecuteSchema() {
  auto any_list_ty = c10::ListType::create(c10::AnyType::get());
  c10::Argument self("self", c10::AnyType::get());
  c10::Argument handle("handle", c10::AnyType::get());
  c10::Argument input("input", any_list_ty);
  c10::Argument output("output", any_list_ty);
  return c10::FunctionSchema(
      "execute",
      /*overload_name=*/"",
      /*arguments=*/{self, handle, input},
      /*returns=*/{output});
}

// 返回一个函数对象，用于执行is_available函数的操作
template <typename TBackendInterface>
std::function<void(Stack&)> getIsAvailableFunc() {
  return [](Stack& stack) {
    auto self = pop(stack).toCustomClass<TBackendInterface>();
    auto ret = self->is_available();
    push(stack, ret);
  };
}

// 返回一个函数对象，用于执行compile函数的操作
template <typename TBackendInterface>
std::function<void(Stack&)> getCompileFunc() {
  return [](Stack& stack) {
    auto method_compile_spec = pop(stack).toGenericDict();
    auto processed = pop(stack);
    auto self = pop(stack).toCustomClass<TBackendInterface>();
    auto ret = self->compile(processed, method_compile_spec);
    push(stack, ret);
  };
}

// 返回一个函数对象，用于执行execute函数的操作
template <typename TBackendInterface>
std::function<void(Stack&)> getExecuteFunc() {
  return [](Stack& stack) {
    auto args = pop(stack);
    auto handle = pop(stack);
    auto self = pop(stack);
    auto backend = self.toCustomClass<TBackendInterface>();
    auto res = backend->execute(handle, args.toList());
    push(stack, res);
  };
}
} // namespace

// Static registration API for backends.
template <class TBackendInterface>
class backend {
  // 静态断言，确保 TBackendInterface 继承自 PyTorchBackendInterface
  static_assert(
      std::is_base_of<PyTorchBackendInterface, TBackendInterface>::value,
      "torch::jit::backend<T> requires T to inherit from PyTorchBackendInterface");
  // 后端名称
  std::string backend_name_;

 public:
  // 使用给定的名称注册一个新的后端，并提供指定的预处理函数
  backend(const std::string& name) : backend_name_(name) {
    // 在 kBackendsNamespace 命名空间中注册 TBackendInterface 类
    static auto cls = torch::class_<TBackendInterface>(kBackendsNamespace, name)
                          .def(torch::init<>()) // 定义默认构造函数
                          ._def_unboxed(
                              "is_available",
                              getIsAvailableFunc<TBackendInterface>(), // 注册 is_available 函数
                              getIsAvailableSchema()) // 获取 is_available 函数的模式
                          ._def_unboxed(
                              "compile",
                              getCompileFunc<TBackendInterface>(), // 注册 compile 函数
                              getCompileSchema()) // 获取 compile 函数的模式
                          ._def_unboxed(
                              "execute",
                              getExecuteFunc<TBackendInterface>(), // 注册 execute 函数
                              getExecuteSchema()); // 获取 execute 函数的模式
  }
};

} // namespace jit
} // namespace torch
```