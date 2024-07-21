# `.\pytorch\torch\csrc\utils\disable_torch_function.h`

```
#pragma once
// 预处理指令，确保本文件只被编译一次

#include <c10/core/DispatchKey.h>
// 包含 c10 核心库中的 DispatchKey 头文件
#include <c10/core/impl/LocalDispatchKeySet.h>
// 包含 c10 核心库中的 LocalDispatchKeySet 实现头文件
#include <torch/csrc/python_headers.h>
// 包含 Torch Python 组件的头文件

namespace torch {
// 进入 torch 命名空间

// 有时我们不希望子类无限递归，
// 或者希望实现旧行为。

// 这是一个内部实用程序，不对用户公开。
bool torch_function_enabled();
// 声明一个函数，返回布尔值，用于检查是否启用了 torch function
PyObject* disabled_torch_function_impl();
// 声明一个函数，返回 PyObject 指针，用于实现禁用 torch function
PyObject* disabled_torch_dispatch_impl();
// 声明一个函数，返回 PyObject 指针，用于实现禁用 torch dispatch
void set_disabled_torch_function_impl(PyObject* value);
// 声明一个函数，接受 PyObject 参数，用于设置禁用 torch function 的实现
void set_disabled_torch_dispatch_impl(PyObject* value);
// 声明一个函数，接受 PyObject 参数，用于设置禁用 torch dispatch 的实现

// 如果你尝试收集重载参数，将 ignore_mode 设置为 true；
// 在此处使用 mode 将不正确地导致你将所有对象添加到重载列表中，
// 即使它们实际上没有 __torch_function__
bool check_has_torch_function(PyObject* obj, bool ignore_mode = false);
// 声明一个函数，接受 PyObject 和 bool 参数，用于检查对象是否具有 torch function

struct DisableTorchDispatch {
  // 定义 DisableTorchDispatch 结构体

  DisableTorchDispatch()
      : guard_(c10::DispatchKeySet(
            {c10::DispatchKey::Python, c10::DispatchKey::PreDispatch})),
        guard_tls_snapshot_(c10::DispatchKey::PythonTLSSnapshot) {}
  // 构造函数初始化 guard_ 和 guard_tls_snapshot_

  c10::impl::ExcludeDispatchKeyGuard guard_;
  // 使用 ExcludeDispatchKeyGuard 对象 guard_

  c10::impl::ExcludeDispatchKeyGuard guard_tls_snapshot_;
  // 使用 ExcludeDispatchKeyGuard 对象 guard_tls_snapshot_
};

} // namespace torch
// 结束 torch 命名空间

PyObject* THPModule_isEnabledTorchFunction(PyObject* self, PyObject* unused);
// 声明一个函数，返回 PyObject 指针，用于检查是否启用了 Torch function
PyObject* THPModule_DisableTorchFunctionType();
// 声明一个函数，返回 PyObject 指针，用于禁用 Torch function 类型
PyObject* THPModule_DisableTorchFunctionSubclassType();
// 声明一个函数，返回 PyObject 指针，用于禁用 Torch function 子类类型
PyObject* THPModule_disable_torch_function(PyObject* self, PyObject* args);
// 声明一个函数，返回 PyObject 指针，用于禁用 Torch function
PyObject* THPModule_disable_torch_dispatch(PyObject* self, PyObject* args);
// 声明一个函数，返回 PyObject 指针，用于禁用 Torch dispatch
PyObject* THPModule_has_torch_function(PyObject*, PyObject* arg);
// 声明一个函数，返回 PyObject 指针，用于检查对象是否具有 Torch function
PyObject* THPModule_has_torch_function_unary(PyObject*, PyObject* obj);
// 声明一个函数，返回 PyObject 指针，用于检查对象是否具有 Torch function（一元）
PyObject* THPModule_has_torch_function_variadic(
    PyObject*,
    PyObject* const* args,
    Py_ssize_t nargs);
// 声明一个函数，返回 PyObject 指针，用于检查对象是否具有 Torch function（多元）
```