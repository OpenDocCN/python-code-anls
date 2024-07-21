# `.\pytorch\c10\util\DeadlockDetection.h`

```
// 当前文件只会被包含一次，以避免重复定义
#pragma once

// 包含导出符号的定义，用于宏 C10_API
#include <c10/macros/Export.h>

// 包含异常处理工具类的定义
#include <c10/util/Exception.h>

/// This file provides some simple utilities for detecting common deadlocks in
/// PyTorch.  For now, we focus exclusively on detecting Python GIL deadlocks,
/// as the GIL is a wide ranging lock that is taken out in many situations.
/// The basic strategy is before performing an operation that may block, you
/// can use TORCH_ASSERT_NO_GIL_WITHOUT_PYTHON_DEP() to assert that the GIL is
/// not held.  This macro is to be used in contexts where no static dependency
/// on Python is available (we will handle indirecting a virtual call for you).
///
/// If the GIL is held by a torchdeploy interpreter, we always report false.
/// If you are in a context where Python bindings are available, it's better
/// to directly assert on PyGILState_Check (as it avoids a vcall and also
/// works correctly with torchdeploy.)

// 定义宏 TORCH_ASSERT_NO_GIL_WITHOUT_PYTHON_DEP()，用于在没有静态依赖 Python 的上下文中检查 GIL 是否持有
#define TORCH_ASSERT_NO_GIL_WITHOUT_PYTHON_DEP() \
  TORCH_INTERNAL_ASSERT(                         \
      !c10::impl::check_python_gil(),            \
      "Holding GIL before a blocking operation!  Please release the GIL before blocking, or see https://github.com/pytorch/pytorch/issues/56297 for how to release the GIL for destructors of objects")

// 定义 c10::impl 命名空间
namespace c10::impl {

// 声明检查 Python GIL 的函数
C10_API bool check_python_gil();

// 定义 PythonGILHooks 结构体，用于管理 Python GIL 的钩子函数
struct C10_API PythonGILHooks {
  virtual ~PythonGILHooks() = default;
  // 返回当前是否持有 GIL，如果没有链接到 Python，则始终返回 false
  virtual bool check_python_gil() const = 0;
};

// 设置 PythonGILHooks 对象的全局实例
C10_API void SetPythonGILHooks(PythonGILHooks* factory);

// 通过 PythonGILHooksRegisterer 类注册和注销 PythonGILHooks 的实例
// 不要在 torch deploy 实例中调用此注册器，以免覆盖其他注册
struct C10_API PythonGILHooksRegisterer {
  explicit PythonGILHooksRegisterer(PythonGILHooks* factory) {
    SetPythonGILHooks(factory);
  }
  ~PythonGILHooksRegisterer() {
    SetPythonGILHooks(nullptr);
  }
};

} // namespace c10::impl
```