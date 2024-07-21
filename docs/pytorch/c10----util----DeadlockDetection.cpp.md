# `.\pytorch\c10\util\DeadlockDetection.cpp`

```py
#include <c10/util/DeadlockDetection.h>  // 包含头文件 c10/util/DeadlockDetection.h

#include <cstdlib>  // 包含标准库头文件 cstdlib

namespace c10::impl {

namespace {
PythonGILHooks* python_gil_hooks = nullptr;  // 声明静态指针变量 python_gil_hooks，初始为 nullptr

bool disable_detection() {
  // 检查环境变量 TORCH_DISABLE_DEADLOCK_DETECTION 是否设置，返回其存在与否的布尔值
  return std::getenv("TORCH_DISABLE_DEADLOCK_DETECTION") != nullptr;
}
} // namespace

bool check_python_gil() {
  if (!python_gil_hooks) {  // 如果 python_gil_hooks 为 nullptr
    return false;  // 返回 false
  }
  // 调用 PythonGILHooks 对象的 check_python_gil() 方法，返回其结果
  return python_gil_hooks->check_python_gil();
}

void SetPythonGILHooks(PythonGILHooks* hooks) {
  if (disable_detection()) {  // 如果检测到需要禁用死锁检测
    return;  // 直接返回，不执行后续操作
  }
  // 断言确保 hooks 为 nullptr 或者 python_gil_hooks 为 nullptr
  TORCH_INTERNAL_ASSERT(!hooks || !python_gil_hooks);
  // 将 hooks 赋值给 python_gil_hooks
  python_gil_hooks = hooks;
}

} // namespace c10::impl
```