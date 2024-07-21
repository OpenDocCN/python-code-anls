# `.\pytorch\c10\core\impl\PythonDispatcherTLS.cpp`

```py
#include <c10/core/DispatchKey.h>
#include <c10/core/impl/LocalDispatchKeySet.h>
#include <c10/core/impl/PythonDispatcherTLS.h>

namespace c10::impl {

// 定义线程局部变量，用于存储 Python 解释器状态的指针
thread_local PyInterpreter* pythonDispatcherState;

// 设置 Python 解释器状态的方法
void PythonDispatcherTLS::set_state(PyInterpreter* state) {
  // 如果状态非空，则将 Python 调度器的 DispatchKey 设置为包含 PythonDispatcher
  if (state) {
    c10::impl::tls_set_dispatch_key_included(
        DispatchKey::PythonDispatcher, true);
  } else {
    // 否则重置 Python 解释器状态
    PythonDispatcherTLS::reset_state();
  }
  // 更新线程局部变量为新的解释器状态
  pythonDispatcherState = state;
}

// 获取当前线程的 Python 解释器状态
PyInterpreter* PythonDispatcherTLS::get_state() {
  return pythonDispatcherState;
}

// 重置当前线程的 Python 解释器状态
void PythonDispatcherTLS::reset_state() {
  pythonDispatcherState = nullptr;
  // 将 Python 调度器的 DispatchKey 设置为不包含 PythonDispatcher
  c10::impl::tls_set_dispatch_key_included(
      DispatchKey::PythonDispatcher, false);
}

} // namespace c10::impl
```