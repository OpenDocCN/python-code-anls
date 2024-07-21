# `.\pytorch\aten\src\ATen\core\PythonOpRegistrationTrampoline.cpp`

```py
// 定义命名空间 `at::impl`，实现 Python 操作的注册桥接器
namespace at::impl {

// 策略是所有 Python 解释器尝试注册为主解释器，但只有一个会成功。
// 只有该解释器能与 C++ 调度器交互。此外，当我们在该解释器上执行逻辑时，
// 我们以隔离的方式进行，永不设置 Tensor 的 pyobj 字段。

// 原子变量，用于存储当前的 Python 解释器指针
std::atomic<c10::impl::PyInterpreter*> PythonOpRegistrationTrampoline::interpreter_{nullptr};

// 获取当前注册的 Python 解释器指针
c10::impl::PyInterpreter* PythonOpRegistrationTrampoline::getInterpreter() {
  return PythonOpRegistrationTrampoline::interpreter_.load();
}

// 注册 Python 解释器
bool PythonOpRegistrationTrampoline::registerInterpreter(c10::impl::PyInterpreter* interp) {
  // 尝试原子性地设置当前 Python 解释器指针，如果已经被设置则失败
  c10::impl::PyInterpreter* expected = nullptr;
  interpreter_.compare_exchange_strong(expected, interp);
  if (expected != nullptr) {
    // 如果不是第一个 Python 解释器，则需要初始化非平凡的隔离 PyObject TLS
    c10::impl::HermeticPyObjectTLS::init_state();
    return false;
  } else {
    return true;
  }
}

} // namespace at::impl
```