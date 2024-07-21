# `.\pytorch\c10\core\impl\HermeticPyObjectTLS.cpp`

```
#include <c10/core/impl/HermeticPyObjectTLS.h>
// 包含 HermeticPyObjectTLS.h 头文件，用于声明 HermeticPyObjectTLS 类的接口

namespace c10::impl {

// 定义命名空间 c10::impl

thread_local std::atomic<bool> hermeticPyObjectState{false};
// 声明一个线程局部变量 hermeticPyObjectState，类型为 std::atomic<bool>，初始值为 false

std::atomic<bool> HermeticPyObjectTLS::haveState_{false};
// 静态成员变量 haveState_ 的定义，类型为 std::atomic<bool>，初始值为 false

void HermeticPyObjectTLS::set_state(bool state) {
  // 实现 HermeticPyObjectTLS 类的成员函数 set_state，设置 hermeticPyObjectState 的值
  hermeticPyObjectState = state;
}

bool HermeticPyObjectTLS::get_tls_state() {
  // 实现 HermeticPyObjectTLS 类的成员函数 get_tls_state，获取 hermeticPyObjectState 的值并返回
  return hermeticPyObjectState;
}

void HermeticPyObjectTLS::init_state() {
  // 实现 HermeticPyObjectTLS 类的成员函数 init_state，初始化 haveState_ 的状态为 true
  haveState_ = true;
}

} // namespace c10::impl
// 结束命名空间 c10::impl
```