# `.\pytorch\c10\util\ScopeExit.h`

```
#pragma once

#include <type_traits> // 包含类型特性的标准库头文件
#include <utility> // 包含实用工具的标准库头文件

namespace c10 {

/**
 * Mostly copied from https://llvm.org/doxygen/ScopeExit_8h_source.html
 */
template <typename Callable>
class scope_exit {
  Callable ExitFunction; // 存储可调用对象的成员变量
  bool Engaged = true; // 标记是否仍然有效，初始为true，移动或释放后置为false

 public:
  template <typename Fp>
  // NOLINTNEXTLINE(bugprone-forwarding-reference-overload)
  explicit scope_exit(Fp&& F) : ExitFunction(std::forward<Fp>(F)) {} // 构造函数，接受可调用对象，并移动或复制到ExitFunction

  scope_exit(scope_exit&& Rhs) noexcept
      : ExitFunction(std::move(Rhs.ExitFunction)), Engaged(Rhs.Engaged) { // 移动构造函数，从右值引用中移动ExitFunction和Engaged
    Rhs.release(); // 调用右值引用对象的release()方法，释放其标记
  }
  scope_exit(const scope_exit&) = delete; // 禁用拷贝构造函数
  scope_exit& operator=(scope_exit&&) = delete; // 禁用移动赋值运算符
  scope_exit& operator=(const scope_exit&) = delete; // 禁用拷贝赋值运算符

  void release() {
    Engaged = false; // 将Engaged标记为false，表示对象已释放
  }

  ~scope_exit() {
    if (Engaged) {
      ExitFunction(); // 在对象销毁时如果仍然有效，则调用ExitFunction执行相应操作
    }
  }
};

// Keeps the callable object that is passed in, and execute it at the
// destruction of the returned object (usually at the scope exit where the
// returned object is kept).
//
// Interface is specified by p0052r2.
template <typename Callable>
scope_exit<std::decay_t<Callable>> make_scope_exit(Callable&& F) {
  return scope_exit<std::decay_t<Callable>>(std::forward<Callable>(F)); // 创建并返回scope_exit对象，存储传入的可调用对象
}

} // namespace c10
```