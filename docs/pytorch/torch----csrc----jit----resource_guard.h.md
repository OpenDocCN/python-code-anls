# `.\pytorch\torch\csrc\jit\resource_guard.h`

```py
#pragma once
#include <functional>  // 包含函数对象标准库头文件

namespace torch {
namespace jit {

class ResourceGuard {
  std::function<void()> _destructor;  // 存储析构函数的函数对象
  bool _released{false};  // 标记资源是否已释放

 public:
  // 构造函数，接受一个析构函数的函数对象，并进行存储
  ResourceGuard(std::function<void()> destructor)
      : _destructor(std::move(destructor)) {}

  // 析构函数，如果资源未释放，则调用存储的析构函数
  // NOLINTNEXTLINE(bugprone-exception-escape)
  ~ResourceGuard() {
    if (!_released)
      _destructor();
  }

  // 手动释放资源的方法，将 _released 标记设置为 true
  void release() {
    _released = true;
  }
};

} // namespace jit
} // namespace torch
```