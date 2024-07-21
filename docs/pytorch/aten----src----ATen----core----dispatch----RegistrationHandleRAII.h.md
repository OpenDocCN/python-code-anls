# `.\pytorch\aten\src\ATen\core\dispatch\RegistrationHandleRAII.h`

```
#pragma once
// 预处理指令，确保本文件只被编译一次

#include <functional>
// 引入标准库中的 functional 头文件，用于支持函数对象

namespace c10 {

class RegistrationHandleRAII final {
// 定义名为 RegistrationHandleRAII 的类，使用 final 禁止派生

public:
  explicit RegistrationHandleRAII(std::function<void()> onDestruction)
      : onDestruction_(std::move(onDestruction)) {}
  // 构造函数，接受一个 std::function<void()> 对象作为参数，初始化 onDestruction_ 成员

  ~RegistrationHandleRAII() {
    if (onDestruction_) {
      onDestruction_();
    }
  }
  // 析构函数，销毁对象时调用，如果 onDestruction_ 不为空，则调用其存储的函数对象

  RegistrationHandleRAII(const RegistrationHandleRAII&) = delete;
  // 删除复制构造函数，禁止对象的复制构造

  RegistrationHandleRAII& operator=(const RegistrationHandleRAII&) = delete;
  // 删除复制赋值运算符，禁止对象的复制赋值

  RegistrationHandleRAII(RegistrationHandleRAII&& rhs) noexcept
      : onDestruction_(std::move(rhs.onDestruction_)) {
    rhs.onDestruction_ = nullptr;
  }
  // 移动构造函数，接受一个右值引用对象作为参数，使用移动语义初始化 onDestruction_ 成员，并置右值对象的成员为 nullptr

  RegistrationHandleRAII& operator=(RegistrationHandleRAII&& rhs) noexcept {
    onDestruction_ = std::move(rhs.onDestruction_);
    rhs.onDestruction_ = nullptr;
    return *this;
  }
  // 移动赋值运算符，接受一个右值引用对象作为参数，使用移动语义赋值 onDestruction_ 成员，并置右值对象的成员为 nullptr

private:
  std::function<void()> onDestruction_;
  // 私有成员变量，存储一个可调用对象（函数对象）的 std::function 类型
};

}
// 命名空间 c10 结束
```