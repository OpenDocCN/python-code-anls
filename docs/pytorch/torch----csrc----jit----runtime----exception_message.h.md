# `.\pytorch\torch\csrc\jit\runtime\exception_message.h`

```py
#pragma once
// 引入 C10 库中的 Exception.h 头文件，用于处理异常
#include <c10/util/Exception.h>
// 引入标准异常处理库
#include <stdexcept>

// 定义 torch::jit 命名空间
namespace torch::jit {

// 定义 ExceptionMessage 结构体，用于封装异常信息
struct ExceptionMessage {
  // 构造函数，接收标准异常对象作为参数
  ExceptionMessage(const std::exception& e) : e_(e) {}

 private:
  // 存储对异常对象的引用
  const std::exception& e_;

  // 声明友元函数，用于重载输出流操作符
  friend std::ostream& operator<<(
      std::ostream& out,
      const ExceptionMessage& msg);
};

// 定义重载的输出流操作符 <<，用于打印异常信息
inline std::ostream& operator<<(
    std::ostream& out,
    const ExceptionMessage& msg) {
  // 尝试将异常对象转换为 c10::Error 类型
  auto c10_error = dynamic_cast<const c10::Error*>(&msg.e_);
  // 如果转换成功，使用 c10::Error 提供的特定打印方法
  if (c10_error) {
    out << c10_error->what_without_backtrace();
  } else {
    // 否则，使用标准异常对象提供的打印方法
    out << msg.e_.what();
  }
  return out;
}

} // namespace torch::jit
```