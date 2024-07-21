# `.\pytorch\torch\csrc\jit\runtime\jit_exception.cpp`

```
// 包含 torch/csrc/jit/runtime/jit_exception.h 头文件，其中定义了 JITException 类
#include <torch/csrc/jit/runtime/jit_exception.h>

// 声明 torch::jit 命名空间
namespace torch::jit {

// 声明静态线程局部变量，用于保存原始异常消息和 Python 类名
static thread_local std::string caughtOriginalMsg = "";
static thread_local std::string caughtPythonClassName = "";

// JITException 类的构造函数定义
JITException::JITException(
    const std::string& msg,  // 异常消息字符串的引用
    std::optional<std::string> python_class_name,  // 可选的 Python 类名字符串的引用
    std::optional<std::string> original_msg)  // 可选的原始异常消息字符串的引用
    : std::runtime_error(msg),  // 调用基类 std::runtime_error 的构造函数
      python_class_name_(std::move(python_class_name)),  // 初始化 Python 类名成员变量
      original_msg_(std::move(original_msg)) {}  // 初始化原始异常消息成员变量

// 返回静态成员变量 caughtOriginalMsg 的引用，获取捕获的原始异常消息
const std::string& JITException::getCaughtOriginalMsg() {
  return caughtOriginalMsg;
}

// 返回静态成员变量 caughtPythonClassName 的引用，获取捕获的 Python 类名
const std::string& JITException::getCaughtPythonClassName() {
  return caughtPythonClassName;
}

// 设置静态成员变量 caughtOriginalMsg 的值，设置捕获的原始异常消息
void JITException::setCaughtOriginalMsg(const std::string& msg) {
  caughtOriginalMsg = msg;
}

// 设置静态成员变量 caughtPythonClassName 的值，设置捕获的 Python 类名
void JITException::setCaughtPythonClassName(const std::string& pythonClassName) {
  caughtPythonClassName = pythonClassName;
}

} // 命名空间结束，torch::jit
```