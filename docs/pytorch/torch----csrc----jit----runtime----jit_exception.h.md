# `.\pytorch\torch\csrc\jit\runtime\jit_exception.h`

```py
#pragma once
// 预处理指令，确保本文件仅被编译一次

#include <stdexcept>
// 引入标准异常库，用于定义和抛出异常

#include <c10/util/Optional.h>
// 引入C10库中的Optional模块，用于支持可选值的操作

#include <torch/csrc/Export.h>
// 引入Torch的导出定义头文件，用于声明导出接口

#include <string>
// 引入标准字符串库，支持字符串操作

namespace torch::jit {
// 进入torch::jit命名空间

struct TORCH_API JITException : public std::runtime_error {
  // 定义JITException结构体，继承自std::runtime_error

  explicit JITException(
      const std::string& msg,
      std::optional<std::string> python_class_name = c10::nullopt,
      std::optional<std::string> original_msg = c10::nullopt);
  // JITException结构体的构造函数声明，接受异常消息、Python类名和原始消息作为参数

  std::optional<std::string> getPythonClassName() const {
    return python_class_name_;
  }
  // 获取Python类名的可选值函数声明

  std::optional<std::string> getOriginalMsg() const {
    return original_msg_;
  }
  // 获取原始消息的可选值函数声明

  static const std::string& getCaughtOriginalMsg();
  // 静态函数声明，获取捕获的原始消息

  static const std::string& getCaughtPythonClassName();
  // 静态函数声明，获取捕获的Python类名

  static void setCaughtOriginalMsg(const std::string& msg);
  // 静态函数声明，设置捕获的原始消息

  static void setCaughtPythonClassName(const std::string& pythonClassName);
  // 静态函数声明，设置捕获的Python类名

 private:
  std::optional<std::string> python_class_name_;
  // 私有成员变量，存储Python类名的可选值

  std::optional<std::string> original_msg_;
  // 私有成员变量，存储原始消息的可选值
};

} // namespace torch::jit
// 退出torch::jit命名空间
```