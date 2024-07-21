# `.\pytorch\torch\csrc\distributed\c10d\control_plane\Handlers.hpp`

```
#pragma once
// 预处理指令，确保头文件只被编译一次

#include <functional>
// 引入函数对象（函数指针的包装器）的标准库

#include <map>
// 引入关联容器 map 的标准库

#include <string>
// 引入处理字符串的标准库

#include <c10/macros/Export.h>
// 引入 c10 库中的导出宏

namespace c10d {
namespace control_plane {

// Request represents a request to the handler. This conceptually maps to an
// HTTP request but could be called via other transports.
class TORCH_API Request {
 public:
  virtual ~Request() = default;

  virtual const std::string& body() = 0;
  // 定义纯虚函数，获取请求体内容的字符串引用

  virtual const std::multimap<std::string, std::string>& params() const = 0;
  // 定义纯虚函数，获取请求参数的多值映射容器的常量引用
};

// Response represents a response to the handler. This conceptually maps to an
// HTTP response but could be called via other transports.
class TORCH_API Response {
 public:
  virtual ~Response() = default;

  // Set the response body to the provided string.
  // TODO: add support for chunked responses
  virtual void setContent(
      std::string&& content,
      const std::string& content_type) = 0;
  // 定义纯虚函数，设置响应体内容为提供的字符串，并指定内容类型

  // Set the response status code.
  // These should match standard HTTP status codes.
  virtual void setStatus(int status) = 0;
  // 定义纯虚函数，设置响应状态码，应符合标准的 HTTP 状态码
};

using HandlerFunc = std::function<void(const Request&, Response&)>;
// 使用函数对象类型定义 HandlerFunc，接受 Request 和 Response 作为参数，返回 void

// Registers a handler. The name needs to be unique and can be called by using
// getHandler directly or via WorkerServer for remote requests.
// These handlers are called from a background C++ thread concurrently with the
// main thread. These handlers need to be thread safe and not cause issues
// during Python training.
TORCH_API void registerHandler(const std::string& name, HandlerFunc f);
// 注册一个处理函数，需要提供唯一的名字和 HandlerFunc 函数对象

// Fetches a handler by name.
TORCH_API HandlerFunc getHandler(const std::string& name);
// 根据名字获取已注册的处理函数 HandlerFunc

TORCH_API std::vector<std::string> getHandlerNames();
// 获取所有已注册处理函数的名字列表

// Registers a handler statically.
// See registerHandler for more details.
class TORCH_API RegisterHandler {
 public:
  RegisterHandler(const std::string& name, HandlerFunc f) {
    registerHandler(name, f);
  }
  // 构造函数，静态注册一个处理函数

  // disable move, copy
  RegisterHandler(const RegisterHandler&) = delete;
  RegisterHandler(RegisterHandler&&) = delete;
  RegisterHandler& operator=(const RegisterHandler&) = delete;
  RegisterHandler& operator=(RegisterHandler&&) = delete;
  // 禁用移动构造、复制构造和移动赋值、复制赋值运算符
};

} // namespace control_plane
} // namespace c10d
```