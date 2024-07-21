# `.\pytorch\torch\csrc\distributed\c10d\socket.h`

```
// 版权声明和许可信息
// Meta Platforms, Inc. 及其关联公司版权所有。
//
// 此源代码根据根目录中的 LICENSE 文件中的 BSD-style 许可证授权。

#pragma once

#include <chrono>                          // 包含时间相关的头文件
#include <cstdint>                         // 包含整数类型的头文件
#include <memory>                          // 包含智能指针的头文件
#include <string>                          // 包含字符串操作的头文件

#include <c10/macros/Macros.h>             // 包含 c10 宏定义的头文件
#include <c10/util/Exception.h>            // 包含异常处理的头文件
#include <torch/csrc/distributed/c10d/Backoff.hpp>   // 包含 Backoff 策略的头文件
#include <torch/csrc/distributed/c10d/exception.h>    // 包含分布式异常处理的头文件

namespace c10d {
namespace detail {

class SocketOptions {
 public:
  // 设置是否优先使用 IPv6
  SocketOptions& prefer_ipv6(bool value) noexcept {
    prefer_ipv6_ = value;

    return *this;
  }

  // 返回是否优先使用 IPv6
  bool prefer_ipv6() const noexcept {
    return prefer_ipv6_;
  }

  // 设置连接超时时间
  SocketOptions& connect_timeout(std::chrono::seconds value) noexcept {
    connect_timeout_ = value;

    return *this;
  }

  // 返回连接超时时间
  std::chrono::seconds connect_timeout() const noexcept {
    return connect_timeout_;
  }

  // 设置连接回退策略
  // 用于 socket 连接操作的回退策略
  SocketOptions& connect_backoff(std::shared_ptr<Backoff> value) noexcept {
    connect_backoff_ = std::move(value);

    return *this;
  }

  // 返回连接回退策略
  const std::shared_ptr<Backoff>& connect_backoff() const noexcept {
    return connect_backoff_;
  }

 private:
  bool prefer_ipv6_ = true;   // 默认优先使用 IPv6
  std::chrono::seconds connect_timeout_{30};   // 默认连接超时时间为 30 秒
  std::shared_ptr<Backoff> connect_backoff_{   // 默认连接回退策略为 FixedBackoff，间隔 1000 毫秒
      std::make_shared<FixedBackoff>(std::chrono::milliseconds(1000))};
};

class SocketImpl;

class Socket {
 public:
  // 初始化底层 socket 库，必须在调用其他 socket 函数之前调用
  static void initialize();

  // 创建并监听指定端口的 Socket
  static Socket listen(std::uint16_t port, const SocketOptions& opts = {});

  // 从给定文件描述符创建监听的 Socket，并指定预期端口号
  static Socket listenFromFd(int fd, std::uint16_t expected_port);

  // 连接指定主机和端口的 Socket
  static Socket connect(
      const std::string& host,
      std::uint16_t port,
      const SocketOptions& opts = {});

  Socket() noexcept = default;   // 默认构造函数

  Socket(const Socket& other) = delete;   // 禁止拷贝构造函数

  Socket& operator=(const Socket& other) = delete;   // 禁止赋值运算符重载

  Socket(Socket&& other) noexcept;   // 移动构造函数

  Socket& operator=(Socket&& other) noexcept;   // 移动赋值运算符重载

  ~Socket();   // 析构函数，用于清理资源

  // 接受连接请求，返回新的 Socket 对象
  Socket accept() const;

  // 返回当前 Socket 的文件描述符
  int handle() const noexcept;

  // 返回当前 Socket 绑定的端口号
  std::uint16_t port() const;

  // 等待输入事件发生，超时时间为指定的毫秒数
  bool waitForInput(std::chrono::milliseconds timeout);

  // 返回 Socket 的字符串表示形式
  std::string repr() const;

 private:
  explicit Socket(std::unique_ptr<SocketImpl>&& impl) noexcept;

  std::unique_ptr<SocketImpl> impl_;
};

} // namespace detail

} // namespace c10d
```