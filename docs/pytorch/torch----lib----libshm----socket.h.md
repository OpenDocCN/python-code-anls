# `.\pytorch\torch\lib\libshm\socket.h`

```py
#pragma once
// 引入必要的头文件
#include <poll.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/un.h>
#include <unistd.h>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <string>

#include <libshm/alloc_info.h>
#include <libshm/err.h>

// Socket 类定义
class Socket {
 public:
  int socket_fd; // Socket 文件描述符

 protected:
  // 默认构造函数，创建 AF_UNIX 套接字
  Socket() {
    SYSCHECK_ERR_RETURN_NEG1(socket_fd = socket(AF_UNIX, SOCK_STREAM, 0));
  }
  // 禁用拷贝构造函数
  Socket(const Socket& other) = delete;
  // 移动构造函数，使用 noexcept 保证不抛出异常
  Socket(Socket&& other) noexcept : socket_fd(other.socket_fd) {
    other.socket_fd = -1;
  };
  // 使用现有文件描述符创建 Socket 对象
  explicit Socket(int fd) : socket_fd(fd) {}

  // 虚析构函数，关闭套接字
  virtual ~Socket() {
    if (socket_fd != -1)
      close(socket_fd);
  }

  // 准备 AF_UNIX 地址结构体
  struct sockaddr_un prepare_address(const char* path) {
    struct sockaddr_un address;
    address.sun_family = AF_UNIX;
    strcpy(address.sun_path, path);
    return address;
  }

  // 计算 sockaddr_un 结构体的长度
  size_t address_length(struct sockaddr_un address) {
    return offsetof(sockaddr_un, sun_path) + strlen(address.sun_path) + 1;
  }

  // 接收数据函数，使用 poll 实现超时和错误处理
  void recv(void* _buffer, size_t num_bytes) {
    char* buffer = (char*)_buffer;
    size_t bytes_received = 0;
    ssize_t step_received;
    struct pollfd pfd = {};
    pfd.fd = socket_fd;
    pfd.events = POLLIN;
    while (bytes_received < num_bytes) {
      SYSCHECK_ERR_RETURN_NEG1(poll(&pfd, 1, 1000));
      if (pfd.revents & POLLIN) {
        SYSCHECK_ERR_RETURN_NEG1(
            step_received =
                ::read(socket_fd, buffer, num_bytes - bytes_received));
        if (step_received == 0)
          throw std::runtime_error("Other end has closed the connection");
        bytes_received += step_received;
        buffer += step_received;
      } else if (pfd.revents & (POLLERR | POLLHUP)) {
        throw std::runtime_error(
            "An error occurred while waiting for the data");
      } else {
        throw std::runtime_error(
            "Shared memory manager connection has timed out");
      }
    }
  }

  // 发送数据函数
  void send(const void* _buffer, size_t num_bytes) {
    const char* buffer = (const char*)_buffer;
    size_t bytes_sent = 0;
    ssize_t step_sent;
    while (bytes_sent < num_bytes) {
      SYSCHECK_ERR_RETURN_NEG1(
          step_sent = ::write(socket_fd, buffer, num_bytes));
      bytes_sent += step_sent;
      buffer += step_sent;
    }
  }
};

// 继承自 Socket 类的 ManagerSocket 类
class ManagerSocket : public Socket {
 public:
  // 使用现有文件描述符创建 ManagerSocket 对象
  explicit ManagerSocket(int fd) : Socket(fd) {}

  // 接收 AllocInfo 结构体数据
  AllocInfo receive() {
    AllocInfo info;
    recv(&info, sizeof(info));
    return info;
  }

  // 发送确认信息
  void confirm() {
    send("OK", 2);
  }
};

// 继承自 Socket 类的 ManagerServerSocket 类
class ManagerServerSocket : public Socket {
 public:
  // 使用指定路径创建 ManagerServerSocket 对象
  explicit ManagerServerSocket(const std::string& path) {
    socket_path = path;
  try {
    // 准备 Unix 域套接字地址结构体
    struct sockaddr_un address = prepare_address(path.c_str());
    // 计算地址结构体的长度
    size_t len = address_length(address);
    // 将套接字绑定到指定地址
    SYSCHECK_ERR_RETURN_NEG1(
        bind(socket_fd, (struct sockaddr*)&address, len));
    // 开始监听连接请求，允许最多 10 个待连接
    SYSCHECK_ERR_RETURN_NEG1(listen(socket_fd, 10));
  } catch (std::exception&) {
    // 如果出现异常，关闭套接字并重新抛出异常
    SYSCHECK_ERR_RETURN_NEG1(close(socket_fd));
    throw;
  }
}

void remove() {
  // 获取套接字文件的状态信息
  struct stat file_stat;
  if (fstat(socket_fd, &file_stat) == 0)
    // 如果文件存在，则尝试删除套接字文件
    SYSCHECK_ERR_RETURN_NEG1(unlink(socket_path.c_str()));
}

virtual ~ManagerServerSocket() {
  // 在对象销毁时，删除关联的套接字文件
  unlink(socket_path.c_str());
}

ManagerSocket accept() {
  int client_fd;
  struct sockaddr_un addr;
  socklen_t addr_len = sizeof(addr);
  // 接受客户端的连接请求，返回客户端套接字文件描述符
  SYSCHECK_ERR_RETURN_NEG1(
      client_fd = ::accept(socket_fd, (struct sockaddr*)&addr, &addr_len));
  // 返回一个 ManagerSocket 对象，用于管理客户端连接
  return ManagerSocket(client_fd);
}

std::string socket_path;
};

class ClientSocket : public Socket {
 public:
  // 构造函数，接受一个路径作为参数
  explicit ClientSocket(const std::string& path) {
    try {
      // 准备 UNIX 域套接字地址结构
      struct sockaddr_un address = prepare_address(path.c_str());
      // 计算地址结构的长度
      size_t len = address_length(address);
      // 连接到指定的 UNIX 域套接字
      SYSCHECK_ERR_RETURN_NEG1(
          connect(socket_fd, (struct sockaddr*)&address, len));
    } catch (std::exception&) {
      // 如果发生异常，关闭套接字并重新抛出异常
      SYSCHECK_ERR_RETURN_NEG1(close(socket_fd));
      throw;
    }
  }

  // 向共享内存管理器注册内存分配信息
  void register_allocation(AllocInfo& info) {
    // 创建一个长度为 3 的缓冲区，并初始化为零
    char buffer[3] = {0, 0, 0};
    // 发送分配信息到服务器
    send(&info, sizeof(info));
    // 接收服务器的响应
    recv(buffer, 2);
    // 如果响应不是 "OK"，抛出运行时异常
    if (strcmp(buffer, "OK") != 0)
      throw std::runtime_error(
          "Shared memory manager didn't respond with an OK");
  }

  // 向共享内存管理器注册内存释放信息
  void register_deallocation(AllocInfo& info) {
    // 发送释放信息到服务器
    send(&info, sizeof(info));
  }
};
```