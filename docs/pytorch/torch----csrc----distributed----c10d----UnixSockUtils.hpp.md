# `.\pytorch\torch\csrc\distributed\c10d\UnixSockUtils.hpp`

```
#pragma once
// 预处理指令：确保头文件只被编译一次

#include <torch/csrc/distributed/c10d/Utils.hpp>
// 包含 Torch 分布式库的工具函数头文件

namespace c10d::tcputil {
// 进入命名空间 c10d::tcputil

#define CONNECT_SOCKET_OFFSET 2
// 定义连接套接字偏移量为 2

inline int poll(struct pollfd* fds, unsigned long nfds, int timeout) {
  return ::poll(fds, nfds, timeout);
}
// 内联函数 poll：调用系统的 poll 函数进行文件描述符的轮询操作

inline void addPollfd(
    std::vector<struct pollfd>& fds,
    int socket,
    short events) {
  fds.push_back({.fd = socket, .events = events});
}
// 内联函数 addPollfd：向 fds 向量中添加一个新的 pollfd 结构体，表示一个需要轮询的套接字

inline struct ::pollfd getPollfd(int socket, short events) {
  struct ::pollfd res = {.fd = socket, .events = events};
  return res;
}
// 内联函数 getPollfd：返回一个新的 pollfd 结构体，表示一个需要轮询的套接字

} // namespace c10d::tcputil
// 离开命名空间 c10d::tcputil
```