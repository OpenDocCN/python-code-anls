# `.\pytorch\torch\csrc\distributed\c10d\WinSockUtils.hpp`

```
#pragma once
// 使用#pragma once确保头文件只被编译一次

#include <torch/csrc/distributed/c10d/Utils.hpp>
// 引入torch分布式库中的Utils.hpp头文件

namespace c10d::tcputil {
// 进入命名空间c10d::tcputil

#define CONNECT_SOCKET_OFFSET 1
// 定义连接套接字偏移量为1

inline int poll(struct pollfd* fdArray, unsigned long fds, int timeout) {
  // 定义poll函数，用于调用Windows平台的WSAPoll函数进行轮询
  return WSAPoll(fdArray, fds, timeout);
}

inline void addPollfd(
    std::vector<struct pollfd>& fds,
    int socket,
    short events) {
  // 定义addPollfd函数，向fdArray中添加新的pollfd结构体
  fds.push_back({(SOCKET)socket, events});
}

inline struct ::pollfd getPollfd(int socket, short events) {
  // 定义getPollfd函数，返回一个包含给定套接字和事件的pollfd结构体
  struct ::pollfd res = {(SOCKET)socket, events};
  return res;
}

} // namespace c10d::tcputil
// 结束命名空间c10d::tcputil
```