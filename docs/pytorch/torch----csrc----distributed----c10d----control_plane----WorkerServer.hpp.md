# `.\pytorch\torch\csrc\distributed\c10d\control_plane\WorkerServer.hpp`

```
#pragma once
// 预处理指令，确保头文件只被编译一次

#include <string>
// 包含 C++ 标准库中的 string 头文件

#include <thread>
// 包含 C++ 标准库中的 thread 头文件，用于多线程支持

#include <unordered_map>
// 包含 C++ 标准库中的 unordered_map 头文件，实现无序映射容器

#include <httplib.h>
// 包含 httplib 库的头文件，提供 HTTP 服务器和客户端功能

#include <c10/util/intrusive_ptr.h>
// 包含 c10 库中的 intrusive_ptr 头文件，实现了指针的引用计数

#include <torch/csrc/distributed/c10d/control_plane/Handlers.hpp>
// 包含 Torch 库中的分布式控制平面相关处理程序的头文件

namespace c10d {
namespace control_plane {

class TORCH_API WorkerServer : public c10::intrusive_ptr_target {
// 定义一个名为 WorkerServer 的类，继承自 c10::intrusive_ptr_target
 public:
  WorkerServer(const std::string& hostOrFile, int port = -1);
  // 构造函数声明，接受一个字符串和一个可选的整数参数

  ~WorkerServer();
  // 析构函数声明，用于清理资源

  void shutdown();
  // 声明一个公共成员函数 shutdown，用于关闭服务器

 private:
  httplib::Server server_;
  // httplib 库中的 Server 对象，用于实现 HTTP 服务器功能

  std::thread serverThread_;
  // C++ 标准库中的线程对象，用于在单独的线程中运行服务器
};

} // namespace control_plane
} // namespace c10d
```