# `.\pytorch\torch\lib\libshm\manager.cpp`

```py
// 包含必要的系统和库头文件
#include <fcntl.h>
#include <poll.h>
#include <sys/mman.h>
#include <unistd.h>
#include <algorithm>  // 用于算法操作，如std::remove_if
#include <cerrno>     // 包含errno变量和错误代码定义
#include <memory>     // 提供std::unique_ptr等智能指针工具
#include <set>        // 提供std::set容器，用于存储唯一的元素集合
#include <unordered_map>  // 提供std::unordered_map容器，用于关联容器操作
#include <vector>     // 提供std::vector容器，用于动态数组操作

#include <c10/util/tempfile.h>  // 提供c10::TempDir工具类

#include <libshm/err.h>    // 提供与共享内存错误相关的宏和函数
#include <libshm/socket.h>  // 提供与共享内存套接字操作相关的功能

const int SHUTDOWN_TIMEOUT = 2000;  // 定义2秒的关闭超时时间

#ifdef DEBUG_LOG
#define COLOR "\033[31;1m"  // 定义调试信息颜色
#define RESET "\033[0m"     // 定义控制台颜色重置
#define __DEBUG(msg, ...) fprintf(stderr, COLOR msg "%c" RESET, __VA_ARGS__);  // 定义带颜色的调试输出宏
#define DEBUG(...) __DEBUG(__VA_ARGS__, '\n')  // 调用带颜色的调试输出宏
#else
#define DEBUG(...) (void)0  // 如果没有定义DEBUG_LOG，则禁用调试输出
#endif

struct ClientSession {
  ClientSession(ManagerSocket s) : socket(std::move(s)), pid(0) {}  // 定义客户端会话结构，包含套接字和进程ID

  ManagerSocket socket;  // 管理套接字对象
  pid_t pid;             // 进程ID
};

std::vector<struct pollfd> pollfds;  // 定义用于poll系统调用的文件描述符向量
std::unordered_map<int, ClientSession> client_sessions;  // 定义客户端会话的哈希表，键为文件描述符
// TODO: 时不时检查对象是否已被释放
std::set<std::string> used_objects;  // 定义存储已使用对象名称的集合

// 注册文件描述符到pollfds向量中
void register_fd(int fd) {
  struct pollfd pfd = {0};  // 初始化pollfd结构体
  pfd.fd = fd;               // 设置文件描述符
  pfd.events = POLLIN;       // 设置关注的事件为可读事件
  pollfds.push_back(pfd);    // 将设置好的pollfd结构体添加到向量中
}

// 从pollfds向量中注销文件描述符
void unregister_fd(int fd) {
  pollfds.erase(
      std::remove_if(
          pollfds.begin(),
          pollfds.end(),
          [fd](const struct pollfd& pfd) { return pfd.fd == fd; }),  // 使用lambda函数查找并移除特定的文件描述符
      pollfds.end());
  client_sessions.erase(fd);  // 同时从客户端会话哈希表中移除对应的文件描述符条目
}

// 打印初始化消息到标准输出
void print_init_message(std::string_view message) {
  ssize_t written_bytes = -1;  // 初始化写入字节数为-1
  while (!message.empty()) {
    // NOLINTNEXTLINE(bugprone-assignment-in-if-condition)
    SYSCHECK_ERR_RETURN_NEG1(
        written_bytes = write(1, message.data(), message.size()));  // 使用write系统调用写入消息到标准输出
    message.remove_prefix(written_bytes);  // 更新消息视图，去除已写入的部分
  }
  written_bytes = 0;  // 重新初始化写入字节数为0
  while (written_bytes != 1) {
    // NOLINTNEXTLINE(bugprone-assignment-in-if-condition)
    SYSCHECK_ERR_RETURN_NEG1(written_bytes = write(1, "\n", 1));  // 继续使用write系统调用写入换行符到标准输出
  }
}

// 检查给定名称的对象是否存在
bool object_exists(const char* name) {
  int fd = shm_open(name, O_RDONLY, 0);  // 尝试以只读方式打开共享内存对象
  if (fd >= 0) {  // 如果成功打开文件描述符
    close(fd);    // 关闭文件描述符
    return true;  // 返回对象存在的标志
  } else {
    return false;  // 否则返回对象不存在的标志
  }
}

// 释放已使用的对象
void free_used_object(const std::string& name) {
  if (!object_exists(name.c_str())) {  // 如果对象不存在
    DEBUG("object %s appears to have been freed", name.c_str());  // 输出调试信息表明对象已被释放
    used_objects.erase(name);  // 从已使用对象集合中移除该对象
  } else {
    DEBUG("object %s still exists", name.c_str());  // 输出调试信息表明对象仍然存在
  }
}

// 主函数入口
// NOLINTNEXTLINE(bugprone-exception-escape)
int main(int argc, char* argv[]) {
  setsid();  // 创建一个新的会话，并将调用进程的进程组ID设置为新会话的会话ID

  std::unique_ptr<ManagerServerSocket> srv_socket;  // 声明管理服务器套接字的唯一指针
  std::optional<c10::TempDir> tempdir;  // 声明一个可选的临时目录对象
  try {
    tempdir = c10::try_make_tempdir(/*name_prefix=*/"torch-shm-dir-");  // 尝试创建一个带有指定前缀的临时目录
    if (!tempdir.has_value()) {  // 如果未成功创建临时目录
      throw std::runtime_error(
          "could not generate a random directory for manager socket");  // 抛出运行时错误
    }

    std::string tempfile = tempdir->name + "/manager.sock";  // 生成管理套接字的临时文件路径

    srv_socket = std::make_unique<ManagerServerSocket>(tempfile);  // 创建管理服务器套接字对象
    register_fd(srv_socket->socket_fd);  // 注册套接字文件描述符到pollfds向量中
    print_init_message(tempfile.c_str());  // 打印初始化消息到标准输出
    DEBUG("opened socket %s", tempfile.c_str());  // 输出调试信息表明套接字已打开
  } catch (const std::exception& e) {
    std::string message("ERROR: ");  // 定义错误消息字符串
    message += e.what();  // 追加异常信息到错误消息字符串
    print_init_message(message.c_str());  // 打印错误消息到标准输出
  // 返回整数1，表示函数执行成功
  return 1;
} catch (...) {
  // 捕获所有异常，打印错误消息并返回1
  print_init_message("ERROR: unhandled exception");
  return 1;
}

// 设置超时时间为-1
int timeout = -1;
// 创建存储要添加和移除的文件描述符的向量
std::vector<int> to_add;
std::vector<int> to_remove;
// 无限循环，处理事件
for (;;) {
  // 初始化事件数为-1
  int nevents = -1;
  // 如果客户端会话为空，则设置超时时间为SHUTDOWN_TIMEOUT
  if (client_sessions.empty())
    timeout = SHUTDOWN_TIMEOUT;
  // 调用poll函数等待事件，返回值nevents保存事件数，检查错误并处理
  // NOLINTNEXTLINE(bugprone-assignment-in-if-condition)
  SYSCHECK_ERR_RETURN_NEG1(
      nevents = poll(pollfds.data(), pollfds.size(), timeout));
  timeout = -1;
  // 如果没有事件发生且客户端会话为空，则退出循环
  if (nevents == 0 && client_sessions.empty())
    break;

  // 遍历pollfds中的每一个文件描述符
  for (auto& pfd : pollfds) {
    // 如果有错误或挂起事件发生
    if (pfd.revents & (POLLERR | POLLHUP)) {
      // 某个进程结束，打印调试信息并准备移除该文件描述符
      DEBUG("detaching process");
      auto& session = client_sessions.at(pfd.fd);
      (void)session;
      DEBUG("%d has died", session.pid);
      to_remove.push_back(pfd.fd);
    } else if (pfd.revents & POLLIN) {
      // 如果有数据可读
      if (pfd.fd == srv_socket->socket_fd) {
        // 新客户端连接，打印调试信息并接受连接
        DEBUG("registered new client");
        auto client = srv_socket->accept();
        int fd = client.socket_fd;
        to_add.push_back(fd);
        client_sessions.emplace(fd, std::move(client));
      } else {
        // 收到分配信息，打印调试信息并处理分配请求
        DEBUG("got alloc info");
        auto& session = client_sessions.at(pfd.fd);
        AllocInfo info = session.socket.receive();
        session.pid = info.pid;
        DEBUG(
            "got alloc info: %d %d %s",
            (int)info.free,
            info.pid,
            info.filename);
        // 如果是释放请求，则释放对应的对象
        if (info.free) {
          free_used_object(info.filename);
        } else {
          // 否则注册使用的对象，并确认处理结果
          used_objects.insert(info.filename);
          DEBUG("registered object %s", info.filename);
          session.socket.confirm();
        }
      }
    }
  }

  // 注册需要添加的文件描述符
  for (int fd : to_add)
    register_fd(fd);
  to_add.clear();

  // 移除需要移除的文件描述符
  for (int fd : to_remove)
    unregister_fd(fd);
  to_remove.clear();
}

// 对于每个正在使用的对象，打印调试信息并释放共享内存
for (auto& obj_name : used_objects) {
  DEBUG("freeing %s", obj_name.c_str());
  shm_unlink(obj_name.c_str());
}

// 清理所有文件描述符
for (auto& pfd : pollfds) {
  unregister_fd(pfd.fd);
}
// 清理manager.sock文件
srv_socket->remove();
// 自动清理目录

// 打印调试信息，表示管理器完成工作
DEBUG("manager done");
// 返回整数0，表示函数执行成功
return 0;
}



# 这是一个代码块的结束，对应于某个控制流语句（例如 if、for、while等）的结束。
```