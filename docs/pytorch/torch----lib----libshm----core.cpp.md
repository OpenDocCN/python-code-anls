# `.\pytorch\torch\lib\libshm\core.cpp`

```
// 引入必要的头文件：数组，字符串处理，字符串，无序映射
#include <array>
#include <cstring>
#include <string>
#include <unordered_map>

// 引入 libshm 库的头文件
#include <libshm/err.h>
#include <libshm/libshm.h>
#include <libshm/socket.h>

// 管理客户端套接字的无序映射
std::unordered_map<std::string, ClientSocket> managers;
// 管理器可执行文件的路径
std::string manager_executable_path;

// 获取文件分配信息的函数定义
AllocInfo get_alloc_info(const char* filename) {
  AllocInfo info = {};
  info.pid = getpid();  // 获取当前进程的 ID
  info.free = false;    // 初始化分配信息中的空闲状态为 false
  size_t len = strlen(filename);  // 计算文件名的长度
  if (len >= sizeof(info.filename)) {
    throw std::runtime_error("MapAllocatorContext_filename too long");  // 如果文件名过长，抛出运行时错误
  }
  memcpy(info.filename, filename, len + 1);  // 复制文件名到分配信息结构体中
  return info;  // 返回填充后的分配信息结构体
}

// 启动管理器进程的函数定义
void start_manager() {
  std::array<int, 2> pipe_ends;  // 创建整数数组以保存管道两端描述符
  SYSCHECK_ERR_RETURN_NEG1(pipe(pipe_ends.data()));  // 创建管道，检查错误并返回 -1

  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  pid_t pid;
  SYSCHECK_ERR_RETURN_NEG1(pid = fork());  // 创建子进程，检查错误并返回 -1
  if (!pid) {  // 在子进程中执行以下代码块
    SYSCHECK_ERR_RETURN_NEG1(close(pipe_ends[0]));  // 关闭管道读取端
    SYSCHECK_ERR_RETURN_NEG1(dup2(pipe_ends[1], 1)); // 将标准输出重定向到管道写入端
    SYSCHECK_ERR_RETURN_NEG1(close(pipe_ends[1]));  // 关闭管道写入端
    execl(manager_executable_path.c_str(), "torch_shm_manager", NULL);  // 用管理器可执行文件替换当前进程，执行出错则输出错误信息并退出

    std::string msg("ERROR: execl failed: ");
    msg += std::strerror(errno);  // 获取错误信息
    msg += '\n';  // 添加换行符
    auto res = write(1, msg.c_str(), msg.size());  // 将错误信息写入标准输出
    (void)res;  // 空操作，避免编译器警告

    exit(1);  // 退出子进程，返回错误状态码
  }
  SYSCHECK_ERR_RETURN_NEG1(close(pipe_ends[1]));  // 关闭父进程中的管道写入端

  constexpr auto MAX_BUFFER_SIZE = 1000;  // 定义最大缓冲区大小为 1000 字节
  std::array<char, MAX_BUFFER_SIZE> buffer;  // 创建固定大小的字符数组作为缓冲区
  std::string handle;  // 创建字符串来存储管理器的句柄
  while (handle.empty() || handle.back() != '\n') {  // 当句柄为空或最后一个字符不是换行符时循环
    const auto bytes_read = read(pipe_ends[0], buffer.data(), buffer.size());  // 从管道读取数据到缓冲区
    SYSCHECK_ERR_RETURN_NEG1(bytes_read);  // 检查读取错误并返回 -1
    if (bytes_read == 0) {  // 如果读取到的字节数为 0
      break;  // 跳出循环
    }
    handle.append(buffer.data(), bytes_read);  // 将读取的数据追加到句柄字符串末尾
  }
  SYSCHECK_ERR_RETURN_NEG1(close(pipe_ends[0]));  // 关闭管道读取端
  if (handle.length() == 0) {  // 如果句柄长度为 0
    std::string msg("no response from torch_shm_manager at \"");
    msg += manager_executable_path;
    msg += "\"";  // 构造错误信息字符串
    throw std::runtime_error(msg);  // 抛出运行时错误
  }

  handle.pop_back();  // 移除末尾的换行符
  if (handle.rfind("ERROR: ", 0) == 0) {  // 如果句柄以 "ERROR: " 开头
    std::string msg("torch_shm_manager at \"");
    msg += manager_executable_path;
    msg += "\": ";
    msg += handle.substr(7);  // 移除开头的 "ERROR: "
    throw std::runtime_error(msg);  // 抛出运行时错误
  }

  ClientSocket manager{handle};  // 使用句柄创建客户端套接字对象
  managers.emplace(std::move(handle), std::move(manager));  // 将句柄和对应的客户端套接字对象插入到无序映射中
}

// 根据管理器句柄获取客户端套接字的函数定义
ClientSocket& get_manager_socket(const std::string& manager_handle) {
  auto it = managers.find(manager_handle);  // 在映射中查找给定句柄的客户端套接字
  if (it == managers.end()) {  // 如果未找到对应的套接字
    auto socket = ClientSocket(manager_handle);  // 创建新的客户端套接字对象
    auto result = managers.emplace(manager_handle, std::move(socket));  // 将新对象插入映射中
    return result.first->second;  // 返回插入的客户端套接字对象的引用
  } else {  // 如果找到了对应的套接字
    return it->second;  // 返回找到的客户端套接字对象的引用
  }
}

// 初始化 libshm 库的函数定义
void libshm_init(const char* manager_exec_path) {
  manager_executable_path = std::string(manager_exec_path);  // 设置管理器可执行文件的路径
}

// THManagedMapAllocatorInit 类的构造函数定义
THManagedMapAllocatorInit::THManagedMapAllocatorInit(
    const char* manager_handle,
    const char* filename)
    : manager_handle_(manager_handle ? manager_handle : "") {
  // TODO: unlock GIL when contacting the manager
  try {
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    ClientSocket* socket;
    // 声明一个指向 ClientSocket 的指针变量 socket

    if (!manager_handle_.empty()) {
      // 如果 manager_handle_ 不为空，说明已经有管理器的句柄
      socket = &get_manager_socket(manager_handle_);
      // 将 socket 指向通过 manager_handle_ 获取的管理器的 socket
    } else {
      // 如果 manager_handle_ 为空，则需要选择一个管理器
      if (managers.empty()) {
        // 如果 managers 容器为空，表示没有可用的管理器，需要启动一个新的管理器
        start_manager();
      }
      // 获取第一个管理器的迭代器
      const auto& manager = managers.begin();
      // 将 manager_handle_ 设置为第一个管理器的句柄
      manager_handle_ = manager->first;
      // 将 socket 指向第一个管理器的 socket
      socket = &manager->second;
    }

    // 获取文件的分配信息
    AllocInfo info = get_alloc_info(filename);
    // 在选定的 socket 上注册分配信息
    socket->register_allocation(info);

  } catch (std::exception& e) {
    // 捕获任何异常，并使用 TORCH_CHECK 抛出错误
    TORCH_CHECK(false, e.what());
  }
}

// THManagedMapAllocator 构造函数实现
THManagedMapAllocator::THManagedMapAllocator(
    const char* manager_handle,
    const char* filename,
    int flags,
    size_t size)
    : THManagedMapAllocatorInit(manager_handle, filename),  // 调用基类 THManagedMapAllocatorInit 的构造函数初始化
      at::RefcountedMapAllocator(filename, flags, size) {}  // 调用基类 at::RefcountedMapAllocator 的构造函数初始化

// THManagedMapAllocator 类的 close 方法实现
void THManagedMapAllocator::close() {
  // 如果已经关闭，则直接返回
  if (closed_)
    return;
  // 获取文件的分配信息
  AllocInfo info = get_alloc_info(filename());
  // 标记为释放状态
  info.free = true;
  // 获取管理器套接字并注册释放信息
  ClientSocket& socket = get_manager_socket(manager_handle_);
  // 调用基类的 close 方法
  at::RefcountedMapAllocator::close();
  // 注册释放信息到套接字
  socket.register_deallocation(info);
}

// 静态函数，用于删除 THManagedMapAllocator 实例
static void deleteTHManagedMapAllocator(void* ptr) {
  delete static_cast<THManagedMapAllocator*>(ptr);
}

// 创建并返回 DataPtr 对象
at::DataPtr THManagedMapAllocator::makeDataPtr(
    const char* manager_handle,
    const char* filename,
    int flags,
    size_t size) {
  // 创建 THManagedMapAllocator 实例
  auto* context =
      new THManagedMapAllocator(manager_handle, filename, flags, size);
  // 返回 DataPtr 对象，包括数据指针、上下文指针、删除函数和设备类型
  return {
      context->data(),
      context,
      &deleteTHManagedMapAllocator,
      at::DeviceType::CPU};
}

// 从 DataPtr 中获取 THManagedMapAllocator 实例指针
THManagedMapAllocator* THManagedMapAllocator::fromDataPtr(
    const at::DataPtr& dptr) {
  // 转换并返回 THManagedMapAllocator 实例指针
  return dptr.cast_context<THManagedMapAllocator>(&deleteTHManagedMapAllocator);
}
```