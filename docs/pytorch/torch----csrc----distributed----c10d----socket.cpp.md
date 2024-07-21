# `.\pytorch\torch\csrc\distributed\c10d\socket.cpp`

```
// 在源代码的开头包含必要的版权和许可声明信息

#include <torch/csrc/distributed/c10d/socket.h>

#include <cstring>  // C-style字符串操作函数
#include <system_error>  // 标准系统错误处理库
#include <utility>  // 各种实用工具函数
#include <vector>  // 向量容器库

#ifdef _WIN32
#include <mutex>  // 互斥量
#include <winsock2.h>  // Windows下的网络编程接口
#include <ws2tcpip.h>  // Windows下的TCP/IP协议相关接口
#else
#include <arpa/inet.h>  // 网络地址转换函数
#include <fcntl.h>  // 文件控制操作
#include <netdb.h>  // 网络数据库操作
#include <netinet/tcp.h>  // TCP协议定义
#include <poll.h>  // 网络I/O多路复用
#include <sys/socket.h>  // 套接字接口
#include <sys/types.h>  // 系统数据类型定义
#include <unistd.h>  // POSIX操作系统API
#endif

C10_DIAGNOSTIC_PUSH_AND_IGNORED_IF_DEFINED("-Wdeprecated")
#include <fmt/chrono.h>  // 日期和时间格式化输出
C10_DIAGNOSTIC_POP()
#include <fmt/format.h>  // 字符串格式化库

#include <torch/csrc/distributed/c10d/error.h>  // 分布式错误处理
#include <torch/csrc/distributed/c10d/exception.h>  // 分布式异常处理
#include <torch/csrc/distributed/c10d/logging.h>  // 分布式日志记录

#include <c10/util/CallOnce.h>  // 一次调用库
#include <c10/util/Optional.h>  // 可选值类型库

namespace c10d::detail {

// 匿名命名空间内定义了一些平台相关的函数和宏
namespace {
#ifdef _WIN32

// 在Windows平台下，将WSAPoll重命名为pollFd，避免源代码中的#ifdef条件
const auto pollFd = ::WSAPoll;

// Windows下的套接字选项函数，需要将void*类型的参数转换为char*类型
int getSocketOption(
    SOCKET s,
    int level,
    int optname,
    void* optval,
    int* optlen) {
  return ::getsockopt(s, level, optname, static_cast<char*>(optval), optlen);
}

// Windows下的套接字设置函数，同样需要类型转换
int setSocketOption(
    SOCKET s,
    int level,
    int optname,
    const void* optval,
    int optlen) {
  return ::setsockopt(
      s, level, optname, static_cast<const char*>(optval), optlen);
}

// 获取Windows下套接字错误的函数，使用WSAGetLastError()
inline std::error_code getSocketError() noexcept {
  return std::error_code{::WSAGetLastError(), std::system_category()};
}

// 设置Windows下套接字错误的函数，使用WSASetLastError()
inline void setSocketError(int val) noexcept {
  ::WSASetLastError(val);
}

#else

// 在非Windows平台下，使用标准的poll函数
const auto pollFd = ::poll;

// 非Windows平台下的套接字选项和设置函数，直接使用标准的函数名
const auto getSocketOption = ::getsockopt;
const auto setSocketOption = ::setsockopt;

// 获取非Windows平台下套接字错误的函数
inline std::error_code getSocketError() noexcept {
  return lastError();
}

// 设置非Windows平台下套接字错误的函数
inline void setSocketError(int val) noexcept {
  errno = val;
}

#endif

// 挂起当前线程一段指定的时间
void delay(std::chrono::milliseconds d) {
#ifdef _WIN32
  std::this_thread::sleep_for(d);  // Windows下的线程休眠函数
#else
  ::timespec req{};  // POSIX标准下的时间结构体
  auto ms = d.count();  // 将毫秒转换为秒和纳秒
  req.tv_sec = ms / 1000;
  req.tv_nsec = (ms % 1000) * 1000000;

  // 如果C++标准未指定`sleep_for()`是否支持信号处理，使用`nanosleep()`系统调用
  if (::nanosleep(&req, nullptr) != 0) {
    std::error_code err = getSocketError();
    // 忽略除EINTR以外的错误条件，因为此处失败不是关键性错误
    # 如果错误代码是 std::errc::interrupted
    if (err == std::errc::interrupted) {
        # 抛出 DistNetworkError 异常，并使用 err.value() 返回的错误消息
        C10_THROW_ERROR(DistNetworkError, std::strerror(err.value()));
    }
#endif
}

// 声明 SocketListenOp 类
class SocketListenOp;
// 声明 SocketConnectOp 类
class SocketConnectOp;
} // namespace

// SocketImpl 类的实现
class SocketImpl {
  // 友元类声明：SocketListenOp 和 SocketConnectOp
  friend class SocketListenOp;
  friend class SocketConnectOp;

 public:
#ifdef _WIN32
  // 如果在 Windows 平台，使用 SOCKET 类型作为 Handle
  using Handle = SOCKET;
#else
  // 否则，使用 int 类型作为 Handle
  using Handle = int;
#endif

#ifdef _WIN32
  // 在 Windows 平台，定义无效的 socket 值
  static constexpr Handle invalid_socket = INVALID_SOCKET;
#else
  // 在非 Windows 平台，定义无效的 socket 值
  static constexpr Handle invalid_socket = -1;
#endif

  // 构造函数，初始化 SocketImpl 实例
  explicit SocketImpl(
      Handle hnd,
      c10::optional<::addrinfo> remote = c10::nullopt) noexcept
      : hnd_{hnd}, remote_(remote) {}

  // 禁用拷贝构造函数
  SocketImpl(const SocketImpl& other) = delete;

  // 禁用拷贝赋值运算符
  SocketImpl& operator=(const SocketImpl& other) = delete;

  // 禁用移动构造函数
  SocketImpl(SocketImpl&& other) noexcept = delete;

  // 禁用移动赋值运算符
  SocketImpl& operator=(SocketImpl&& other) noexcept = delete;

  // 析构函数声明
  ~SocketImpl();

  // 接受连接请求，返回一个 SocketImpl 的智能指针
  std::unique_ptr<SocketImpl> accept() const;

  // 在执行时关闭 socket 的执行权限
  void closeOnExec() noexcept;

  // 设置 socket 为非阻塞模式
  void enableNonBlocking();

  // 设置 socket 为阻塞模式
  void disableNonBlocking();

  // 开启 TCP_NODELAY 选项，禁用 Nagle 算法
  bool enableNoDelay() noexcept;

  // 开启 IPV6_V6ONLY 选项，启用 IPv6 的双栈支持
  bool enableDualStack() noexcept;

#ifndef _WIN32
  // 在非 Windows 平台，开启地址重用选项
  bool enableAddressReuse() noexcept;
#endif

#ifdef _WIN32
  // 在 Windows 平台，开启独占地址使用选项
  bool enableExclusiveAddressUse() noexcept;
#endif

  // 获取 socket 绑定的端口号
  std::uint16_t getPort() const;

  // 返回 socket 的句柄
  Handle handle() const noexcept {
    return hnd_;
  }

  // 返回远程地址信息的可选对象引用
  const c10::optional<::addrinfo>& remote() const noexcept {
    return remote_;
  }

  // 等待 socket 输入事件发生，设置超时时间
  bool waitForInput(std::chrono::milliseconds timeout);

 private:
  // 设置 socket 的特定选项
  bool setSocketFlag(int level, int optname, bool value) noexcept;

  // socket 句柄
  Handle hnd_;
  // 远程地址信息的可选对象
  const c10::optional<::addrinfo> remote_;
};
} // namespace c10d::detail

//
// libfmt 格式化器，用于 addrinfo 和 Socket 的格式化
//
namespace fmt {

// addrinfo 的格式化器特化
template <>
struct formatter<::addrinfo> {
  // 解析格式化参数
  constexpr decltype(auto) parse(format_parse_context& ctx) const {
    return ctx.begin();
  }

  // 格式化 addrinfo 对象
  template <typename FormatContext>
  decltype(auto) format(const ::addrinfo& addr, FormatContext& ctx) const {
    char host[NI_MAXHOST], port[NI_MAXSERV]; // NOLINT

    // 尝试解析主机名和端口号
    int r = ::getnameinfo(
        addr.ai_addr,
        addr.ai_addrlen,
        host,
        NI_MAXHOST,
        port,
        NI_MAXSERV,
        NI_NUMERICSERV);
    if (r != 0) {
      // 如果无法解析主机名，显示 IP 地址
      if (addr.ai_family == AF_INET) {
        struct sockaddr_in* psai = (struct sockaddr_in*)addr.ai_addr;
        char ip[INET_ADDRSTRLEN];
        if (inet_ntop(addr.ai_family, &(psai->sin_addr), ip, INET_ADDRSTRLEN) !=
            NULL) {
          return fmt::format_to(ctx.out(), "{}:{}", ip, psai->sin_port);
        }
      } else if (addr.ai_family == AF_INET6) {
        struct sockaddr_in6* psai = (struct sockaddr_in6*)addr.ai_addr;
        char ip[INET6_ADDRSTRLEN];
        if (inet_ntop(
                addr.ai_family, &(psai->sin6_addr), ip, INET6_ADDRSTRLEN) !=
            NULL) {
          return fmt::format_to(ctx.out(), "[{}]:{}", ip, psai->sin6_port);
        }
      }
      // 如果出现未知地址家族，抛出异常
      C10_THROW_ERROR(
          DistNetworkError,
          fmt::format(
              "failed to format addr, unknown family={}", addr.ai_family));
    }
    # 检查地址结构体中地址族是否为IPv4（AF_INET）
    if (addr.ai_addr->sa_family == AF_INET) {
        # 如果是IPv4地址族，使用格式化字符串输出主机和端口号（host:port）
        return fmt::format_to(ctx.out(), "{}:{}", host, port);
    } else {
        # 如果不是IPv4地址族，使用格式化字符串输出带有方括号的IPv6地址和端口号（[host]:port）
        return fmt::format_to(ctx.out(), "[{}]:{}", host, port);
    }
};

// 用于自定义格式化 c10d::detail::SocketImpl 结构体的输出格式
template <>
struct formatter<c10d::detail::SocketImpl> {
  // 解析格式化字符串的上下文
  constexpr decltype(auto) parse(format_parse_context& ctx) const {
    return ctx.begin();
  }

  // 格式化输出 SocketImpl 结构体的内容
  template <typename FormatContext>
  decltype(auto) format(
      const c10d::detail::SocketImpl& socket,
      FormatContext& ctx) const {
    // 创建一个 sockaddr_storage 结构体
    ::sockaddr_storage addr_s{};

    // 转换地址指针类型
    auto addr_ptr = reinterpret_cast<::sockaddr*>(&addr_s);

    // 设置地址长度
    ::socklen_t addr_len = sizeof(addr_s);

    // 获取套接字的文件描述符
    auto fd = socket.handle();

    // 获取套接字的本地地址
    if (::getsockname(fd, addr_ptr, &addr_len) != 0) {
      // 如果获取失败，返回未知字符串
      return fmt::format_to(ctx.out(), "?UNKNOWN?");
    }

    // 创建 addrinfo 结构体并设置地址和长度
    ::addrinfo addr{};
    addr.ai_addr = addr_ptr;
    addr.ai_addrlen = addr_len;

    // 获取远程地址并格式化为字符串
    auto remote = socket.remote();
    std::string remoteStr = remote ? fmt::format("{}", *remote) : "none";

    // 格式化输出整个 SocketImpl 对象的内容
    return fmt::format_to(
        ctx.out(),
        "SocketImpl(fd={}, addr={}, remote={})",
        fd,
        addr,
        remoteStr);
  }
};

// 命名空间 fmt 中的代码结束

} // namespace fmt

// 开始命名空间 c10d::detail 中的代码

// SocketImpl 类的析构函数
SocketImpl::~SocketImpl() {
#ifdef _WIN32
  // Windows 平台关闭套接字
  ::closesocket(hnd_);
#else
  // 非 Windows 平台关闭套接字
  ::close(hnd_);
#endif
}

// 接受客户端连接请求并返回新的 SocketImpl 对象
std::unique_ptr<SocketImpl> SocketImpl::accept() const {
  // 创建 sockaddr_storage 结构体
  ::sockaddr_storage addr_s{};

  // 转换地址指针类型
  auto addr_ptr = reinterpret_cast<::sockaddr*>(&addr_s);

  // 设置地址长度
  ::socklen_t addr_len = sizeof(addr_s);

  // 接受连接请求并返回新的套接字句柄
  Handle hnd = ::accept(hnd_, addr_ptr, &addr_len);
  if (hnd == invalid_socket) {
    // 如果接受失败，处理错误信息并抛出异常
    std::error_code err = getSocketError();
    if (err == std::errc::interrupted) {
      C10_THROW_ERROR(DistNetworkError, std::strerror(err.value()));
    }

    std::string msg{};
    if (err == std::errc::invalid_argument) {
      msg = fmt::format(
          "The server socket on {} is not listening for connections.", *this);
    } else {
      msg = fmt::format(
          "The server socket on {} has failed to accept a connection {}.",
          *this,
          err);
    }

    C10D_ERROR(msg);

    C10D_THROW_ERROR(SocketError, msg);
  }

  // 创建 addrinfo 结构体并设置地址和长度
  ::addrinfo addr{};
  addr.ai_addr = addr_ptr;
  addr.ai_addrlen = addr_len;

  // 记录调试信息
  C10D_DEBUG(
      "The server socket on {} has accepted a connection from {}.",
      *this,
      addr);

  // 创建新的 SocketImpl 对象
  auto impl = std::make_unique<SocketImpl>(hnd, addr);

  // 确保不会将文件描述符泄露给子进程
  impl->closeOnExec();

  // 尝试启用 TCP_NODELAY 选项
  if (!impl->enableNoDelay()) {
    C10D_WARNING(
        "The no-delay option cannot be enabled for the client socket on {}.",
        addr);
  }

  // 返回新创建的 SocketImpl 对象
  return impl;
}

// 设置在 exec 中关闭套接字的选项
void SocketImpl::closeOnExec() noexcept {
#ifndef _WIN32
  ::fcntl(hnd_, F_SETFD, FD_CLOEXEC);
#endif
}

// 启用非阻塞模式
void SocketImpl::enableNonBlocking() {
#ifdef _WIN32
  unsigned long value = 1;
  if (::ioctlsocket(hnd_, FIONBIO, &value) == 0) {
    return;
  }
#else
  int flg = ::fcntl(hnd_, F_GETFL);
  if (flg != -1) {
    if (::fcntl(hnd_, F_SETFL, flg | O_NONBLOCK) == 0) {
      return;
    }
  }
#endif
  // 如果设置失败，则抛出异常
  C10D_THROW_ERROR(
      SocketError, "The socket cannot be switched to non-blocking mode.");
}

// TODO: Remove once we migrate everything to non-blocking mode.
void SocketImpl::disableNonBlocking() {
#ifdef _WIN32
  // 如果是 Windows 平台，设置非阻塞模式为阻塞模式
  unsigned long value = 0;
  if (::ioctlsocket(hnd_, FIONBIO, &value) == 0) {
    return;
  }
#else
  // 如果不是 Windows 平台，在 Linux 平台下获取当前文件描述符的文件状态标志
  int flg = ::fcntl(hnd_, F_GETFL);
  if (flg != -1) {
    // 清除 O_NONBLOCK 标志，将文件描述符设置为阻塞模式
    if (::fcntl(hnd_, F_SETFL, flg & ~O_NONBLOCK) == 0) {
      return;
    }
  }
#endif
  // 如果无法设置为阻塞模式，则抛出异常
  C10D_THROW_ERROR(
      SocketError, "The socket cannot be switched to blocking mode.");
}

bool SocketImpl::enableNoDelay() noexcept {
  // 启用 TCP 的无延迟选项
  return setSocketFlag(IPPROTO_TCP, TCP_NODELAY, true);
}

bool SocketImpl::enableDualStack() noexcept {
  // 启用 IPv6 的双栈支持
  return setSocketFlag(IPPROTO_IPV6, IPV6_V6ONLY, false);
}

#ifndef _WIN32
bool SocketImpl::enableAddressReuse() noexcept {
  // 启用地址重用选项
  return setSocketFlag(SOL_SOCKET, SO_REUSEADDR, true);
}
#endif

#ifdef _WIN32
bool SocketImpl::enableExclusiveAddressUse() noexcept {
  // 启用独占地址使用选项
  return setSocketFlag(SOL_SOCKET, SO_EXCLUSIVEADDRUSE, true);
}
#endif

std::uint16_t SocketImpl::getPort() const {
  // 获取套接字绑定的端口号
  ::sockaddr_storage addr_s{};

  ::socklen_t addr_len = sizeof(addr_s);

  if (::getsockname(hnd_, reinterpret_cast<::sockaddr*>(&addr_s), &addr_len) !=
      0) {
    // 获取失败，抛出异常
    C10D_THROW_ERROR(
        SocketError, "The port number of the socket cannot be retrieved.");
  }

  if (addr_s.ss_family == AF_INET) {
    // 返回 IPv4 地址的端口号
    return ntohs(reinterpret_cast<::sockaddr_in*>(&addr_s)->sin_port);
  } else {
    // 返回 IPv6 地址的端口号
    return ntohs(reinterpret_cast<::sockaddr_in6*>(&addr_s)->sin6_port);
  }
}

bool SocketImpl::setSocketFlag(int level, int optname, bool value) noexcept {
#ifdef _WIN32
  // Windows 平台下，根据 value 设置选项值为 TRUE 或 FALSE
  auto buf = value ? TRUE : FALSE;
#else
  // 非 Windows 平台下，根据 value 设置选项值为 1 或 0
  auto buf = value ? 1 : 0;
#endif
  // 设置套接字选项
  return setSocketOption(hnd_, level, optname, &buf, sizeof(buf)) == 0;
}

bool SocketImpl::waitForInput(std::chrono::milliseconds timeout) {
  // 等待套接字可读事件发生，超时时间为 timeout
  using Clock = std::chrono::steady_clock;

  auto deadline = Clock::now() + timeout;
  do {
    ::pollfd pfd{};
    pfd.fd = hnd_;
    pfd.events = POLLIN;

    // 调用系统的 poll 函数等待套接字可读事件
    int res = pollFd(&pfd, 1, static_cast<int>(timeout.count()));
    if (res > 0) {
      // 如果有事件发生，返回 true
      return true;
    } else if (res == 0) {
      // 超时警告
      C10D_WARNING(
          "waitForInput: poll for socket {} returned 0, likely a timeout",
          *this);
      continue;
    }

    // 获取套接字错误码
    std::error_code err = getSocketError();
    if (err == std::errc::operation_in_progress) {
      bool timedout = Clock::now() >= deadline;
      if (timedout) {
        // 超时，返回 false
        return false;
      }
      // 操作仍在进行中的警告
      C10D_WARNING(
          "waitForInput: poll for socket {} returned operation_in_progress before a timeout",
          *this);
    } else if (err != std::errc::interrupted) {
      // 其他错误警告
      C10D_WARNING(
          "waitForInput: poll for socket {} failed with res={}, err={}.",
          *this,
          res,
          err);
      return false;
    }
  } while (Clock::now() < deadline);

  // 超时警告
  C10D_WARNING(
      "waitForInput: socket {} timed out after {}ms", *this, timeout.count());
  return false;
}

namespace {

struct addrinfo_delete {
  void operator()(::addrinfo* addr) const noexcept {
    ::freeaddrinfo(addr);
  }
};
// 使用别名 addrinfo_ptr 表示 std::unique_ptr<::addrinfo, addrinfo_delete>
using addrinfo_ptr = std::unique_ptr<::addrinfo, addrinfo_delete>;

// SocketListenOp 类定义，用于执行 socket 监听操作
class SocketListenOp {
 public:
  // 构造函数，初始化监听端口和选项
  SocketListenOp(std::uint16_t port, const SocketOptions& opts);

  // 执行监听操作，返回 socket 对象的 unique_ptr
  std::unique_ptr<SocketImpl> run();

 private:
  // 尝试在指定协议族下进行监听
  bool tryListen(int family);

  // 尝试在给定的 addrinfo 地址信息下进行监听
  bool tryListen(const ::addrinfo& addr);

  // 模板函数，记录错误消息到 errors_ 容器中
  template <typename... Args>
  // 禁止 LINT 项: cppcoreguidelines-missing-std-forward
  void recordError(fmt::string_view format, Args&&... args) {
    auto msg = fmt::vformat(format, fmt::make_format_args(args...));

    // 记录警告消息
    C10D_WARNING(msg);

    // 将错误消息添加到 errors_ 容器中
    errors_.emplace_back(std::move(msg));
  }

  std::string port_;                    // 监听端口号
  const SocketOptions* opts_;           // 指向 SocketOptions 的指针
  std::vector<std::string> errors_{};   // 存储错误消息的容器
  std::unique_ptr<SocketImpl> socket_{}; // socket 对象的 unique_ptr
};

// SocketListenOp 构造函数实现
SocketListenOp::SocketListenOp(std::uint16_t port, const SocketOptions& opts)
    : port_{fmt::to_string(port)}, opts_{&opts} {}

// SocketListenOp 类的 run 方法实现，执行监听操作
std::unique_ptr<SocketImpl> SocketListenOp::run() {
  // 如果 prefer_ipv6 返回 true，则尝试在 IPv6 地址上进行监听
  if (opts_->prefer_ipv6()) {
    C10D_DEBUG("The server socket will attempt to listen on an IPv6 address.");
    if (tryListen(AF_INET6)) {
      return std::move(socket_);
    }

    // 尝试在 IPv4 地址上进行监听
    C10D_DEBUG("The server socket will attempt to listen on an IPv4 address.");
    if (tryListen(AF_INET)) {
      return std::move(socket_);
    }
  } else {
    // 否则尝试在 IPv4 或 IPv6 地址上进行监听
    C10D_DEBUG(
        "The server socket will attempt to listen on an IPv4 or IPv6 address.");
    if (tryListen(AF_UNSPEC)) {
      return std::move(socket_);
    }
  }

  // 如果监听失败，构造错误消息并抛出异常
  constexpr auto* msg =
      "The server socket has failed to listen on any local network address.";

  C10D_ERROR(msg);

  C10D_THROW_ERROR(
      SocketError, fmt::format("{} {}", msg, fmt::join(errors_, " ")));
}

// SocketListenOp 类的 tryListen(int family) 方法实现
bool SocketListenOp::tryListen(int family) {
  ::addrinfo hints{}, *naked_result = nullptr;

  hints.ai_flags = AI_PASSIVE | AI_NUMERICSERV;  // 设置地址信息标志
  hints.ai_family = family;                     // 设置地址族
  hints.ai_socktype = SOCK_STREAM;              // 设置套接字类型

  // 获取本地地址信息
  int r = ::getaddrinfo(nullptr, port_.c_str(), &hints, &naked_result);
  if (r != 0) {
    const char* gai_err = ::gai_strerror(r);

    // 记录错误信息到 errors_ 容器中
    recordError(
        "The local {}network addresses cannot be retrieved (gai error: {} - {}).",
        family == AF_INET        ? "IPv4 "
            : family == AF_INET6 ? "IPv6 "
                                 : "",
        r,
        gai_err);

    return false;
  }

  addrinfo_ptr result{naked_result};  // 使用 unique_ptr 管理 addrinfo 结果的生命周期

  // 遍历获取到的地址信息列表，尝试进行监听
  for (::addrinfo* addr = naked_result; addr != nullptr; addr = addr->ai_next) {
    C10D_DEBUG("The server socket is attempting to listen on {}.", *addr);
    if (tryListen(*addr)) {
      return true;
    }
  }

  return false;
}

// SocketListenOp 类的 tryListen(const ::addrinfo& addr) 方法实现
bool SocketListenOp::tryListen(const ::addrinfo& addr) {
  // 创建套接字
  SocketImpl::Handle hnd =
      ::socket(addr.ai_family, addr.ai_socktype, addr.ai_protocol);
  if (hnd == SocketImpl::invalid_socket) {
    // 记录套接字初始化失败的错误信息
    recordError(
        "The server socket cannot be initialized on {} {}.",
        addr,
        getSocketError());

    return false;
  }

  socket_ = std::make_unique<SocketImpl>(hnd);  // 创建 socket 对象的 unique_ptr

#ifndef _WIN32
  // 在非 Windows 平台下尝试启用地址重用
  if (!socket_->enableAddressReuse()) {
    C10D_WARNING(
        "The address reuse option cannot be enabled for the server socket on {}.",
        addr);



// 调用 C10D_WARNING 宏，输出警告信息，指示服务器套接字上不能启用地址重用选项
#endif

#ifdef _WIN32
  // 在 Windows 系统中，SO_REUSEADDR 标志与 Unix-like 系统有显著不同的行为。
  // 它允许两个或多个进程同时共享相同的端口，这是非常不安全的。
  //
  // 在这里，我们遵循 Microsoft 的建议，使用非标准的 SO_EXCLUSIVEADDRUSE 标志。
  if (!socket_->enableExclusiveAddressUse()) {
    C10D_WARNING(
        "The exclusive address use option cannot be enabled for the server socket on {}.",
        addr);
  }
#endif

  // 并非所有操作系统默认支持双栈套接字。因为我们希望使用 IPv6 套接字进行 IPv4 通信，
  // 我们显式地要求系统启用它。
  if (addr.ai_family == AF_INET6 && !socket_->enableDualStack()) {
    C10D_WARNING(
        "The server socket does not support IPv4 communication on {}.", addr);
  }

  // 将套接字绑定到指定地址和端口
  if (::bind(socket_->handle(), addr.ai_addr, addr.ai_addrlen) != 0) {
    recordError(
        "The server socket has failed to bind to {} {}.",
        addr,
        getSocketError());

    return false;
  }

  // 开始监听连接请求，-1 表示允许任意大小的连接队列
  // NOLINTNEXTLINE(bugprone-argument-comment)
  if (::listen(socket_->handle(), -1 /* backlog */) != 0) {
    recordError(
        "The server socket has failed to listen on {} {}.",
        addr,
        getSocketError());

    return false;
  }

  // 在执行新进程时关闭套接字
  socket_->closeOnExec();

  // 记录服务器套接字已经开始监听的信息
  C10D_INFO("The server socket has started to listen on {}.", addr);

  return true;
}

class SocketListenFromFdOp {
 public:
  SocketListenFromFdOp(int fd, std::uint16_t expected_port);

  std::unique_ptr<SocketImpl> run() const;

 private:
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  const int fd_;
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  const std::uint16_t expected_port_;
};

SocketListenFromFdOp::SocketListenFromFdOp(int fd, std::uint16_t expected_port)
    : fd_(fd), expected_port_(expected_port) {}

std::unique_ptr<SocketImpl> SocketListenFromFdOp::run() const {
  C10D_DEBUG("listenFromFd: fd {}, expected port {}", fd_, expected_port_);

  // 获取套接字的本地地址信息
  ::sockaddr_storage addr_storage{};
  ::socklen_t addr_len = sizeof(addr_storage);
  if (::getsockname(
          fd_, reinterpret_cast<::sockaddr*>(&addr_storage), &addr_len) < 0) {
    C10D_THROW_ERROR(
        SocketError,
        fmt::format("getsockname failed for fd {}: {}", fd_, getSocketError()));
  }

  // 创建套接字实例并验证端口号是否与期望的端口号一致
  auto socket = std::make_unique<SocketImpl>(fd_);
  const auto port = socket->getPort();

  if (port != expected_port_) {
    C10D_THROW_ERROR(
        SocketError,
        fmt::format(
            "listen fd {} is bound to port {}, expected to be bound to port {}",
            fd_,
            port,
            expected_port_));
  }

  // 开始监听连接请求，-1 表示允许任意大小的连接队列
  if (::listen(socket->handle(), -1 /* backlog */) != 0) {

        recordError(
            "Failed to listen on socket fd {} {}: {}",
            fd_,
            port,
            getSocketError());
  
    // 如果监听失败，则抛出异常
    C10D_THROW_ERROR(
        SocketError,
        fmt::format("Failed to listen on socket fd {} {}: {}", fd_, port, getSocketError()));
  }

  // 返回创建的套接字实例
  return socket;
}
    # 抛出 SocketError 异常，包含错误信息
    C10D_THROW_ERROR(
        SocketError,
        fmt::format(
            "Failed to listen on socket initialized from fd {}: {}.",
            socket->handle(),
            getSocketError()));
  }

  # 在执行时关闭 socket
  socket->closeOnExec();

  # 输出信息，表示服务器已接管监听 socket
  C10D_INFO(
      "The server has taken over the listening socket with fd {}, address {}",
      fd_,
      *socket);
  # 返回 socket
  return socket;
// 结束前一个代码块的类定义
}

// SocketConnectOp 类的实现
class SocketConnectOp {
  using Clock = std::chrono::steady_clock;  // 使用 steady_clock 作为时钟
  using Duration = std::chrono::steady_clock::duration;  // 定义持续时间类型
  using TimePoint = std::chrono::time_point<std::chrono::steady_clock>;  // 定义时间点类型

  enum class ConnectResult : uint8_t { Success, Error, Retry };  // 定义连接结果枚举类，包括成功、错误和重试

public:
  // 构造函数，初始化 SocketConnectOp 实例
  SocketConnectOp(
      const std::string& host,
      std::uint16_t port,
      const SocketOptions& opts);

  // 运行连接操作，返回一个唯一指针指向 SocketImpl 对象
  std::unique_ptr<SocketImpl> run();

private:
  // 尝试连接指定地址族
  bool tryConnect(int family);

  // 尝试连接特定地址信息结构体
  ConnectResult tryConnect(const ::addrinfo& addr);

  // 实际连接核心功能的尝试
  ConnectResult tryConnectCore(const ::addrinfo& addr);

  // 抛出超时错误的方法，标记为不会返回
  [[noreturn]] void throwTimeoutError() const;

  // 记录错误信息的模板函数，接受格式化字符串和参数
  template <typename... Args>
  // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
  void recordError(fmt::string_view format, Args&&... args) {
    auto msg = fmt::vformat(format, fmt::make_format_args(args...));

    C10D_WARNING(msg);  // 发出警告信息

    errors_.emplace_back(std::move(msg));  // 将错误信息添加到 errors_ 容器中
  }

  const char* host_;  // 主机名 C 字符串指针
  std::string port_;  // 端口号字符串
  const SocketOptions* opts_;  // 指向 SocketOptions 对象的指针
  TimePoint deadline_{};  // 连接超时时间点，默认初始化
  std::vector<std::string> errors_{};  // 错误信息列表
  std::unique_ptr<SocketImpl> socket_{};  // 唯一指针指向 SocketImpl 对象
};

// SocketConnectOp 构造函数的实现，初始化主机名、端口号和选项
SocketConnectOp::SocketConnectOp(
    const std::string& host,
    std::uint16_t port,
    const SocketOptions& opts)
    : host_{host.c_str()}, port_{fmt::to_string(port)}, opts_{&opts} {}

// SocketConnectOp 类的 run 方法实现，执行连接操作
std::unique_ptr<SocketImpl> SocketConnectOp::run() {
  if (opts_->prefer_ipv6()) {
    // 如果优先使用 IPv6，记录调试信息
    C10D_DEBUG(
        "The client socket will attempt to connect to an IPv6 address of ({}, {}).",
        host_,
        port_);

    // 尝试连接 IPv6 地址
    if (tryConnect(AF_INET6)) {
      return std::move(socket_);
    }

    // 记录调试信息，尝试连接 IPv4 地址
    C10D_DEBUG(
        "The client socket will attempt to connect to an IPv4 address of ({}, {}).",
        host_,
        port_);

    // 尝试连接 IPv4 地址
    if (tryConnect(AF_INET)) {
      return std::move(socket_);
    }
  } else {
    // 如果不优先使用 IPv6，记录调试信息，尝试连接 IPv4 或 IPv6 地址
    C10D_DEBUG(
        "The client socket will attempt to connect to an IPv4 or IPv6 address of ({}, {}).",
        host_,
        port_);

    // 尝试连接未指定地址族
    if (tryConnect(AF_UNSPEC)) {
      return std::move(socket_);
    }
  }

  // 如果连接失败，生成错误信息并抛出异常
  auto msg = fmt::format(
      "The client socket has failed to connect to any network address of ({}, {}).",
      host_,
      port_);

  C10D_ERROR(msg);  // 记录错误信息

  // 抛出 SocketError 异常，包含详细错误信息
  C10D_THROW_ERROR(
      SocketError, fmt::format("{} {}", msg, fmt::join(errors_, " ")));
}

// 尝试连接指定地址族的方法实现
bool SocketConnectOp::tryConnect(int family) {
  ::addrinfo hints{};  // 初始化地址信息结构体
  hints.ai_flags = AI_V4MAPPED | AI_ALL | AI_NUMERICSERV;  // 设置地址信息标志
  hints.ai_family = family;  // 指定地址族
  hints.ai_socktype = SOCK_STREAM;  // 指定套接字类型为流式套接字

  deadline_ = Clock::now() + opts_->connect_timeout();  // 设置连接超时时间点

  bool retry; // NOLINT(cppcoreguidelines-init-variables)
  do {
    retry = false;  // 初始化重试标志为 false

    errors_.clear();  // 清空错误信息列表

    ::addrinfo* naked_result = nullptr;  // 定义裸指针，存放解析结果
    // patternlint-disable cpp-dns-deps
    int r = ::getaddrinfo(host_, port_.c_str(), &hints, &naked_result);  // 解析主机名和端口号
    // 如果获取地址出错，则记录错误信息并设置重试标志
    if (r != 0) {
      // 获取 gai_strerror 返回的错误信息
      const char* gai_err = ::gai_strerror(r);

      // 记录错误信息到日志，包括主机名、端口号、错误码和错误信息
      recordError(
          "The {}network addresses of ({}, {}) cannot be retrieved (gai error: {} - {}).",
          family == AF_INET        ? "IPv4 "
              : family == AF_INET6 ? "IPv6 "
                                   : "",
          host_,
          port_,
          r,
          gai_err);
      // 设置重试标志，表示将会尝试重新连接
      retry = true;
    } else {
      // 如果获取地址成功，则尝试连接每一个返回的地址
      addrinfo_ptr result{naked_result};

      // 遍历返回的地址列表
      for (::addrinfo* addr = naked_result; addr != nullptr;
           addr = addr->ai_next) {
        // 记录尝试连接的地址信息到日志
        C10D_TRACE("The client socket is attempting to connect to {}.", *addr);

        // 尝试连接当前地址
        ConnectResult cr = tryConnect(*addr);
        // 如果连接成功，直接返回 true
        if (cr == ConnectResult::Success) {
          return true;
        }

        // 如果连接返回需要重试，则设置重试标志
        if (cr == ConnectResult::Retry) {
          retry = true;
        }
      }
    }

    // 如果需要重试连接
    if (retry) {
      // 获取连接重试的间隔时间
      auto connectBackoff = opts_->connect_backoff();
      auto delayDuration = connectBackoff->nextBackoff();

      // 如果当前时间还未超过连接超时时间，则进行重试
      if (Clock::now() < deadline_ - delayDuration) {
        // 控制日志输出频率，每 30 秒输出一次警告信息
        static auto lastLog = std::chrono::steady_clock::now();
        auto now = std::chrono::steady_clock::now();
        if ((now - lastLog) >= std::chrono::seconds(30)) {
          // 记录警告信息到日志，指示将会重试连接
          C10D_INFO(
              "No socket on ({}, {}) is listening yet, will retry.",
              host_,
              port_);

          lastLog = now;
        }

        // 等待一段时间，以避免过多负荷请求服务器
        delay(delayDuration);
      } else {
        // 如果超过连接超时时间仍未连接成功，则抛出超时错误
        throwTimeoutError();
      }
    }
  } while (retry);

  // 若循环结束仍未成功连接，则返回 false
  return false;
// 尝试建立与指定地址的连接操作
SocketConnectOp::ConnectResult SocketConnectOp::tryConnect(
    const ::addrinfo& addr) {
  // 如果当前时间超过连接截止时间，则抛出超时错误
  if (Clock::now() >= deadline_) {
    throwTimeoutError();
  }

  // 创建一个套接字，使用给定地址信息中的协议族、套接字类型和协议
  SocketImpl::Handle hnd =
      ::socket(addr.ai_family, addr.ai_socktype, addr.ai_protocol);
  // 如果无法创建套接字，记录错误信息并返回连接错误
  if (hnd == SocketImpl::invalid_socket) {
    recordError(
        "The client socket cannot be initialized to connect to {} {}.",
        addr,
        getSocketError());

    return ConnectResult::Error;
  }

  // 使用套接字句柄和地址信息创建 SocketImpl 对象
  socket_ = std::make_unique<SocketImpl>(hnd, addr);

  // 启用非阻塞模式
  socket_->enableNonBlocking();

  // 调用 tryConnectCore 方法尝试连接到指定地址
  ConnectResult cr = tryConnectCore(addr);
  // 如果连接操作返回错误，处理具体错误情况
  if (cr == ConnectResult::Error) {
    std::error_code err = getSocketError();
    // 如果是被中断错误，抛出 DistNetworkError 异常
    if (err == std::errc::interrupted) {
      C10_THROW_ERROR(DistNetworkError, std::strerror(err.value()));
    }

    // 如果是连接被拒绝或重置错误，返回重试状态
    if (err == std::errc::connection_refused ||
        err == std::errc::connection_reset) {
      C10D_TRACE(
          "The server socket on {} is not yet listening {}, will retry.",
          addr,
          err);

      return ConnectResult::Retry;
    } else {
      // 记录连接错误信息并返回连接错误状态
      recordError(
          "The client socket has failed to connect to {} {}.", addr, err);

      return ConnectResult::Error;
    }
  }

  // 在执行后关闭套接字的执行权限
  socket_->closeOnExec();

  // TODO: Remove once we fully migrate to non-blocking mode.
  // 在完全迁移到非阻塞模式后移除这段代码

  // 禁用非阻塞模式
  socket_->disableNonBlocking();

  // 记录成功的连接信息
  C10D_INFO("The client socket has connected to {} on {}.", addr, *socket_);

  // 如果无法开启 TCP_NODELAY 选项，记录警告信息
  if (!socket_->enableNoDelay()) {
    C10D_WARNING(
        "The no-delay option cannot be enabled for the client socket on {}.",
        *socket_);
  }

  // 返回连接成功状态
  return ConnectResult::Success;
}

// 尝试核心连接方法，返回连接结果状态
SocketConnectOp::ConnectResult SocketConnectOp::tryConnectCore(
    const ::addrinfo& addr) {
  // 尝试连接到指定地址的套接字，返回连接状态
  int r = ::connect(socket_->handle(), addr.ai_addr, addr.ai_addrlen);
  // 如果连接成功，返回连接成功状态
  if (r == 0) {
    return ConnectResult::Success;
  }

  // 获取套接字错误码
  std::error_code err = getSocketError();
  // 如果已经连接，则返回连接成功状态
  if (err == std::errc::already_connected) {
    return ConnectResult::Success;
  }

  // 如果操作正在进行中或即将阻塞，则返回连接错误状态
  if (err != std::errc::operation_in_progress &&
      err != std::errc::operation_would_block) {
    return ConnectResult::Error;
  }

  // 计算剩余时间
  Duration remaining = deadline_ - Clock::now();
  // 如果剩余时间小于等于零，抛出超时错误
  if (remaining <= Duration::zero()) {
    throwTimeoutError();
  }

  // 构造 pollfd 结构体
  ::pollfd pfd{};
  pfd.fd = socket_->handle();
  pfd.events = POLLOUT;

  // 转换剩余时间为毫秒
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(remaining);

  // 调用 pollFd 函数进行套接字轮询
  r = pollFd(&pfd, 1, static_cast<int>(ms.count()));
  // 如果轮询超时，抛出超时错误
  if (r == 0) {
    throwTimeoutError();
  }
  // 如果轮询失败，返回连接错误状态
  if (r == -1) {
    return ConnectResult::Error;
  }

  // 初始化错误码变量
  int err_code = 0;
  // 错误码长度
  ::socklen_t err_len = sizeof(int);

  // 获取套接字选项错误码
  r = getSocketOption(
      socket_->handle(), SOL_SOCKET, SO_ERROR, &err_code, &err_len);
  // 如果获取失败，返回连接错误状态
  if (r != 0) {
    return ConnectResult::Error;
  }

  // 如果错误码不为零，设置套接字错误并返回连接错误状态；否则返回连接成功状态
  if (err_code != 0) {
    setSocketError(err_code);

    return ConnectResult::Error;
  } else {
    return ConnectResult::Success;
  }
}
void SocketConnectOp::throwTimeoutError() const {
  // 格式化错误消息，指明连接超时的详细信息
  auto msg = fmt::format(
      "The client socket has timed out after {} while trying to connect to ({}, {}).",
      opts_->connect_timeout(),
      host_,
      port_);

  // 抛出连接超时错误并记录消息
  C10D_ERROR(msg);

  // 抛出连接超时异常，并携带错误消息
  C10D_THROW_ERROR(TimeoutError, msg);
}

} // namespace

void Socket::initialize() {
#ifdef _WIN32
  static c10::once_flag init_flag{};

  // 在 Windows 上初始化 Winsock 库，确保在使用 socket 函数前进行初始化
  c10::call_once(init_flag, []() {
    WSADATA data{};
    // 尝试初始化 Winsock，版本号 2.2
    if (::WSAStartup(MAKEWORD(2, 2), &data) != 0) {
      // 如果初始化失败，则抛出 SocketError 异常
      C10D_THROW_ERROR(
          SocketError, "The initialization of Winsock has failed.");
    }
  });
#endif
}

Socket Socket::listen(std::uint16_t port, const SocketOptions& opts) {
  // 创建一个监听操作对象，并运行它以获取实际的 Socket 对象
  SocketListenOp op{port, opts};

  return Socket{op.run()};
}

Socket Socket::listenFromFd(int fd, std::uint16_t expected_port) {
  // 创建一个从文件描述符监听的操作对象，并运行它以获取实际的 Socket 对象
  SocketListenFromFdOp op{fd, expected_port};

  return Socket{op.run()};
}

Socket Socket::connect(
    const std::string& host,
    std::uint16_t port,
    const SocketOptions& opts) {
  // 创建一个连接操作对象，并运行它以获取实际的 Socket 对象
  SocketConnectOp op{host, port, opts};

  return Socket{op.run()};
}

Socket::Socket(Socket&& other) noexcept = default;

Socket& Socket::operator=(Socket&& other) noexcept = default;

Socket::~Socket() = default;

Socket Socket::accept() const {
  // 如果 Socket 对象已初始化，则调用底层实现的 accept 方法获取新的 Socket 对象
  if (impl_) {
    return Socket{impl_->accept()};
  }

  // 如果 Socket 对象未初始化，则抛出 SocketError 异常
  C10D_THROW_ERROR(SocketError, "The socket is not initialized.");
}

int Socket::handle() const noexcept {
  // 如果 Socket 对象已初始化，则返回底层实现的句柄
  if (impl_) {
    return impl_->handle();
  }
  // 如果未初始化，则返回无效的句柄值
  return SocketImpl::invalid_socket;
}

std::uint16_t Socket::port() const {
  // 如果 Socket 对象已初始化，则获取底层实现的端口号
  if (impl_) {
    return impl_->getPort();
  }
  // 如果未初始化，则返回 0
  return 0;
}

Socket::Socket(std::unique_ptr<SocketImpl>&& impl) noexcept
    : impl_{std::move(impl)} {}

bool Socket::waitForInput(std::chrono::milliseconds timeout) {
  // 调用底层实现的方法等待输入，指定超时时间
  return impl_->waitForInput(timeout);
}

std::string Socket::repr() const {
  // 如果 Socket 对象已初始化，则返回其底层实现的字符串表示
  if (impl_) {
    return fmt::format("{}", *impl_);
  }
  // 如果未初始化，则返回默认字符串表示
  return "Socket(no-impl)";
}

} // namespace c10d::detail
```