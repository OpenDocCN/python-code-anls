# `.\pytorch\torch\csrc\distributed\c10d\TCPStore.cpp`

```
// 包含必要的头文件
#include <c10/util/irange.h>
#include <fmt/format.h>
#include <torch/csrc/distributed/c10d/Backoff.hpp>
#include <torch/csrc/distributed/c10d/TCPStore.hpp>
#include <torch/csrc/distributed/c10d/TCPStoreBackend.hpp>
#include <torch/csrc/distributed/c10d/logging.h>

#include <fcntl.h>           // 提供对文件控制的功能
#include <chrono>            // 提供时间点和持续时间计算
#include <fstream>           // 提供文件输入输出操作
#include <random>            // 提供随机数生成器
#include <thread>            // 提供线程支持
#include <unordered_map>     // 提供无序映射容器
#include <utility>           // 提供通用工具

#ifdef _WIN32
#include <io.h>              // 提供对文件和输入输出设备的访问
#include <winsock2.h>        // Windows 平台的套接字编程接口
#else
#include <poll.h>            // 提供对文件描述符的轮询操作
#include <unistd.h>          // 提供对 POSIX 系统 API 的访问
#endif

#ifdef _WIN32
#include <torch/csrc/distributed/c10d/WinSockUtils.hpp>
#else
#include <torch/csrc/distributed/c10d/UnixSockUtils.hpp>
#endif

#include <torch/csrc/distributed/c10d/socket.h>  // 提供与套接字相关的功能

namespace c10d {
namespace detail {

// 一个计时器，用于测量时间和更新计数
class timing_guard {
  Counter& counter_;   // 计数器对象的引用
  typedef std::chrono::time_point<std::chrono::high_resolution_clock>
      time_point;
  time_point start_;   // 计时开始的时间点

 public:
  timing_guard(Counter& counter)
      : counter_(counter), start_(std::chrono::high_resolution_clock::now()) {}

  ~timing_guard() {
    stop();   // 析构函数，在对象销毁时停止计时
  }

  void stop() {
    if (start_ != time_point()) {   // 如果计时已经开始
      auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(
                      std::chrono::high_resolution_clock::now() - start_)
                      .count();   // 计算时间差并转换为毫秒
      counter_.update(diff);   // 更新计数器的值
      start_ = time_point();   // 重置开始时间点
    }
  }
};

// 计数器类，用于记录计数和统计计数时间的统计量
void Counter::update(double val) {
  count_ += 1;   // 增加计数器值

  auto delta = val - mean_;   // 计算值与平均值的差异
  mean_ += delta / count_;    // 更新平均值

  auto delta2 = val - mean_;  // 计算更新后的值与平均值的差异
  m2_ += delta2 * delta2;     // 更新平方差的累积和
}

// 返回计数器的观测结果，包括计数、平均值和样本方差
std::unordered_map<std::string, double> Counter::observe() const {
  std::unordered_map<std::string, double> res;   // 创建结果的无序映射
  res["count"] = (double)count_;   // 记录计数的值
  res["mean"] = mean_;             // 记录平均值

  if (count_ >= 2) {
    res["sample_variance"] = m2_ / (count_ - 1);   // 如果计数大于等于2，记录样本方差
  } else {
    res["sample_variance"] = std::nan("1");        // 否则记录非数值
  }
  return res;   // 返回结果
}

// 管理服务器守护程序的生命周期
class TCPServer {
 public:
  static std::shared_ptr<TCPServer> start(const TCPStoreOptions& opts);   // 启动服务器的静态方法

  std::uint16_t port() const noexcept {   // 返回服务器端口号的方法
    return port_;   // 返回存储的端口号
  }

  explicit TCPServer(
      std::uint16_t port,
      std::unique_ptr<BackgroundThread>&& daemon)
      : port_{port}, daemon_{std::move(daemon)} {}   // 构造函数，初始化端口号和后台线程

 private:
  std::uint16_t port_;                          // 存储端口号
  std::unique_ptr<BackgroundThread> daemon_;    // 存储后台线程的唯一指针

  // 对于所有请求多租户的 TCPServer，我们存储弱引用
  static std::unordered_map<std::uint16_t, std::weak_ptr<TCPServer>>
      cachedServers_;   // 存储服务器的静态弱引用映射

  static std::mutex cache_mutex_;   // 用于同步访问缓存映射的互斥量
};

std::unordered_map<std::uint16_t, std::weak_ptr<TCPServer>>
    TCPServer::cachedServers_{};   // 初始化静态缓存映射

std::mutex TCPServer::cache_mutex_{};   // 初始化静态互斥量

// 开始 TCP 服务器的方法，根据选项选择创建 LibUV 或者 TCP 后端
std::shared_ptr<TCPServer> TCPServer::start(const TCPStoreOptions& opts) {
  auto startCore = [&opts]() {   // 创建启动核心的 lambda 函数
    auto daemon = opts.useLibUV ? create_libuv_tcpstore_backend(opts)
                                : create_tcpstore_backend(opts);   // 根据选项创建后端对象
    daemon->start();   // 启动后端服务
    return std::make_shared<TCPServer>(daemon->port(), std::move(daemon));
  };

  std::shared_ptr<TCPServer> server{};

  if (opts.multiTenant) {
    std::lock_guard<std::mutex> guard{cache_mutex_};

    // 如果调用者允许使用多租户存储，首先检查是否已经有一个 TCPServer 在指定端口上运行。
    if (opts.port > 0) {
      // 在缓存的服务器列表中查找指定端口的服务器
      auto pos = cachedServers_.find(opts.port);
      if (pos != cachedServers_.end()) {
        // 尝试获取指向 TCPServer 的弱引用
        server = pos->second.lock();
        if (server != nullptr) {
          // 如果服务器仍然有效，直接返回该服务器
          return server;
        }

        // 看起来 TCPServer 已经被销毁，确保释放控制块
        cachedServers_.erase(pos);
      }
    }

    // 启动核心功能，返回新创建的 TCPServer
    server = startCore();

    // 将新创建的服务器加入缓存列表
    cachedServers_.emplace(server->port(), server);
  } else {
    // 单租户模式下，直接启动核心功能并返回新创建的 TCPServer
    server = startCore();
  }

  // 返回启动或者从缓存获取的 TCPServer
  return server;
  // 如果缓冲区大小达到 FLUSH_WATERMARK，调用 flush() 来发送数据
  void maybeFlush() {
    if (buffer.size() >= FLUSH_WATERMARK) {
      flush();
    }
  }

public:
  // SendBuffer 构造函数，初始化 SendBuffer 对象
  SendBuffer(detail::TCPClient& client, detail::QueryType cmd)
      : client(client) {
    // 预留 32 字节的缓冲区空间，通常足够存放大多数指令
    buffer.reserve(32);
    // 将指令转换为 uint8_t 类型添加到缓冲区中
    buffer.push_back((uint8_t)cmd);
  }

  // 向缓冲区中添加字符串的长度和内容
  void appendString(const std::string& str) {
    // 调用模板方法，将字符串的长度添加到缓冲区
    appendValue<uint64_t>(str.size());
    // 将字符串内容添加到缓冲区
    // 将字符串 `str` 的内容从其开始到结束插入到缓冲区的末尾
    buffer.insert(buffer.end(), str.begin(), str.end());
    // 可能触发缓冲区刷新操作
    maybeFlush();
  }

  void appendBytes(const std::vector<uint8_t>& vec) {
    // 在缓冲区末尾添加向量 `vec` 的大小作为 uint64_t 类型的值
    appendValue<uint64_t>(vec.size());
    // 将向量 `vec` 的内容从其开始到结束插入到缓冲区的末尾
    buffer.insert(buffer.end(), vec.begin(), vec.end());
    // 可能触发缓冲区刷新操作
    maybeFlush();
  }

  template <typename T>
  void appendValue(T value) {
    // 获取值 `value` 的起始地址，并将以字节为单位的值插入到缓冲区的末尾
    uint8_t* begin = (uint8_t*)&value;
    buffer.insert(buffer.end(), begin, begin + sizeof(T));
    // 可能触发缓冲区刷新操作
    maybeFlush();
  }

  void flush() {
    // 如果缓冲区不为空，则将缓冲区中的数据发送给客户端，并清空缓冲区
    if (!buffer.empty()) {
      client.sendRaw(buffer.data(), buffer.size());
      buffer.clear();
    }
  }
};

} // namespace detail

using detail::Socket;

// TCPStore class methods
TCPStore::TCPStore(
    const std::string& masterAddr,                   // 构造函数，接收主地址
    std::uint16_t masterPort,                        // 主端口号
    std::optional<int> numWorkers,                   // 可选的工作进程数量
    bool isServer,                                   // 是否为服务器端
    const std::chrono::milliseconds& timeout,        // 超时时间
    bool waitWorkers)                                // 是否等待工作进程
    : TCPStore{                                      // 委托构造函数，调用另一个构造函数
          masterAddr,
          TCPStoreOptions{                           // 创建TCPStoreOptions对象
              masterPort,
              isServer,
              numWorkers ? std::optional<std::size_t>(*numWorkers)
                         : c10::nullopt,            // 如果有工作进程数量，转换为std::size_t；否则为空
              waitWorkers,
              timeout}} {}

TCPStore::TCPStore(std::string host, const TCPStoreOptions& opts)
    : Store{opts.timeout},                          // 调用基类Store的构造函数
      addr_{std::move(host)},                       // 移动赋值主机地址
      numWorkers_{opts.numWorkers},                 // 设置工作进程数量
      usingLibUv_{opts.useLibUV} {                  // 是否使用LibUV

  if (opts.useLibUV) {                             // 如果使用LibUV
    TORCH_CHECK(
        ::c10d::detail::is_libuv_tcpstore_backend_available(),  // 检查LibUV后端是否可用
        "use_libuv was requested but PyTorch was build without libuv support");  // 如果没有支持LibUV，则报错

    if (opts.masterListenFd.has_value()) {
      // TODO(xilunwu): support this init method after testing
      constexpr auto* msg =
          "The libuv TCPStore backend does not support initialization with an listen fd. "
          "Please switch to the legacy TCPStore by setting environment variable USE_LIBUV "
          "to \"0\".";
      C10D_ERROR(msg);                            // 报错
      C10_THROW_ERROR(NotImplementedError, msg);  // 抛出未实现的错误
      return;                                     // 返回
    }
  }

  Socket::initialize();                           // 初始化Socket

  if (opts.isServer) {                            // 如果是服务器端
    server_ = detail::TCPServer::start(opts);     // 启动TCP服务器
    // server successfully started
    C10D_DEBUG("The server has started on port = {}.", server_->port());  // 调试信息，服务器成功启动

    std::ifstream maxconnFile("/proc/sys/net/core/somaxconn");  // 打开系统文件，获取最大连接数
    if (maxconnFile.good() && numWorkers_.has_value()) {  // 如果文件可用且工作进程数量有值
      try {
        std::string str(
            (std::istreambuf_iterator<char>(maxconnFile)),
            std::istreambuf_iterator<char>());
        std::size_t somaxconn = std::stoll(str);   // 将文件内容转换为整数
        if (somaxconn < *numWorkers_) {
          C10D_WARNING(
              "Starting store with {} workers but somaxconn is {}."
              "This might cause instability during bootstrap, consider increasing it.",
              *numWorkers_,
              somaxconn);                          // 发出警告，工作进程数大于最大连接数可能导致不稳定
        }
      } catch (std::logic_error& e) {
        C10D_INFO("failed to parse somaxconn proc file due to {}", e.what());  // 解析错误，记录信息
      }
    }

    addr_.port = server_->port();                  // 设置地址端口为服务器端口
  } else {
    addr_.port = opts.port;                        // 否则，设置地址端口为指定端口
  }

  // Try connecting several times -- if the server listen backlog is full it may
  // fail on the first send in validate.
  auto deadline = std::chrono::steady_clock::now() + opts.timeout;  // 计算连接超时时间
  auto backoff = std::make_shared<ExponentialBackoffWithJitter>();  // 创建指数退避器对象

  auto retry = 0;                                   // 连接重试次数
  do {
    try {
      client_ = detail::TCPClient::connect(addr_, opts, backoff);  // 尝试连接客户端
      // TCP connection established
      C10D_DEBUG("TCP client connected to host {}:{}", addr_.host, addr_.port);  // 调试信息，TCP客户端成功连接

      // client's first query for validation
      validate();                                   // 验证连接

      // success
      break;                                        // 连接成功，退出循环
    } catch (const c10::DistNetworkError& ex) {
      // 捕获 c10::DistNetworkError 异常，表示网络操作错误
      if (deadline < std::chrono::steady_clock::now()) {
        // 如果当前时间超过了截止时间，则抛出超时错误
        C10D_ERROR(
            "TCP client failed to connect/validate to host {}:{} - timed out (try={}, timeout={}ms): {}",
            addr_.host,
            addr_.port,
            retry,
            opts.timeout.count(),
            ex.what());
        throw;
      }

      // 获取下一个重试间隔时间
      auto delayDuration = backoff->nextBackoff();

      // 输出警告信息，表示当前尝试连接失败，将进行重试
      C10D_WARNING(
          "TCP client failed to connect/validate to host {}:{} - retrying (try={}, timeout={}ms, delay={}ms): {}",
          addr_.host,
          addr_.port,
          retry,
          opts.timeout.count(),
          delayDuration.count(),
          ex.what());

      // 线程休眠一段时间后再重试连接
      std::this_thread::sleep_for(delayDuration);
      retry += 1;
    }
  } while (true);

  // 如果 opts.waitWorkers 为真，则等待所有工作线程完成
  if (opts.waitWorkers) {
    waitForWorkers();
  }
}

// 默认析构函数的定义，使用默认的析构行为
TCPStore::~TCPStore() = default;

// 等待所有工作线程完成的方法
void TCPStore::waitForWorkers() {
  // 计时器，记录 waitForWorkers 方法的执行时间
  detail::timing_guard tguard(clientCounters_["waitForWorkers"]);
  // 如果未设置工作线程数量，直接返回
  if (numWorkers_ == c10::nullopt) {
    return;
  }

  // 增加初始键的值
  incrementValueBy(initKey_, 1);

  // 让服务器阻塞，直到所有工作线程完成，确保服务器守护线程一直运行到最后
  if (server_) {
    const auto start = std::chrono::steady_clock::now();
    while (true) {
      // TODO: 有没有更简洁的方式来实现这一部分？
      // 获取初始键对应的值
      std::vector<uint8_t> value = doGet(initKey_);
      // 将值解析为字符指针
      auto buf = reinterpret_cast<const char*>(value.data());
      // 获取值的长度
      auto len = value.size();
      // 将字符数组转换为整数，表示已完成工作的工作线程数
      int numWorkersCompleted = std::stoi(std::string(buf, len));
      // 如果已完成工作的工作线程数达到预期，则跳出循环
      if (numWorkersCompleted >= static_cast<int>(*numWorkers_)) {
        break;
      }
      // 计算已经过的时间
      const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
          std::chrono::steady_clock::now() - start);
      // 如果设置了超时时间且超时，则抛出异常
      if (timeout_ != kNoTimeout && elapsed > timeout_) {
        C10_THROW_ERROR(
            DistStoreError,
            fmt::format(
                "Timed out after {} seconds waiting for clients. {}/{} clients joined.",
                elapsed.count(),
                numWorkersCompleted,
                *numWorkers_));
      }
      // 线程休眠，避免过于频繁地检查
      /* sleep override */
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }
}

// 执行验证操作的方法
void TCPStore::validate() {
  // 使用互斥锁保护活动操作的区域
  const std::lock_guard<std::mutex> lock(activeOpLock_);
  // 创建发送缓冲区对象，执行验证查询
  detail::SendBuffer buffer(*client_, detail::QueryType::VALIDATE);
  // 向缓冲区追加验证魔术数字的值
  buffer.appendValue<std::uint32_t>(c10d::detail::validationMagicNumber);
  // 清空缓冲区，发送数据
  buffer.flush();
}

// 分割设置操作的方法
void TCPStore::_splitSet(
    const std::string& key,
    const std::vector<uint8_t>& data) {
  // 使用互斥锁保护活动操作的区域
  const std::lock_guard<std::mutex> lock(activeOpLock_);
  // 创建发送缓冲区对象，执行设置查询
  detail::SendBuffer buffer(*client_, detail::QueryType::SET);
  // 向缓冲区追加键的前缀和键值
  buffer.appendString(keyPrefix_ + key);
  // 清空缓冲区，发送数据
  buffer.flush();
  // 线程休眠一段时间，模拟操作的延迟
  std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  // 向缓冲区追加数据内容
  buffer.appendBytes(data);
  // 清空缓冲区，发送数据
  buffer.flush();
}

// 设置操作的方法
void TCPStore::set(const std::string& key, const std::vector<uint8_t>& data) {
  // 记录 set 方法的执行时间
  detail::timing_guard tguard(clientCounters_["set"]);
  // 使用互斥锁保护活动操作的区域
  const std::lock_guard<std::mutex> lock(activeOpLock_);
  // 创建发送缓冲区对象，执行设置查询
  detail::SendBuffer buffer(*client_, detail::QueryType::SET);
  // 向缓冲区追加键的前缀和键值
  buffer.appendString(keyPrefix_ + key);
  // 向缓冲区追加数据内容
  buffer.appendBytes(data);
  // 清空缓冲区，发送数据
  buffer.flush();
}

// 比较并设置操作的方法
std::vector<uint8_t> TCPStore::compareSet(
    const std::string& key,
    const std::vector<uint8_t>& expectedValue,
    const std::vector<uint8_t>& desiredValue) {
  // 记录 compareSet 方法的执行时间
  detail::timing_guard tguard(clientCounters_["compareSet"]);
  // 使用互斥锁保护活动操作的区域
  const std::lock_guard<std::mutex> lock(activeOpLock_);
  // 创建发送缓冲区对象，执行比较设置查询
  detail::SendBuffer buffer(*client_, detail::QueryType::COMPARE_SET);
  // 向缓冲区追加键的前缀和键值
  buffer.appendString(keyPrefix_ + key);
  // 向缓冲区追加期望的值和目标值
  buffer.appendBytes(expectedValue);
  buffer.appendBytes(desiredValue);
  // 清空缓冲区，发送数据
  buffer.flush();

  // 接收并返回客户端的响应数据
  return client_->receiveBits();
}
// 调用get方法，计时并记录到clientCounters_中，保护activeOpLock_，然后执行具体的读取操作
std::vector<uint8_t> TCPStore::get(const std::string& key) {
  detail::timing_guard tguard(clientCounters_["get"]);  // 计时并记录到clientCounters_中
  const std::lock_guard<std::mutex> lock(activeOpLock_);  // 加锁activeOpLock_
  return doGet(keyPrefix_ + key);  // 调用具体的读取操作
}

// 执行实际的读取操作，包括等待、发送GET请求并接收数据
std::vector<uint8_t> TCPStore::doGet(const std::string& key) {
  doWait(key, timeout_);  // 等待操作完成
  detail::SendBuffer buffer(*client_, detail::QueryType::GET);  // 创建发送缓冲区，发送GET请求
  buffer.appendString(key);  // 添加键值到缓冲区
  buffer.flush();  // 刷新缓冲区，发送数据

  return client_->receiveBits();  // 接收并返回客户端接收到的数据
}

// 添加操作，计时并记录到clientCounters_中，保护activeOpLock_，然后执行增加数值的操作
int64_t TCPStore::add(const std::string& key, int64_t value) {
  detail::timing_guard tguard(clientCounters_["add"]);  // 计时并记录到clientCounters_中
  const std::lock_guard<std::mutex> lock(activeOpLock_);  // 加锁activeOpLock_
  return incrementValueBy(keyPrefix_ + key, value);  // 调用具体的增加数值操作
}

// 删除键操作，计时并记录到clientCounters_中，保护activeOpLock_，然后执行删除键的操作
bool TCPStore::deleteKey(const std::string& key) {
  detail::timing_guard tguard(clientCounters_["deleteKey"]);  // 计时并记录到clientCounters_中
  const std::lock_guard<std::mutex> lock(activeOpLock_);  // 加锁activeOpLock_
  detail::SendBuffer buffer(*client_, detail::QueryType::DELETE_KEY);  // 创建发送缓冲区，发送DELETE_KEY请求
  buffer.appendString(keyPrefix_ + key);  // 添加键值到缓冲区
  buffer.flush();  // 刷新缓冲区，发送数据

  auto numDeleted = client_->receiveValue<std::int64_t>();  // 接收并返回客户端接收到的数据
  return numDeleted == 1;  // 返回是否成功删除的布尔值
}

// 增加数值操作，发送ADD请求并接收数据
int64_t TCPStore::incrementValueBy(const std::string& key, int64_t delta) {
  detail::SendBuffer buff(*client_, detail::QueryType::ADD);  // 创建发送缓冲区，发送ADD请求
  buff.appendString(key);  // 添加键值到缓冲区
  buff.appendValue<std::int64_t>(delta);  // 添加增量值到缓冲区
  buff.flush();  // 刷新缓冲区，发送数据

  return client_->receiveValue<std::int64_t>();  // 接收并返回客户端接收到的数据
}

// 获取键数量操作，保护activeOpLock_，发送GETNUMKEYS请求并接收数据
int64_t TCPStore::getNumKeys() {
  const std::lock_guard<std::mutex> lock(activeOpLock_);  // 加锁activeOpLock_
  detail::SendBuffer buffer(*client_, detail::QueryType::GETNUMKEYS);  // 创建发送缓冲区，发送GETNUMKEYS请求
  buffer.flush();  // 刷新缓冲区，发送数据

  return client_->receiveValue<std::int64_t>();  // 接收并返回客户端接收到的数据
}

// 检查键是否就绪，计时并记录到clientCounters_中，保护activeOpLock_，发送CHECK请求并接收数据
bool TCPStore::check(const std::vector<std::string>& keys) {
  detail::timing_guard tguard(clientCounters_["check"]);  // 计时并记录到clientCounters_中
  const std::lock_guard<std::mutex> lock(activeOpLock_);  // 加锁activeOpLock_
  detail::SendBuffer buffer(*client_, detail::QueryType::CHECK);  // 创建发送缓冲区，发送CHECK请求
  buffer.appendValue(keys.size());  // 添加键数量到缓冲区

  for (const std::string& key : keys) {
    buffer.appendString(keyPrefix_ + key);  // 添加带前缀的键值到缓冲区
  }
  buffer.flush();  // 刷新缓冲区，发送数据

  auto response = client_->receiveValue<detail::CheckResponseType>();  // 接收并返回客户端接收到的响应类型

  if (response == detail::CheckResponseType::READY) {  // 如果响应为READY
    return true;  // 返回true
  }
  if (response == detail::CheckResponseType::NOT_READY) {  // 如果响应为NOT_READY
    return false;  // 返回false
  }
  TORCH_CHECK(false, "ready or not_ready response expected");  // 抛出异常，预期为ready或not_ready响应
}

// 等待键操作，默认使用timeout_超时时长
void TCPStore::wait(const std::vector<std::string>& keys) {
  wait(keys, timeout_);  // 调用重载的wait方法，使用默认的timeout_
}

// 等待键操作，保护activeOpLock_，创建带前缀的键列表，然后执行等待操作
void TCPStore::wait(
    const std::vector<std::string>& keys,
    const std::chrono::milliseconds& timeout) {
  detail::timing_guard tguard(clientCounters_["wait"]);  // 计时并记录到clientCounters_中
  const std::lock_guard<std::mutex> lock(activeOpLock_);  // 加锁activeOpLock_
  std::vector<std::string> prefixedKeys{};  // 创建带前缀的键列表
  prefixedKeys.reserve(keys.size());
  for (const std::string& key : keys) {
    prefixedKeys.emplace_back(keyPrefix_ + key);  // 添加带前缀的键值到列表中
  }

  doWait(prefixedKeys, timeout);  // 执行具体的等待操作
}

// 执行等待操作，发送WAIT请求并接收数据
void TCPStore::doWait(
    c10::ArrayRef<std::string> keys,
    std::chrono::milliseconds timeout) {
  {
    detail::SendBuffer buffer(*client_, detail::QueryType::WAIT);  // 创建发送缓冲区，发送WAIT请求
    buffer.appendValue(keys.size());  // 添加键数量到缓冲区
    // 遍历 keys 中的每个字符串 key
    for (const std::string& key : keys) {
      // 将每个 key 添加到 buffer 中
      buffer.appendString(key);
    }
    // 将 buffer 中积累的数据发送出去
    buffer.flush();
  }

  // 定义变量 response，用于接收从客户端接收到的响应
  detail::WaitResponseType response;
  // 通过客户端接收一个带有超时的值，将结果存入 response
  if (client_->receiveValueWithTimeout<detail::WaitResponseType>(
          response, timeout)) {
    // 如果接收到的 response 不是 STOP_WAITING，抛出错误信息
    if (response != detail::WaitResponseType::STOP_WAITING) {
      TORCH_CHECK(false, "Stop_waiting response is expected");
    }
    return;
  }
  // 这里是取消等待的超时处理，此处期望服务器能及时响应
  {
    // 创建一个发送缓冲区对象 buffer，发送类型为 CANCEL_WAIT 的查询
    detail::SendBuffer buffer(*client_, detail::QueryType::CANCEL_WAIT);
    // 将 buffer 中的数据发送出去
    buffer.flush();
  }

  // 从客户端接收一个 WaitResponseType 类型的响应，存入 response
  response = client_->receiveValue<detail::WaitResponseType>();
  // 如果此时 response 不是 WAIT_CANCELED，则继续处理
  // 这可能发生在取消之前服务器已经响应，可以忽略这种情况
  if (response != detail::WaitResponseType::WAIT_CANCELED) {
    // 如果 response 不是 STOP_WAITING，则抛出错误信息
    if (response != detail::WaitResponseType::STOP_WAITING) {
      TORCH_CHECK(false, "Stop_waiting response is expected");
    }

    // 继续从客户端接收一个 WaitResponseType 类型的响应，忽略它
    response = client_->receiveValue<detail::WaitResponseType>(); // ignore
    // 如果此时 response 不是 WAIT_CANCELED，则抛出错误信息
    if (response != detail::WaitResponseType::WAIT_CANCELED) {
      TORCH_CHECK(false, "wait_canceled response is expected");
    }
  }
  // 抛出一个 DistStoreError 异常，表示套接字超时
  C10_THROW_ERROR(DistStoreError, "Socket Timeout");
}

// 实现 TCPStore 类的 append 方法，用于向存储中追加数据
void TCPStore::append(
    const std::string& key,                    // 键名
    const std::vector<uint8_t>& data) {        // 数据向量
  // 计时器，用于统计 append 操作的耗时
  detail::timing_guard tguard(clientCounters_["append"]);
  // 使用互斥锁保护共享资源 activeOpLock_
  const std::lock_guard<std::mutex> lock(activeOpLock_);
  // 创建发送缓冲区，并指定操作类型为 APPEND
  detail::SendBuffer buffer(*client_, detail::QueryType::APPEND);
  // 在缓冲区中添加带有前缀的键名
  buffer.appendString(keyPrefix_ + key);
  // 在缓冲区中添加数据内容
  buffer.appendBytes(data);
  // 刷新缓冲区，发送数据
  buffer.flush();
}

// 实现 TCPStore 类的 multiGet 方法，用于获取多个键的数据
std::vector<std::vector<uint8_t>> TCPStore::multiGet(
    const std::vector<std::string>& keys) {     // 键名向量
  // 计时器，用于统计 multiGet 操作的耗时
  detail::timing_guard tguard(clientCounters_["multiGet"]);
  // 使用互斥锁保护共享资源 activeOpLock_
  const std::lock_guard<std::mutex> lock(activeOpLock_);
  // 创建带有前缀的键名向量
  std::vector<std::string> prefixedKeys;
  prefixedKeys.reserve(keys.size());
  // 为每个键名添加前缀
  for (const std::string& key : keys) {
    prefixedKeys.emplace_back(keyPrefix_ + key);
  }
  // 执行等待操作，等待响应返回
  doWait(prefixedKeys, timeout_);

  // 创建发送缓冲区，并指定操作类型为 MULTI_GET
  detail::SendBuffer buffer(*client_, detail::QueryType::MULTI_GET);
  // 在缓冲区中添加键名数量
  buffer.appendValue(keys.size());
  // 在缓冲区中添加所有带有前缀的键名
  for (auto& key : prefixedKeys) {
    buffer.appendString(key);
  }
  // 刷新缓冲区，发送数据
  buffer.flush();

  // 接收并返回多个键对应的数据向量
  std::vector<std::vector<uint8_t>> result;
  result.reserve(keys.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    result.emplace_back(client_->receiveBits());
  }
  return result;
}

// 实现 TCPStore 类的 multiSet 方法，用于设置多个键的数据
void TCPStore::multiSet(
    const std::vector<std::string>& keys,                      // 键名向量
    const std::vector<std::vector<uint8_t>>& values) {         // 数据向量
  // 计时器，用于统计 multiSet 操作的耗时
  detail::timing_guard tguard(clientCounters_["multiSet"]);
  // 检查键名向量和数据向量的大小是否相同
  TORCH_CHECK(
      keys.size() == values.size(),
      "multiSet keys and values vectors must be of same size");
  // 使用互斥锁保护共享资源 activeOpLock_
  const std::lock_guard<std::mutex> lock(activeOpLock_);

  // 创建发送缓冲区，并指定操作类型为 MULTI_SET
  detail::SendBuffer buffer(*client_, detail::QueryType::MULTI_SET);
  // 在缓冲区中添加键值对数量
  buffer.appendValue<std::int64_t>(keys.size());
  // 遍历所有键值对，为每个键名添加前缀并添加数据
  for (auto i : c10::irange(keys.size())) {
    buffer.appendString(keyPrefix_ + keys[i]);
    buffer.appendBytes(values[i]);
  }
  // 刷新缓冲区，发送数据
  buffer.flush();
}

// 返回 TCPStore 类是否支持扩展 API
bool TCPStore::hasExtendedApi() const {
  return true;
}

// 收集客户端计数器信息并返回
std::unordered_map<std::string, std::unordered_map<std::string, double>>
TCPStore::collectClientCounters() const noexcept {
  // 创建结果字典
  std::unordered_map<std::string, std::unordered_map<std::string, double>> res;
  // 遍历客户端计数器映射，将观察到的计数值存入结果字典
  for (const auto& kv : clientCounters_) {
    res[kv.first] = kv.second.observe();
  }
  // 返回结果字典
  return res;
}

} // namespace c10d
```