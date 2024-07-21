# `.\pytorch\torch\csrc\distributed\c10d\TCPStore.hpp`

```
#pragma once
// 预处理指令：确保头文件只被编译一次

#include <cstddef>
// 包含标准库头文件，定义了 size_t 类型

#include <cstdint>
// 包含标准整数类型的头文件，如 uint16_t

#include <memory>
// 包含智能指针和相关工具的头文件

#include <torch/csrc/distributed/c10d/Store.hpp>
// 引入 Torch 分布式存储相关的头文件

namespace c10d {
namespace detail {

class TCPServer;
// 声明 TCPServer 类，用于 TCP 服务器功能

class TCPClient;
// 声明 TCPClient 类，用于 TCP 客户端功能

struct SocketAddress {
  std::string host{};
  std::uint16_t port{};
};
// 定义 SocketAddress 结构体，包含主机名和端口号信息

class Counter {
 public:
  void update(double val);
  // 方法：更新计数器的值

  std::unordered_map<std::string, double> observe() const;
  // 方法：返回观察到的数据映射表

  double mean() const noexcept {
    return mean_;
  }
  // 方法：返回计数的平均值，不抛出异常

  int64_t count() const noexcept {
    return count_;
  }
  // 方法：返回计数的数量，不抛出异常

  double variance() const noexcept {
    return m2_ / static_cast<double>(count_);
  }
  // 方法：返回计数的方差，不抛出异常

  double sample_variance() const noexcept {
    return m2_ / static_cast<double>(count_ - 1);
  }
  // 方法：返回样本方差，不抛出异常

 private:
  int64_t count_ = 0;
  // 成员变量：计数器的计数值

  double mean_ = 0;
  // 成员变量：计数的平均值

  double m2_ = 0;
  // 成员变量：计数的二阶矩
};

} // namespace detail

struct TCPStoreOptions {
  static constexpr std::uint16_t kDefaultPort = 29500;
  // 静态成员常量：默认的 TCP 端口号

  std::uint16_t port = kDefaultPort;
  // 成员变量：TCP 服务器或客户端的端口号，默认为 kDefaultPort

  bool isServer = false;
  // 成员变量：标志是否为 TCP 服务器，默认为 false

  std::optional<std::size_t> numWorkers = c10::nullopt;
  // 成员变量：工作线程数的可选值，默认为空

  bool waitWorkers = true;
  // 成员变量：是否等待工作线程，默认为 true

  std::chrono::milliseconds timeout = Store::kDefaultTimeout;
  // 成员变量：超时时间，默认为 Store 类的默认超时时间

  bool multiTenant = false;
  // 成员变量：多租户标志，默认为 false

  std::optional<int> masterListenFd = c10::nullopt;
  // 成员变量：主监听文件描述符的可选值，默认为空

  bool useLibUV = true;
  // 成员变量：是否使用实验性的 libUV 后端，默认为 true
};
// TORCH_API 是一个宏，用于声明该类的公共 API 的可见性
// TCPStore 是一个继承自 Store 的类，实现了 TCP 连接的存储功能
class TORCH_API TCPStore : public Store {
 public:
  // 定义连接重试的延迟时间为 1000 毫秒
  static constexpr std::chrono::milliseconds kConnectRetryDelay{1000};

  // 显式构造函数，接受主机名和 TCPStoreOptions 对象作为参数
  explicit TCPStore(std::string host, const TCPStoreOptions& opts = {});

  // 显式构造函数，用来创建 TCPStore 实例，已被废弃，推荐使用另一构造函数
  [[deprecated("Use TCPStore(host, opts) instead.")]] explicit TCPStore(
      const std::string& masterAddr,
      std::uint16_t masterPort,
      std::optional<int> numWorkers = c10::nullopt,
      bool isServer = false,
      const std::chrono::milliseconds& timeout = kDefaultTimeout,
      bool waitWorkers = true);

  // 虚析构函数，用于释放资源
  ~TCPStore() override;

  // 设置指定键的值
  void set(const std::string& key, const std::vector<uint8_t>& value) override;

  // 比较设置操作，比较预期值和期望值，并返回新的值
  std::vector<uint8_t> compareSet(
      const std::string& key,
      const std::vector<uint8_t>& expectedValue,
      const std::vector<uint8_t>& desiredValue) override;

  // 获取指定键的值
  std::vector<uint8_t> get(const std::string& key) override;

  // 将指定键的值增加指定的增量，并返回新的值
  int64_t add(const std::string& key, int64_t value) override;

  // 删除指定键
  bool deleteKey(const std::string& key) override;

  // 检查给定键是否存在
  bool check(const std::vector<std::string>& keys) override;

  // 获取键的数量
  int64_t getNumKeys() override;

  // 等待指定键的操作完成
  void wait(const std::vector<std::string>& keys) override;

  // 在指定超时时间内等待指定键的操作完成
  void wait(
      const std::vector<std::string>& keys,
      const std::chrono::milliseconds& timeout) override;

  // 向指定键追加数据
  void append(const std::string& key, const std::vector<uint8_t>& value)
      override;

  // 批量获取多个键的值
  std::vector<std::vector<uint8_t>> multiGet(
      const std::vector<std::string>& keys) override;

  // 批量设置多个键的值
  void multiSet(
      const std::vector<std::string>& keys,
      const std::vector<std::vector<uint8_t>>& values) override;

  // 是否支持扩展 API
  bool hasExtendedApi() const override;

  // 等待所有工作线程加入完成
  void waitForWorkers();

  // 返回 TCPStore 使用的主机名
  const std::string& getHost() const noexcept {
    return addr_.host;
  }

  // 返回 TCPStore 使用的端口号
  std::uint16_t getPort() const noexcept {
    return addr_.port;
  }

  // 收集客户端计数器的信息
  std::unordered_map<std::string, std::unordered_map<std::string, double>>
  collectClientCounters() const noexcept;

  // 是否使用 LibUv 后端
  bool isLibUvBackend() const noexcept {
    return usingLibUv_;
  }

  // 内部测试使用的函数，分割指定键的设置操作
  void _splitSet(const std::string& key, const std::vector<uint8_t>& data);

 private:
  // 增加指定键的值，并返回增加后的新值
  int64_t incrementValueBy(const std::string& key, int64_t delta);

  // 验证函数，用于内部状态验证
  void validate();

  // 执行获取指定键值的函数
  std::vector<uint8_t> doGet(const std::string& key);

  // 执行等待操作的函数
  void doWait(
      c10::ArrayRef<std::string> keys,
      std::chrono::milliseconds timeout);

  // 存储 TCP 地址信息
  detail::SocketAddress addr_;

  // TCP 服务端实例的共享指针
  std::shared_ptr<detail::TCPServer> server_;

  // TCP 客户端实例的唯一指针
  std::unique_ptr<detail::TCPClient> client_;

  // 工作线程数量的可选值
  std::optional<std::size_t> numWorkers_;

  // 初始化键的字符串常量
  const std::string initKey_ = "init/";

  // 键的前缀字符串常量
  const std::string keyPrefix_ = "/";

  // 活跃操作锁的互斥量
  std::mutex activeOpLock_;

  // 客户端计数器的哈希映射
  std::unordered_map<std::string, detail::Counter> clientCounters_;

  // 是否使用 LibUv 后端的标志位
  bool usingLibUv_ = true;
};

} // namespace c10d
```