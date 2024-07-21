# `.\pytorch\torch\csrc\distributed\c10d\Store.hpp`

```py
#pragma once
// 预处理指令：确保本头文件仅被编译一次

#include <chrono>
// 引入时间库，用于处理时间相关的功能
#include <cstdint>
// 引入整数类型，如 uint8_t
#include <string>
// 引入字符串类型
#include <vector>
// 引入向量类型

#include <c10/macros/Macros.h>
// 引入 C10 库的宏定义
#include <torch/custom_class.h>
// 引入 PyTorch 的自定义类支持

namespace c10d {

// callback function will be given arguments (optional<string> oldValue,
// optional<string> newValue)
// 回调函数类型声明，接受两个可选的字符串参数：旧值和新值
using WatchKeyCallback =
    std::function<void(std::optional<std::string>, std::optional<std::string>)>;

class TORCH_API Store : public torch::CustomClassHolder {
// TORCH_API Store 类，继承自 torch::CustomClassHolder

 public:
  static constexpr std::chrono::milliseconds kDefaultTimeout =
      std::chrono::seconds(300);
  // 静态常量，表示默认超时时间为 300 秒（5分钟）
  static constexpr std::chrono::milliseconds kNoTimeout =
      std::chrono::milliseconds::zero();
  // 静态常量，表示无超时时间，初始化为零毫秒

  Store() : timeout_(kDefaultTimeout) {}
  // 默认构造函数，设置超时时间为默认超时时间

  explicit Store(const std::chrono::milliseconds& timeout)
      : timeout_(timeout) {}
  // 显式构造函数，接受超时时间参数，设置超时时间为给定值

  Store(const Store&) = default;
  // 拷贝构造函数，默认实现

  Store(Store&&) noexcept = default;
  // 移动构造函数，默认实现

  ~Store() override = default;
  // 虚析构函数，使用默认实现

  void set(const std::string& key, const std::string& value);
  // 设置键值对应关系，接受字符串键和值

  virtual void set(
      const std::string& key,
      const std::vector<uint8_t>& value) = 0;
  // 纯虚函数，设置键和字节数组值

  std::string compareSet(
      const std::string& key,
      const std::string& currentValue,
      const std::string& newValue);
  // 比较并设置字符串键的值，接受当前值和新值参数

  virtual std::vector<uint8_t> compareSet(
      const std::string& key,
      const std::vector<uint8_t>& currentValue,
      const std::vector<uint8_t>& newValue) {
    TORCH_INTERNAL_ASSERT(false, "Not implemented.");
    // 比较并设置字节数组键的值，如果调用则会抛出未实现异常
  }

  std::string get_to_str(const std::string& key);
  // 获取指定键对应的字符串值

  virtual std::vector<uint8_t> get(const std::string& key) = 0;
  // 纯虚函数，获取指定键对应的字节数组值

  virtual int64_t add(const std::string& key, int64_t value) = 0;
  // 纯虚函数，为指定键的值增加整数值

  virtual bool deleteKey(const std::string& key) = 0;
  // 纯虚函数，删除指定键的键值对

  virtual bool check(const std::vector<std::string>& keys) = 0;
  // 纯虚函数，检查给定键的存在性

  virtual int64_t getNumKeys() = 0;
  // 纯虚函数，获取存储中的键数量

  virtual void wait(const std::vector<std::string>& keys) = 0;
  // 纯虚函数，等待给定键变为可用状态

  virtual void wait(
      const std::vector<std::string>& keys,
      const std::chrono::milliseconds& timeout) = 0;
  // 纯虚函数，等待给定键在超时时间内变为可用状态

  virtual const std::chrono::milliseconds& getTimeout() const noexcept;
  // 获取当前超时时间

  virtual void setTimeout(const std::chrono::milliseconds& timeout);
  // 设置超时时间

  // watchKey() is deprecated and no longer supported.
  // watchKey() 方法已弃用，不再支持
  virtual void watchKey(
      const std::string& /* unused */,
      WatchKeyCallback /* unused */) {
    TORCH_CHECK(false, "watchKey is deprecated, no implementation support it.");
    // watchKey 方法的虚实现，抛出异常指示不再支持
  }

  virtual void append(
      const std::string& key,
      const std::vector<uint8_t>& value);
  // 追加方式设置指定键的字节数组值

  virtual std::vector<std::vector<uint8_t>> multiGet(
      const std::vector<std::string>& keys);
  // 批量获取多个键的字节数组值

  virtual void multiSet(
      const std::vector<std::string>& keys,
      const std::vector<std::vector<uint8_t>>& values);
  // 批量设置多个键对应的字节数组值

  // Returns true if this store support append, multiGet and multiSet
  // 如果存储支持追加、批量获取和批量设置，则返回 true
  virtual bool hasExtendedApi() const;

 protected:
  std::chrono::milliseconds timeout_;
  // 超时时间成员变量
};

/*
StoreTimeoutGuard is a RAII guard that will set the store timeout and restore it
when it returns.
*/
// StoreTimeoutGuard 是一个 RAII 守卫，用于设置存储超时时间并在退出时恢复
class StoreTimeoutGuard {
 public:
  explicit StoreTimeoutGuard(
      Store& store,
      const std::chrono::milliseconds& timeout)
      : store_(store), oldTimeout_(store.getTimeout()) {
    // 构造函数：接收一个 Store 引用和超时时间，设置新的超时时间并保存旧超时时间
    store.setTimeout(timeout);
  }

  ~StoreTimeoutGuard() {
    // 析构函数：在对象生命周期结束时，恢复原始的超时时间
    store_.setTimeout(oldTimeout_);
  }

  /* Disabling copy and move semantics */
  // 禁用复制和移动语义
  StoreTimeoutGuard(const StoreTimeoutGuard&) = delete;
  StoreTimeoutGuard& operator=(const StoreTimeoutGuard&) = delete;
  StoreTimeoutGuard(StoreTimeoutGuard&&) = delete;
  StoreTimeoutGuard& operator=(StoreTimeoutGuard&&) = delete;

 private:
  Store& store_;  // 引用一个 Store 对象
  std::chrono::milliseconds oldTimeout_{};  // 保存旧的超时时间
};

} // namespace c10d
```