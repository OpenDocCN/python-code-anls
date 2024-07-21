# `.\pytorch\torch\csrc\distributed\c10d\HashStore.hpp`

```
#pragma once

#include <condition_variable>  // 引入条件变量库，用于多线程同步
#include <mutex>  // 引入互斥锁库，用于多线程互斥访问
#include <unordered_map>  // 引入无序映射库，用于存储键值对

#include <torch/csrc/distributed/c10d/Store.hpp>  // 引入 Torch 分布式存储接口

namespace c10d {

class TORCH_API HashStore : public Store {
 public:
  ~HashStore() override = default;  // 虚析构函数，默认实现

  // 设置给定键的值
  void set(const std::string& key, const std::vector<uint8_t>& data) override;

  // 比较并设置给定键的值，返回旧值
  std::vector<uint8_t> compareSet(
      const std::string& key,
      const std::vector<uint8_t>& expectedValue,
      const std::vector<uint8_t>& desiredValue) override;

  // 获取给定键的值
  std::vector<uint8_t> get(const std::string& key) override;

  // 等待给定键列表中的条件成立
  void wait(const std::vector<std::string>& keys) override {
    wait(keys, timeout_);
  }

  // 在给定超时时间内等待给定键列表中的条件成立
  void wait(
      const std::vector<std::string>& keys,
      const std::chrono::milliseconds& timeout) override;

  // 给定键的值增加指定的整数值
  int64_t add(const std::string& key, int64_t value) override;

  // 返回存储中的键的数量
  int64_t getNumKeys() override;

  // 检查给定键列表是否存在于存储中
  bool check(const std::vector<std::string>& keys) override;

  // 删除给定键及其对应的值
  bool deleteKey(const std::string& key) override;

  // 给定键对应的值末尾追加新的数据
  void append(const std::string& key, const std::vector<uint8_t>& value)
      override;

  // 批量获取给定键列表对应的值
  std::vector<std::vector<uint8_t>> multiGet(
      const std::vector<std::string>& keys) override;

  // 批量设置给定键列表对应的值
  void multiSet(
      const std::vector<std::string>& keys,
      const std::vector<std::vector<uint8_t>>& values) override;

  // 返回是否支持扩展的 API（追加、批量获取和批量设置）
  bool hasExtendedApi() const override;

 protected:
  std::unordered_map<std::string, std::vector<uint8_t>> map_;  // 使用无序映射存储键值对
  std::mutex m_;  // 互斥锁，保护对映射的并发访问
  std::condition_variable cv_;  // 条件变量，用于等待条件达成的通知
};

} // namespace c10d
```