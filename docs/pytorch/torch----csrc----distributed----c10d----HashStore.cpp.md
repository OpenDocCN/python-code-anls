# `.\pytorch\torch\csrc\distributed\c10d\HashStore.cpp`

```
// 包含 HashStore 类的实现文件
#include <torch/csrc/distributed/c10d/HashStore.hpp>

// 包含 POSIX 标准库头文件
#include <unistd.h>
// 包含 C++ 标准库头文件
#include <cstdint>

// 包含用于时间操作的头文件
#include <chrono>

// 包含 C10 库中的异常处理头文件
#include <c10/util/Exception.h>

// 使用 c10d 命名空间
namespace c10d {

// 设置指定键的数据到哈希存储中
void HashStore::set(const std::string& key, const std::vector<uint8_t>& data) {
  // 加锁以保证线程安全
  std::unique_lock<std::mutex> lock(m_);
  // 将键值对存入 map_
  map_[key] = data;
  // 通知所有等待的线程
  cv_.notify_all();
}

// 比较并设置指定键的值
std::vector<uint8_t> HashStore::compareSet(
    const std::string& key,
    const std::vector<uint8_t>& expectedValue,
    const std::vector<uint8_t>& desiredValue) {
  // 加锁以保证线程安全
  std::unique_lock<std::mutex> lock(m_);
  // 查找指定键的值
  auto it = map_.find(key);
  // 如果键不存在且期望值为空，或者键存在且当前值等于期望值，则设置新值
  if ((it == map_.end() && expectedValue.empty()) ||
      (it != map_.end() && it->second == expectedValue)) {
    map_[key] = desiredValue;
    // 通知所有等待的线程
    cv_.notify_all();
    return desiredValue;
  } else if (it == map_.end()) {
    // 如果键不存在，返回期望值
    return expectedValue;
  }
  // 键存在但当前值不符合期望，返回当前值
  return it->second;
}

// 获取指定键的值
std::vector<uint8_t> HashStore::get(const std::string& key) {
  // 加锁以保证线程安全
  std::unique_lock<std::mutex> lock(m_);
  // 查找指定键的值
  auto it = map_.find(key);
  if (it != map_.end()) {
    // 如果键存在，直接返回值
    return it->second;
  }
  // 慢路径：等待直到超时或键出现
  auto pred = [&]() { return map_.find(key) != map_.end(); };
  if (timeout_ == kNoTimeout) {
    // 无超时设定，等待直到条件满足
    cv_.wait(lock, pred);
  } else {
    // 有超时设定，等待一段时间或直到条件满足
    if (!cv_.wait_for(lock, timeout_, pred)) {
      // 超时抛出异常
      C10_THROW_ERROR(DistStoreError, "Wait timeout");
    }
  }
  // 返回键对应的值
  return map_[key];
}

// 等待一组键出现在哈希存储中
void HashStore::wait(
    const std::vector<std::string>& keys,
    const std::chrono::milliseconds& timeout) {
  // 计算超时终止时间点
  const auto end = std::chrono::steady_clock::now() + timeout;
  // 定义等待条件谓词
  auto pred = [&]() {
    auto done = true;
    for (const auto& key : keys) {
      if (map_.find(key) == map_.end()) {
        done = false;
        break;
      }
    }
    return done;
  };

  // 加锁以保证线程安全
  std::unique_lock<std::mutex> lock(m_);
  if (timeout == kNoTimeout) {
    // 无超时设定，等待直到条件满足
    cv_.wait(lock, pred);
  } else {
    // 有超时设定，等待一段时间或直到条件满足
    if (!cv_.wait_until(lock, end, pred)) {
      // 超时抛出异常
      C10_THROW_ERROR(DistStoreError, "Wait timeout");
    }
  }
}

// 将指定键的值增加一个整数，并返回增加后的结果
int64_t HashStore::add(const std::string& key, int64_t i) {
  // 加锁以保证线程安全
  std::unique_lock<std::mutex> lock(m_);
  // 获取当前键的值
  const auto& value = map_[key];
  int64_t ti = i;
  // 如果值不为空，解析并累加整数值
  if (!value.empty()) {
    auto buf = reinterpret_cast<const char*>(value.data());
    auto len = value.size();
    ti += std::stoll(std::string(buf, len));
  }

  // 将累加后的整数值转换为字符串，并存入 map_
  auto str = std::to_string(ti);
  const uint8_t* strB = reinterpret_cast<const uint8_t*>(str.c_str());
  map_[key] = std::vector<uint8_t>(strB, strB + str.size());
  return ti;
}

// 获取哈希存储中键值对的数量
int64_t HashStore::getNumKeys() {
  // 加锁以保证线程安全
  std::unique_lock<std::mutex> lock(m_);
  // 返回哈希存储中键值对的数量
  return static_cast<int64_t>(map_.size());
}

// 删除指定键及其对应的值
bool HashStore::deleteKey(const std::string& key) {
  // 加锁以保证线程安全
  std::unique_lock<std::mutex> lock(m_);
  // 删除指定键及其对应的值，返回删除成功与否
  auto numDeleted = map_.erase(key);
  return (numDeleted == 1);
}

} // namespace c10d
# 检查给定的键列表是否都存在于哈希映射中
bool HashStore::check(const std::vector<std::string>& keys) {
  # 使用互斥锁锁定共享资源以确保线程安全访问
  std::unique_lock<std::mutex> lock(m_);
  # 遍历传入的键列表
  for (const auto& key : keys) {
    # 如果键在哈希映射中找不到，则返回 false
    if (map_.find(key) == map_.end()) {
      return false;
    }
  }
  # 所有键都存在于哈希映射中，则返回 true
  return true;
}

# 向哈希映射中追加键值对
void HashStore::append(
    const std::string& key,
    const std::vector<uint8_t>& value) {
  # 使用互斥锁锁定共享资源以确保线程安全操作
  std::unique_lock<std::mutex> lock(m_);
  # 查找给定键在哈希映射中的位置
  auto it = map_.find(key);
  # 如果键不存在，则直接插入新的键值对
  if (it == map_.end()) {
    map_[key] = value;
  } else {
    # 如果键已存在，则将新的值追加到已有值的末尾
    it->second.insert(it->second.end(), value.begin(), value.end());
  }
  # 通知所有等待的线程有新数据可用
  cv_.notify_all();
}

# 批量获取多个键对应的值
std::vector<std::vector<uint8_t>> HashStore::multiGet(
    const std::vector<std::string>& keys) {
  # 使用互斥锁锁定共享资源以确保线程安全操作
  std::unique_lock<std::mutex> lock(m_);
  # 计算超时时间点
  auto deadline = std::chrono::steady_clock::now() + timeout_;
  # 准备用于存放结果的向量
  std::vector<std::vector<uint8_t>> res;
  res.reserve(keys.size());

  # 遍历传入的键列表
  for (auto& key : keys) {
    # 查找键在哈希映射中的位置
    auto it = map_.find(key);
    # 如果键存在，则将对应的值添加到结果向量中
    if (it != map_.end()) {
      res.emplace_back(it->second);
    } else {
      # 如果键不存在，则等待直到键出现或超时
      auto pred = [&]() { return map_.find(key) != map_.end(); };
      if (timeout_ == kNoTimeout) {
        cv_.wait(lock, pred);
      } else {
        if (!cv_.wait_until(lock, deadline, pred)) {
          # 如果超时则抛出异常
          C10_THROW_ERROR(DistStoreError, "Wait timeout");
        }
      }
      # 将键对应的值添加到结果向量中
      res.emplace_back(map_[key]);
    }
  }
  # 返回获取的所有值的向量
  return res;
}

# 批量设置多个键对应的值
void HashStore::multiSet(
    const std::vector<std::string>& keys,
    const std::vector<std::vector<uint8_t>>& values) {
  # 使用互斥锁锁定共享资源以确保线程安全操作
  std::unique_lock<std::mutex> lock(m_);

  # 遍历传入的键值对列表，将键值对设置到哈希映射中
  for (auto i : ::c10::irange(keys.size())) {
    map_[keys[i]] = values[i];
  }
  # 通知所有等待的线程有新数据可用
  cv_.notify_all();
}

# 返回是否支持扩展的 API
bool HashStore::hasExtendedApi() const {
  return true;
}

} // namespace c10d
```