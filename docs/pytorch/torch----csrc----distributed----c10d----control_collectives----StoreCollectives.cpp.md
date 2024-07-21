# `.\pytorch\torch\csrc\distributed\c10d\control_collectives\StoreCollectives.cpp`

```py
// 引入必要的头文件，包括异常处理、格式化输出、分布式存储等
#include <c10/util/Exception.h>
#include <fmt/format.h>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/control_collectives/StoreCollectives.hpp>
#include <chrono>
#include <exception>
#include <vector>

// 匿名命名空间，定义了一个函数，根据给定的键和排名生成一个组合键
namespace {
std::string getRankKey(const std::string& key, int rank) {
  return fmt::format("{}/{}", key, rank);
}
} // namespace

namespace c10d {

// StoreCollectives 类的构造函数实现
StoreCollectives::StoreCollectives(
    c10::intrusive_ptr<::c10d::Store> store,
    int rank,
    int worldSize)
    : store_(std::move(store)), rank_(rank), worldSize_(worldSize) {}

// barrier 方法实现，用于实现同步屏障操作
void StoreCollectives::barrier(
    const std::string& key,
    std::chrono::milliseconds timeout,
    bool blocking) {
  enforceUnique(key); // 确保键的唯一性

  StoreTimeoutGuard g{*store_, timeout}; // 设置超时保护

  // 为计算参与的成员数量和最后成员创建键
  auto num_members_key = fmt::format("{}/num_members", key);
  auto last_members_key = fmt::format("{}/last_members", key);

  // 将当前成员添加到存储，并获取索引
  auto idx = store_->add(num_members_key, 1);
  store_->set(getRankKey(key, rank_), "joined");

  // 如果所有成员都加入了，设置最后成员键
  if (idx == worldSize_) {
    store_->set(last_members_key, "<val_ignored>");
  } else if (blocking) {
    // 如果阻塞标志为真，等待最后成员加入
    try {
      store_->wait({last_members_key});
    } catch (const std::exception& e) {
      // 捕获异常，生成错误消息
      std::string msg = "barrier failed -- missing ranks: ";
      for (int i = 0; i < worldSize_; i++) {
        if (i == rank_) {
          continue;
        }
        auto rank_key = getRankKey(key, i);
        if (!store_->check({rank_key})) {
          msg += fmt::format("{}, ", i);
        }
      }
      throw std::runtime_error(msg + e.what());
    }
  }
}

// broadcastSend 方法实现，用于发送广播消息
void StoreCollectives::broadcastSend(
    const std::string& key,
    const std::vector<uint8_t>& data,
    std::chrono::milliseconds timeout) {
  enforceUnique(key); // 确保键的唯一性

  StoreTimeoutGuard g{*store_, timeout}; // 设置超时保护

  // 将数据通过键存储到分布式存储中
  store_->set(key, data);
}

// broadcastRecv 方法实现，用于接收广播消息
std::vector<uint8_t> StoreCollectives::broadcastRecv(
    const std::string& key,
    std::chrono::milliseconds timeout) {
  enforceUnique(key); // 确保键的唯一性

  StoreTimeoutGuard g{*store_, timeout}; // 设置超时保护

  // 从分布式存储中获取数据并返回
  return store_->get(key);
}

// gatherSend 方法实现，用于发送聚集消息
void StoreCollectives::gatherSend(
    const std::string& key,
    const std::vector<uint8_t>& data,
    std::chrono::milliseconds timeout) {
  enforceUnique(key); // 确保键的唯一性

  StoreTimeoutGuard g{*store_, timeout}; // 设置超时保护

  // 将数据通过包含当前排名的键存储到分布式存储中
  auto rank_key = getRankKey(key, rank_);
  store_->set(rank_key, data);
}

// gatherRecv 方法实现，用于接收聚集消息
std::vector<std::vector<uint8_t>> StoreCollectives::gatherRecv(
    const std::string& key,
    const std::vector<uint8_t>& data,
    std::chrono::milliseconds timeout) {
  enforceUnique(key); // 确保键的唯一性

  StoreTimeoutGuard g{*store_, timeout}; // 设置超时保护

  // 构造要获取的键列表
  std::vector<std::string> keys;
  keys.reserve(worldSize_);
  for (int i = 0; i < worldSize_; i++) {
    if (i == rank_) {
      continue;
    }
    auto rank_key = getRankKey(key, i);
    keys.emplace_back(rank_key);
  }

  // 执行批量获取操作并返回结果
  std::vector<std::vector<uint8_t>> results;
  results.reserve(worldSize_);

  try {
    results = store_->multiGet(keys);
  } catch (const std::exception& e) {
    // 捕获异常，生成错误消息
    std::string msg = "gather failed -- missing ranks: ";
    for (int i = 0; i < worldSize_; i++) {
      if (i == rank_) {
        continue;
      }
      auto rank_key = getRankKey(key, i);
      if (!store_->check({rank_key})) {
        msg += fmt::format("{}, ", i);
      }
    }
    throw std::runtime_error(msg + e.what());
  }
}
    // 循环遍历从 0 到 worldSize_-1 的整数 i
    for (int i = 0; i < worldSize_; i++) {
      // 如果 i 等于当前进程的 rank_，则跳过当前循环
      if (i == rank_) {
        continue;
      }
      // 调用 getRankKey 函数获取特定于进程 i 的键值
      auto rank_key = getRankKey(key, i);
      // 检查 store_ 中是否存在特定键 rank_key，若不存在则执行以下操作
      if (!store_->check({rank_key})) {
        // 将进程号 i 添加到异常消息 msg 中，格式为 "{}, "
        msg += fmt::format("{}, ", i);
      }
    }
    // 抛出运行时异常，异常信息为 msg 和捕获到的异常 e 的描述
    throw std::runtime_error(msg + e.what());
  }

  // 在本地数据结果中的 rank_ 位置插入数据 data
  results.insert(results.begin() + rank_, data);
  // 返回插入后的结果集合 results
  return results;
// 确保键值唯一性，如果已存在则抛出异常
void StoreCollectives::enforceUnique(const std::string& key) {
    // 查找键是否已经存在于集合中
    auto it = seenKeys_.find(key);
    // 使用内部断言确保键的唯一性，否则抛出错误信息
    TORCH_INTERNAL_ASSERT(
        it == seenKeys_.end(), "Key ", key, " has already been used.");
    // 将键添加到已见键集合中
    seenKeys_.emplace(key);
}

// 执行分散发送操作
std::vector<uint8_t> StoreCollectives::scatterSend(
    const std::string& key,
    const std::vector<std::vector<uint8_t>>& data,
    std::chrono::milliseconds timeout) {
  // 强制确保键的唯一性
  enforceUnique(key);
  // 创建存储超时保护对象
  StoreTimeoutGuard g{*store_, timeout};

  // 存储所有目标键
  std::vector<std::string> keys;
  keys.reserve(worldSize_);
  // 生成所有非本地键
  for (int i = 0; i < worldSize_; i++) {
    // 如果是本地进程，则跳过
    if (i == rank_) {
      continue;
    }
    // 获取特定等级的键
    auto rank_key = getRankKey(key, i);
    // 将键添加到键列表中
    keys.emplace_back(rank_key);
  }
  // 获取本地数据
  auto local = data.at(rank_);

  // 复制数据以供发送
  std::vector<std::vector<uint8_t>> toSend{data};

  // 删除指定的本地数据以便发送
  toSend.erase(toSend.begin() + rank_);

  // 执行多键设置操作
  store_->multiSet(keys, toSend);

  // 返回本地数据
  return local;
}

// 执行分散接收操作
std::vector<uint8_t> StoreCollectives::scatterRecv(
    const std::string& key,
    std::chrono::milliseconds timeout) {
  // 强制确保键的唯一性
  enforceUnique(key);
  // 创建存储超时保护对象
  StoreTimeoutGuard g{*store_, timeout};

  // 获取特定等级的键
  auto rank_key = getRankKey(key, rank_);
  // 返回存储中的数据
  return store_->get(rank_key);
}

// 执行全局聚集操作
std::vector<std::vector<uint8_t>> StoreCollectives::allGather(
    const std::string& key,
    const std::vector<uint8_t>& data,
    std::chrono::milliseconds timeout) {
  // 强制确保键的唯一性
  enforceUnique(key);
  // 创建存储超时保护对象
  StoreTimeoutGuard g{*store_, timeout};

  // 获取本地键
  auto localKey = getRankKey(key, rank_);
  // 将数据设置到存储中
  store_->set(localKey, data);

  // 存储所有目标键
  std::vector<std::string> keys;
  keys.reserve(worldSize_);

  // 生成所有键
  for (int i = 0; i < worldSize_; i++) {
    // 获取特定等级的键
    auto rank_key = getRankKey(key, i);
    // 将键添加到键列表中
    keys.emplace_back(rank_key);
  }

  // 尝试从存储中获取数据
  try {
    return store_->multiGet(keys);
  } catch (const std::exception& e) {
    // 构造异常消息，指出未找到的等级
    std::string msg = "all_gather failed -- missing ranks: ";
    for (int i = 0; i < worldSize_; i++) {
      // 如果是本地进程，则跳过
      if (i == rank_) {
        continue;
      }
      // 获取特定等级的键
      auto rank_key = getRankKey(key, i);
      // 检查存储是否存在键
      if (!store_->check({rank_key})) {
        // 格式化未找到的等级信息并添加到异常消息中
        msg += fmt::format("{}, ", i);
      }
    }
    // 抛出运行时错误，包括异常消息
    throw std::runtime_error(msg + e.what());
  }
}

// 执行全局求和操作
int64_t StoreCollectives::allSum(
    const std::string& key,
    int64_t value,
    std::chrono::milliseconds timeout) {
  // 强制确保键的唯一性
  enforceUnique(key);
  // 创建存储超时保护对象
  StoreTimeoutGuard g{*store_, timeout};

  // 向存储中添加值
  store_->add(key, value);

  // 执行屏障操作以等待所有进程完成
  barrier(key + "/barrier", timeout);

  // 返回存储中的累加结果
  return store_->add(key, 0);
}
```