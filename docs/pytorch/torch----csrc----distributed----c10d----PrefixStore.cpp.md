# `.\pytorch\torch\csrc\distributed\c10d\PrefixStore.cpp`

```
// 包含头文件 <torch/csrc/distributed/c10d/PrefixStore.hpp>，以及 <utility>
#include <torch/csrc/distributed/c10d/PrefixStore.hpp>
#include <utility>

namespace c10d {

// PrefixStore 类的构造函数，接受一个前缀和一个存储对象指针作为参数
PrefixStore::PrefixStore(std::string prefix, c10::intrusive_ptr<Store> store)
    : prefix_(std::move(prefix)), store_(std::move(store)) {}

// 将给定的 key 与前缀 prefix_ 结合，返回合并后的键名
std::string PrefixStore::joinKey(const std::string& key) {
  return prefix_ + "/" + key;
}

// 将给定的 keys 中的每个 key 与前缀 prefix_ 结合，返回合并后的键名的向量
std::vector<std::string> PrefixStore::joinKeys(
    const std::vector<std::string>& keys) {
  std::vector<std::string> joinedKeys;
  joinedKeys.reserve(keys.size());
  for (const auto& key : keys) {
    joinedKeys.emplace_back(joinKey(key));
  }
  return joinedKeys;
}

// 将给定的 key 和 value 存储到 store_ 中，键名为 prefix_ + "/" + key
void PrefixStore::set(
    const std::string& key,
    const std::vector<uint8_t>& value) {
  store_->set(joinKey(key), value);
}

// 在 store_ 中比较并设置给定 key 的值，键名为 prefix_ + "/" + key
std::vector<uint8_t> PrefixStore::compareSet(
    const std::string& key,
    const std::vector<uint8_t>& expectedValue,
    const std::vector<uint8_t>& desiredValue) {
  return store_->compareSet(joinKey(key), expectedValue, desiredValue);
}

// 从 store_ 中获取给定 key 的值，键名为 prefix_ + "/" + key
std::vector<uint8_t> PrefixStore::get(const std::string& key) {
  return store_->get(joinKey(key));
}

// 在 store_ 中将给定 key 的值增加指定的整数值，键名为 prefix_ + "/" + key
int64_t PrefixStore::add(const std::string& key, int64_t value) {
  return store_->add(joinKey(key), value);
}

// 删除 store_ 中的给定 key，键名为 prefix_ + "/" + key
bool PrefixStore::deleteKey(const std::string& key) {
  return store_->deleteKey(joinKey(key));
}

// 获取 store_ 中键值对的数量
int64_t PrefixStore::getNumKeys() {
  return store_->getNumKeys();
}

// 检查给定 keys 中的所有键名（加上 prefix_ 后的）是否存在于 store_ 中
bool PrefixStore::check(const std::vector<std::string>& keys) {
  auto joinedKeys = joinKeys(keys);
  return store_->check(joinedKeys);
}

// 等待给定 keys 中的所有键名（加上 prefix_ 后的）在 store_ 中变为可用状态
void PrefixStore::wait(const std::vector<std::string>& keys) {
  auto joinedKeys = joinKeys(keys);
  store_->wait(joinedKeys);
}

// 在给定的超时时间内，等待给定 keys 中的所有键名（加上 prefix_ 后的）在 store_ 中变为可用状态
void PrefixStore::wait(
    const std::vector<std::string>& keys,
    const std::chrono::milliseconds& timeout) {
  auto joinedKeys = joinKeys(keys);
  store_->wait(joinedKeys, timeout);
}

// 返回存储对象 store_ 的超时时间
const std::chrono::milliseconds& PrefixStore::getTimeout() const noexcept {
  return store_->getTimeout();
}

// 设置存储对象 store_ 的超时时间
void PrefixStore::setTimeout(const std::chrono::milliseconds& timeout) {
  store_->setTimeout(timeout);
}

// 在 store_ 中给 prefix_ + "/" + key 添加指定的 value
void PrefixStore::append(
    const std::string& key,
    const std::vector<uint8_t>& value) {
  store_->append(joinKey(key), value);
}

// 从 store_ 中获取给定 keys 中每个键名（加上 prefix_ 后的）对应的值
std::vector<std::vector<uint8_t>> PrefixStore::multiGet(
    const std::vector<std::string>& keys) {
  std::vector<std::string> prefixed_keys;
  prefixed_keys.reserve(keys.size());
  for (auto& key : keys) {
    prefixed_keys.push_back(joinKey(key));
  }
  return store_->multiGet(prefixed_keys);
}

// 在 store_ 中设置给定 keys 中每个键名（加上 prefix_ 后的）对应的值
void PrefixStore::multiSet(
    const std::vector<std::string>& keys,
    const std::vector<std::vector<uint8_t>>& values) {
  std::vector<std::string> prefixed_keys;
  prefixed_keys.reserve(keys.size());
  for (auto& key : keys) {
    prefixed_keys.push_back(joinKey(key));
  }
  store_->multiSet(prefixed_keys, values);
}

// 返回存储对象 store_ 是否支持扩展 API（包括 append, multiGet 和 multiSet）
bool PrefixStore::hasExtendedApi() const {
  return store_->hasExtendedApi();
}

} // namespace c10d
// 返回存储指针的当前值（intrusive_ptr<Store>），即 store_ 的值
c10::intrusive_ptr<Store> PrefixStore::getUnderlyingStore() {
  return store_;
}

// 返回存储指针的当前值（intrusive_ptr<Store>），但排除所有的 PrefixStore
c10::intrusive_ptr<Store> PrefixStore::getUnderlyingNonPrefixStore() {
  // 复制 store_ 的当前值
  c10::intrusive_ptr<Store> store = store_;

  // 循环直到找到非 PrefixStore 为止
  while (store) {
    // 尝试动态转换为 PrefixStore
    PrefixStore* asPrefixStore = dynamic_cast<PrefixStore*>(store.get());
    if (asPrefixStore) {
      // 如果成功转换，则获取其底层存储并继续搜索
      store = asPrefixStore->getUnderlyingStore();
    } else {
      // 如果无法转换为 PrefixStore，退出循环（找到了非 PrefixStore）
      break;
    }
  }

  // 检查最终的存储指针是否为 nullptr，如果是则抛出异常
  TORCH_CHECK(
      store != nullptr, "Underlying Non-PrefixStore shouldn't be null.");
  // 返回最终的非 PrefixStore 存储指针
  return store;
}

// 命名空间 c10d 的结束标记
} // namespace c10d
```