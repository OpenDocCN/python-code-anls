# `.\pytorch\torch\csrc\distributed\c10d\Store.cpp`

```
// 包含 Torch 分布式库中的 Store 头文件
#include <torch/csrc/distributed/c10d/Store.hpp>

// 定义 c10d 命名空间
namespace c10d {

// 返回超时时间的引用
const std::chrono::milliseconds& Store::getTimeout() const noexcept {
  return timeout_;
}

// 设置超时时间的函数
void Store::setTimeout(const std::chrono::milliseconds& timeout) {
  timeout_ = timeout;
}

// 将键值对存储到 Store 中
void Store::set(const std::string& key, const std::string& value) {
  // 转换字符串值为字节向量并调用重载的 set 函数
  set(key, std::vector<uint8_t>(value.begin(), value.end()));
}

// 比较并设置键的值，返回新值
std::string Store::compareSet(
    const std::string& key,
    const std::string& currentValue,
    const std::string& newValue) {
  // 将字符串值转换为字节向量，并调用重载的 compareSet 函数
  auto value = compareSet(
      key,
      std::vector<uint8_t>(currentValue.begin(), currentValue.end()),
      std::vector<uint8_t>(newValue.begin(), newValue.end()));
  // 将字节向量转换为字符串返回
  return std::string(value.begin(), value.end());
}

// 获取键的值，并转换为字符串返回
std::string Store::get_to_str(const std::string& key) {
  auto value = get(key);
  return std::string(value.begin(), value.end());
}

// 在键的值末尾追加字节向量的数据
void Store::append(const std::string& key, const std::vector<uint8_t>& value) {
  // 初始化期望值为给定的字节向量
  std::vector<uint8_t> expected = value;
  std::vector<uint8_t> current;
  // 不能使用 get(key) 因为如果键不存在可能会永久阻塞，因此使用 compareSet
  current = compareSet(key, current, expected);
  // 循环直到当前值等于期望值
  while (current != expected) {
    expected = current;
    // 将新数据追加到期望值中
    expected.insert(expected.end(), value.begin(), value.end());
    current = compareSet(key, current, expected);
  }
}

// 批量获取多个键的值
std::vector<std::vector<uint8_t>> Store::multiGet(
    const std::vector<std::string>& keys) {
  // 初始化结果向量，并预留空间以容纳所有键的值
  std::vector<std::vector<uint8_t>> result;
  result.reserve(keys.size());
  // 遍历每个键，将其值添加到结果向量中
  for (auto& key : keys) {
    result.emplace_back(get(key));
  }
  return result;
}

// 批量设置多个键的值
void Store::multiSet(
    const std::vector<std::string>& keys,
    const std::vector<std::vector<uint8_t>>& values) {
  // 遍历所有键值对，并逐一调用 set 函数进行设置
  for (auto i : ::c10::irange(keys.size())) {
    set(keys[i], values[i]);
  }
}

// 判断是否具有扩展 API 的能力，总是返回 false
bool Store::hasExtendedApi() const {
  return false;
}

} // namespace c10d
```