# `.\pytorch\torch\csrc\api\include\torch\ordered_dict.h`

```py
#pragma once

#include <cstdint>
#include <initializer_list>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace torch {
/// An ordered dictionary implementation, akin to Python's `OrderedDict`.
template <typename Key, typename Value>
class OrderedDict {
 public:
  /// Nested class representing an item in the ordered dictionary.
  class Item {
   public:
    /// Constructs a new item with a key and a value.
    Item(Key key, Value value) : pair_(std::move(key), std::move(value)) {}

    /// Returns a reference to the value.
    Value& operator*() {
      return value();
    }

    /// Returns a reference to the value (const version).
    const Value& operator*() const {
      return value();
    }

    /// Allows access to the value using the arrow operator.
    Value* operator->() {
      return &value();
    }

    /// Allows access to the value using the arrow operator (const version).
    const Value* operator->() const {
      return &value();
    }

    /// Returns a reference to the key.
    const Key& key() const noexcept {
      return pair_.first;
    }

    /// Returns a reference to the value.
    Value& value() noexcept {
      return pair_.second;
    }

    /// Returns a reference to the value (const version).
    const Value& value() const noexcept {
      return pair_.second;
    }

    /// Returns the pair (key, value).
    const std::pair<Key, Value>& pair() const noexcept {
      return pair_;
    }

   private:
    /// The pair representing key-value.
    ::std::pair<Key, Value> pair_;
  };

  /// Constructs an ordered dictionary with a key description.
  explicit OrderedDict(std::string key_description);

  /// Copy constructor for OrderedDict.
  OrderedDict(const OrderedDict& other);

  /// Assignment operator for OrderedDict.
  OrderedDict& operator=(const OrderedDict& other);

  /// Constructs an ordered dictionary from an initializer list of items.
  OrderedDict(std::initializer_list<Item> initializer_list);

 private:
  /// Vector to maintain the order of items.
  std::vector<Item> items_;

  /// Map to store indices of items for fast lookup.
  std::unordered_map<Key, size_t> index_;

  /// Description of the key.
  std::string key_description_;
};
}  // namespace torch


注释：
# 返回有序字典的起始迭代器
typename OrderedDict<Key, Value>::Iterator OrderedDict<Key, Value>::begin() {
  return items_.begin();
}

# 返回有序字典的常量起始迭代器
template <typename Key, typename Value>
typename OrderedDict<Key, Value>::ConstIterator OrderedDict<Key, Value>::begin()
    const {
  return items_.begin();
}

# 返回有序字典的结束迭代器
template <typename Key, typename Value>
typename OrderedDict<Key, Value>::Iterator OrderedDict<Key, Value>::end() {
  return items_.end();
}

# 返回有序字典的常量结束迭代器
template <typename Key, typename Value>
typename OrderedDict<Key, Value>::ConstIterator OrderedDict<Key, Value>::end()
    const {
  return items_.end();
}

# 返回有序字典的第一个元素的引用
template <typename Key, typename Value>
typename OrderedDict<Key, Value>::Item& OrderedDict<Key, Value>::front() {
  TORCH_CHECK(!items_.empty(), "Called front() on an empty OrderedDict");
  return items_.front();
}

# 返回有序字典的第一个元素的常量引用
template <typename Key, typename Value>
const typename OrderedDict<Key, Value>::Item& OrderedDict<Key, Value>::front()
    const {
  TORCH_CHECK(!items_.empty(), "Called front() on an empty OrderedDict");
  return items_.front();
}

# 返回有序字典的最后一个元素的引用
template <typename Key, typename Value>
typename OrderedDict<Key, Value>::Item& OrderedDict<Key, Value>::back() {
  TORCH_CHECK(!items_.empty(), "Called back() on an empty OrderedDict");
  return items_.back();
}

# 返回有序字典的最后一个元素的常量引用
template <typename Key, typename Value>
const typename OrderedDict<Key, Value>::Item& OrderedDict<Key, Value>::back()
    const {
  TORCH_CHECK(!items_.empty(), "Called back() on an empty OrderedDict");
  return items_.back();
}

# 返回有序字典指定索引位置的元素的引用
template <typename Key, typename Value>
typename OrderedDict<Key, Value>::Item& OrderedDict<Key, Value>::operator[](
    size_t index) {
  TORCH_CHECK(index < items_.size(), "Index ", index, " is out of bounds");
  return items_[index];
}

# 返回有序字典指定索引位置的元素的常量引用
template <typename Key, typename Value>
const typename OrderedDict<Key, Value>::Item& OrderedDict<Key, Value>::
operator[](size_t index) const {
  TORCH_CHECK(index < items_.size(), "Index ", index, " is out of bounds");
  return items_[index];
}

# 返回有序字典中指定键的值的引用
template <typename Key, typename Value>
Value& OrderedDict<Key, Value>::operator[](const Key& key) {
  if (auto* value = find(key)) {
    return *value;
  }
  AT_ERROR(key_description_, " '", key, "' is not defined");
}

# 返回有序字典中指定键的值的常量引用
template <typename Key, typename Value>
const Value& OrderedDict<Key, Value>::operator[](const Key& key) const {
  if (auto* value = find(key)) {
    return *value;
  }
  AT_ERROR(key_description_, " '", key, "' is not defined");
}

# 插入键值对到有序字典，返回值的引用
template <typename Key, typename Value>
template <typename K, typename V>
Value& OrderedDict<Key, Value>::insert(K&& key, V&& value) {
  TORCH_CHECK(
      index_.count(key) == 0, key_description_, " '", key, "' already defined");
  // 在这里复制 `key` 并将其移动到索引中
  items_.emplace_back(key, std::forward<V>(value));
  index_.emplace(std::forward<K>(key), size() - 1);
  return items_.back().value();
}

# 插入键值对到有序字典，返回值的引用
template <typename Key, typename Value>
Value& OrderedDict<Key, Value>::insert(Key key, Value&& value) {
  return insert<Key, Value>(std::move(key), std::move(value));
}
// 更新有序字典，从另一个移动构造的有序字典中获取元素
template <typename Key, typename Value>
void OrderedDict<Key, Value>::update(OrderedDict&& other) {
  // 预留足够的空间以容纳当前字典和另一个字典的所有元素
  reserve(size() + other.size());
  // 遍历另一个字典中的每个项
  for (auto& item : other) {
    // 调用 `insert()` 方法插入元素，以避免重复的键
    insert(std::move(item.key()), std::move(item.value()));
  }
}

// 更新有序字典，从另一个常量引用的有序字典中获取元素
template <typename Key, typename Value>
void OrderedDict<Key, Value>::update(const OrderedDict& other) {
  // 预留足够的空间以容纳当前字典和另一个字典的所有元素
  reserve(size() + other.size());
  // 遍历另一个字典中的每个项
  for (auto& item : other) {
    // 调用 `insert()` 方法插入元素，以避免重复的键
    insert(item.key(), item.value());
  }
}

// 查找指定键对应的值的指针（非常量版本）
template <typename Key, typename Value>
Value* OrderedDict<Key, Value>::find(const Key& key) noexcept {
  // 在索引中查找指定的键
  auto iterator = index_.find(key);
  // 如果键不存在，则返回空指针
  if (iterator == index_.end()) {
    return nullptr;
  }
  // 返回找到的值的指针
  return &items_[iterator->second].value();
}

// 查找指定键对应的值的指针（常量版本）
template <typename Key, typename Value>
const Value* OrderedDict<Key, Value>::find(const Key& key) const noexcept {
  // 在索引中查找指定的键
  auto iterator = index_.find(key);
  // 如果键不存在，则返回空指针
  if (iterator == index_.end()) {
    return nullptr;
  }
  // 返回找到的值的指针
  return &items_[iterator->second].value();
}

// 删除指定键对应的项
template <typename Key, typename Value>
void OrderedDict<Key, Value>::erase(const Key& key) {
  // 在索引中查找指定的键
  auto it = index_.find(key);
  // 检查是否找到键，否则抛出异常
  TORCH_CHECK(it != index_.end(), "Key '", key, "' doesn't exist");
  
  // 获取键对应的索引
  auto index = it->second;
  // 从索引中删除键
  index_.erase(it);
  // 从项中删除索引
  items_.erase(items_.begin() + index);

  // 更新所有后续项的索引值
  for (auto& pair : index_)
    if (pair.second > index)
      --pair.second;
}

// 检查字典中是否包含指定键
template <typename Key, typename Value>
bool OrderedDict<Key, Value>::contains(const Key& key) const noexcept {
  // 调用 `find()` 方法检查键是否存在
  return find(key) != nullptr;
}

// 清空有序字典
template <typename Key, typename Value>
void OrderedDict<Key, Value>::clear() {
  // 清空索引
  index_.clear();
  // 清空项
  items_.clear();
}

// 返回有序字典中的项数
template <typename Key, typename Value>
size_t OrderedDict<Key, Value>::size() const noexcept {
  // 返回项的数量
  return items_.size();
}

// 检查有序字典是否为空
template <typename Key, typename Value>
bool OrderedDict<Key, Value>::is_empty() const noexcept {
  // 检查项是否为空
  return items_.empty();
}

// 返回有序字典的键描述
template <typename Key, typename Value>
const std::string& OrderedDict<Key, Value>::key_description() const noexcept {
  // 返回键描述字符串的引用
  return key_description_;
}

// 返回有序字典的所有项（常量版本）
template <typename Key, typename Value>
const std::vector<typename OrderedDict<Key, Value>::Item>& OrderedDict<
    Key,
    Value>::items() const noexcept {
  // 返回项向量的引用
  return items_;
}

// 返回有序字典的所有键
template <typename Key, typename Value>
::std::vector<Key> OrderedDict<Key, Value>::keys() const {
  // 创建用于存储所有键的向量
  std::vector<Key> keys;
  // 预留足够的空间以容纳所有键
  keys.reserve(size());
  // 遍历所有项，并将键添加到向量中
  for (const auto& item : items_) {
    keys.push_back(item.key());
  }
  // 返回包含所有键的向量
  return keys;
}

// 返回有序字典的所有值
template <typename Key, typename Value>
::std::vector<Value> OrderedDict<Key, Value>::values() const {
  // 创建用于存储所有值的向量
  std::vector<Value> values;
  // 预留足够的空间以容纳所有值
  values.reserve(size());
  // 遍历所有项，并将值添加到向量中
  for (const auto& item : items_) {
    values.push_back(item.value());
  }
  // 返回包含所有值的向量
  return values;
}

// 返回有序字典的所有键值对
template <typename Key, typename Value>
::std::vector<std::pair<Key, Value>> OrderedDict<Key, Value>::pairs() const {
  // 创建用于存储所有键值对的向量
  std::vector<std::pair<Key, Value>> values;
  // 预留足够的空间以容纳所有键值对
  values.reserve(size());
  // 遍历所有项，并将键值对添加到向量中
  for (const auto& item : items_) {
    values.push_back(item.pair());

# 将 item 的 pair 数据添加到 values 的末尾


  }
  return values;

# 返回存储了所有 item pair 数据的 values 容器
}

template <typename Key, typename Value>
void OrderedDict<Key, Value>::reserve(size_t requested_capacity) {
  // 为 index_ 预留指定容量
  index_.reserve(requested_capacity);
  // 为 items_ 预留指定容量
  items_.reserve(requested_capacity);
}

template <typename K, typename V>
bool operator==(
    const torch::OrderedDict<K, V>& a,
    const torch::OrderedDict<K, V>& b) {
  // 使用类型别名简化代码
  using Item = typename torch::OrderedDict<K, V>::Item;
  // 检查索引是否相等
  if (a.index_ != b.index_)
    return false;
  // 检查 items_ 的大小是否相等
  if (a.items_.size() != b.items_.size())
    return false;
  // 注意：对于 items_，不需要比较键，因为已经知道索引相等
  // 比较两个 OrderedDict 的 items_ 内容是否一致
  return std::equal(
      a.items_.begin(),
      a.items_.end(),
      b.items_.begin(),
      [](const Item& a, const Item& b) { return a.value() == b.value(); });
}

} // namespace torch
```