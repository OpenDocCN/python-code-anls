# `.\pytorch\torch\csrc\lazy\core\cache.h`

```
/**
 * Cache utils in this file is adapted from PyTorch/XLA
 * https://github.com/pytorch/xla/blob/master/third_party/xla_client/cache.h
 */

#pragma once

#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>

namespace torch {
namespace lazy {

// Generic key and object cache with LRU expiration policy. The objects of type
// T will be stored as std::shared_ptr<T> and taken and returned as such, by the
// cache API.
template <
    typename K,
    typename T,
    typename H = std::hash<K>,
    typename E = std::equal_to<K>>
class Cache {
 public:
  using TypePtr = std::shared_ptr<T>;
  using Element = std::pair<K, TypePtr>;

  // Constructor initializing the cache with a maximum size.
  explicit Cache(size_t max_size) : max_size_(max_size) {}

  // Adds an object to the cache, unless it already exists. If the cache grows
  // beyond the limit set during construction, the oldest used object will be
  // removed from the cache.
  TypePtr Add(K key, TypePtr object) {
    // If max_size_ is 0, do nothing and return the object.
    if (!max_size_) {
      return object;
    }
    // Locks the cache for thread safety.
    std::lock_guard<std::mutex> slock(lock_);
    // Emplaces the new element at the front of the list.
    element_list_.emplace_front(Element(std::move(key), std::move(object)));
    auto it = element_list_.begin();
    // Inserts the element into the map, checking for existing entries.
    auto emplace_result = element_map_.emplace(&it->first, it);
    if (!emplace_result.second) {
      // If the element already exists, erase the newly added element.
      element_list_.erase(it);
      // Perform LRU eviction on the existing element.
      DoLRU(emplace_result.first->second);
    } else if (element_list_.size() > max_size_) {
      // If cache exceeds max size, remove the least recently used element.
      Element* last = &element_list_.back();
      element_map_.erase(&last->first);
      element_list_.pop_back();
    }
    // Return the stored or evicted object.
    return emplace_result.first->second->second;
  }

  // Retrieves the existing object if it exists. If found, moves it to the head
  // of the LRU list.
  // Returns nullptr if no object with the specified key is found within the
  // cache.
  TypePtr Get(const K& key) {
    if (!max_size_) {
      return nullptr;
    }
    // Locks the cache for thread safety.
    std::lock_guard<std::mutex> slock(lock_);
    // Finds the element in the map based on the key.
    auto it = element_map_.find(&key);
    if (it == element_map_.end()) {
      // If element not found, return nullptr.
      return nullptr;
    }
    // Moves the found element to the front of the LRU list.
    DoLRU(it->second);
    // Returns the stored object.
    return it->second->second;
  }

  // Retrieves the most recently added object from the cache.
  TypePtr GetLatest() {
    // Locks the cache for thread safety.
    std::lock_guard<std::mutex> g(lock_);
    // Ensures the cache is not empty before retrieval.
    TORCH_CHECK(!element_list_.empty());
    // Returns the object at the front of the LRU list.
    return element_list_.front().second;
  }

  // Removes an object from the cache based on the provided key.
  // Returns true if successful, false if the key does not exist in the cache.
  bool Erase(const K& key) {
    if (!max_size_) {
      return false;
    }
    // Locks the cache for thread safety.
    std::lock_guard<std::mutex> slock(lock_);
    // Finds the element in the map based on the key.
    auto it = element_map_.find(&key);
    if (it == element_map_.end()) {
      // If element not found, return false.
      return false;
    }
    // Erases the element from both the map and the list.
    auto lit = it->second;
    element_map_.erase(it);
    element_list_.erase(lit);
    // Returns true indicating successful removal.
    return true;
  }

  // Clears the entire cache, removing all elements.
  void Clear() {
    if (!max_size_) {
      return;
    }
    // Locks the cache for thread safety.
    std::lock_guard<std::mutex> slock(lock_);
    // Clears both the map and the list of elements.
    element_map_.clear();
    element_list_.clear();
  }

  // Returns the current number of elements in the cache.
  int Numel() const {
    if (!max_size_) {
      return 0;
    }
    // Locks the cache for thread safety.
    std::lock_guard<std::mutex> g(lock_);
    // Ensures consistency between map and list sizes.
    TORCH_CHECK(element_map_.size() == element_list_.size());
    // Returns the size of the cache.
    return element_map_.size();
  }

 private:
  size_t max_size_;  // Maximum size of the cache.
  std::list<Element> element_list_;               // List storing elements in LRU order.
  std::unordered_map<const K*, typename std::list<Element>::iterator, H, E> element_map_;  // Map for fast element lookup.
  mutable std::mutex lock_;                      // Mutex for thread safety.

  // Performs LRU eviction by moving the specified iterator to the front of the list.
  void DoLRU(typename std::list<Element>::iterator it) {
    element_list_.splice(element_list_.begin(), element_list_, it);
    element_map_[&it->first] = element_list_.begin();
  }
};

}  // namespace lazy
}  // namespace torch
    return element_map_.size();
  }



# 返回当前元素映射（element_map_）的大小
  }



 private:
  using ElementList = std::list<Element>;

  struct Hasher {
    size_t operator()(const K* key) const {
      return hasher(*key);
    }

    H hasher;
  };



# 定义一个哈希函数结构体 Hasher，用于将指针类型 K* 的键进行哈希计算
    H hasher;
  };



  struct Equaler {
    bool operator()(const K* k1, const K* k2) const {
      return equaler(*k1, *k2);
    }

    E equaler;
  };



# 定义一个相等比较函数结构体 Equaler，用于比较两个指针类型 K* 的键是否相等
    E equaler;
  };



  using ElementMap = std::
      unordered_map<const K*, typename ElementList::iterator, Hasher, Equaler>;



# 使用 std::unordered_map 定义 ElementMap 类型，键为 const K* 指针类型，值为 ElementList 迭代器类型
      unordered_map<const K*, typename ElementList::iterator, Hasher, Equaler>;



  void DoLRU(typename ElementList::iterator it) {
    element_list_.splice(element_list_.begin(), element_list_, it);
  }



# 实现 DoLRU 方法，用于执行最近最少使用（LRU）缓存淘汰策略，将元素列表中的指定迭代器 it 所指向的元素移到列表头部
  }



  mutable std::mutex lock_;



# 可变的互斥锁，用于保护对象的并发访问
  const size_t max_size_ = 0;



# 常量成员变量 max_size_，表示缓存的最大大小
  ElementList element_list_;



# 元素列表，存储缓存中的所有元素
  ElementMap element_map_;



# 元素映射，将键（const K* 类型的指针）映射到其在元素列表中的位置（ElementList::iterator 类型）
};

} // namespace lazy
} // namespace torch
```