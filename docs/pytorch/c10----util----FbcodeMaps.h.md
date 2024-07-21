# `.\pytorch\c10\util\FbcodeMaps.h`

```py
#ifndef C10_UTIL_FBCODEMAPS_H_
#define C10_UTIL_FBCODEMAPS_H_

// Map typedefs so that we can use folly's F14 maps in fbcode without
// taking a folly dependency.

#ifdef FBCODE_CAFFE2
#include <folly/container/F14Map.h>
#include <folly/container/F14Set.h>
#else
#include <unordered_map>
#include <unordered_set>
#endif

// 命名空间 c10 开始
namespace c10 {

// 根据条件选择使用 folly 的 F14FastMap 或者标准库的 std::unordered_map
#ifdef FBCODE_CAFFE2
// 定义 FastMap 为 folly::F14FastMap
template <typename Key, typename Value>
using FastMap = folly::F14FastMap<Key, Value>;
// 定义 FastSet 为 folly::F14FastSet
template <typename Key>
using FastSet = folly::F14FastSet<Key>;
#else
// 定义 FastMap 为 std::unordered_map
template <typename Key, typename Value>
using FastMap = std::unordered_map<Key, Value>;
// 定义 FastSet 为 std::unordered_set
template <typename Key>
using FastSet = std::unordered_set<Key>;
#endif

} // namespace c10

#endif // C10_UTIL_FBCODEMAPS_H_
```