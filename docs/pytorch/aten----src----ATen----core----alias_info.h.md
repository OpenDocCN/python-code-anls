# `.\pytorch\aten\src\ATen\core\alias_info.h`

```py
#pragma once
#include <unordered_set>  // 引入无序集合（哈希集合）头文件
#include <vector>         // 引入向量头文件
#include <ATen/core/symbol.h>     // 引入ATen库中的symbol头文件
#include <c10/util/Exception.h>   // 引入c10库中的Exception头文件
#include <c10/util/hash.h>        // 引入c10库中的hash头文件

namespace c10 {
/**
 * class AliasInfo
 *
 * Data structure to hold aliasing information for an `Argument`. They can be
 * nested to represent aliasing information on contained types.
 *
 * There is a `beforeSet` which describes the aliasing information before the
 * operator executes, and an `afterSet` that describes aliasing info
 * after execution.
 */
class AliasInfo {
 public:
  // Symbol for the set that can alias anything
  static Symbol wildcardSet() {
    static const Symbol wc = Symbol::fromQualString("alias::*");  // 定义静态方法返回能够别名任何东西的符号
    return wc;
  }

  void setIsWrite(bool isWrite) {  // 设置是否写操作
    isWrite_ = isWrite;
  }

  bool isWrite() const {  // 判断是否为写操作
    return isWrite_;
  }

  void addBeforeSet(Symbol aliasSet) {  // 添加操作前别名集合
    beforeSets_.insert(aliasSet);
  }

  void addAfterSet(Symbol aliasSet) {  // 添加操作后别名集合
    afterSets_.insert(aliasSet);
  }

  const std::unordered_set<Symbol>& beforeSets() const {  // 获取操作前别名集合的引用
    return beforeSets_;
  }

  const std::unordered_set<Symbol>& afterSets() const {  // 获取操作后别名集合的引用
    return afterSets_;
  }

  Symbol beforeSet() const {  // 获取操作前的别名集合
    AT_ASSERT(beforeSets_.size() == 1);  // 断言操作前的别名集合大小为1
    return *beforeSets_.begin();
  }

  bool isWildcardBefore() const {  // 判断操作前是否有通配符
    return beforeSets_.count(wildcardSet()) != 0;
  }

  bool isWildcardAfter() const {  // 判断操作后是否有通配符
    return afterSets_.count(wildcardSet()) != 0;
  }

  // the alias info for the contained types of the type
  // e.g. if this is an annotation on List[T], `sets` refers to
  // the alias sets that the list may be in
  // while containedTypes()[0] refers to the sets that members of the list
  // may be in
  void addContainedType(AliasInfo aliasInfo) {  // 添加包含类型的别名信息
    containedTypes_.push_back(std::move(aliasInfo));
  }
  const std::vector<AliasInfo>& containedTypes() const {  // 获取包含类型的别名信息的引用
    return containedTypes_;
  }

 private:
  std::unordered_set<Symbol> beforeSets_;  // 操作前的别名集合
  std::unordered_set<Symbol> afterSets_;   // 操作后的别名集合
  std::vector<AliasInfo> containedTypes_;  // 包含类型的别名信息的向量
  bool isWrite_ = false;  // 是否写操作的标志
};

inline bool operator==(const AliasInfo& lhs, const AliasInfo& rhs) {  // 定义AliasInfo类对象的相等比较操作符
  return lhs.isWrite() == rhs.isWrite()
      && lhs.beforeSets() == rhs.beforeSets()
      && lhs.afterSets() == rhs.afterSets()
      && lhs.containedTypes() == rhs.containedTypes();
}

// this does match the way things are represented in the schema
inline std::ostream& operator<<(std::ostream& out, const AliasInfo& aliasInfo) {  // 定义AliasInfo类对象的输出流操作符
  out << "(";
  bool first = true;
  for (const auto& set : aliasInfo.beforeSets()) {  // 遍历操作前别名集合
    if (first) {
      first = false;
    } else {
      out << "|";
    }
    out << set.toUnqualString();  // 输出未限定字符串形式的别名集合
  }
  if (aliasInfo.isWrite()) {  // 如果是写操作
    out << "!";
  }
  if (aliasInfo.beforeSets() != aliasInfo.afterSets()) {  // 如果操作前后的别名集合不同
    out << " -> ";
    first = true;
    for (const auto& set : aliasInfo.afterSets()) {  // 遍历操作后别名集合
      if (first) {
        first = false;
      } else {
        out << "|";
      }
      out << set.toUnqualString();  // 输出未限定字符串形式的别名集合
    }
  }
  out << ")";
  return out;
}
} // namespace c10
// 定义 std 命名空间，重载 std::hash 结构体模板的模板特化版本，针对 c10::AliasInfo 类型
namespace std {
template <>
  struct hash<c10::AliasInfo> {
    // 定义哈希函数，接受 c10::AliasInfo 类型参数 aliasInfo，返回哈希值 size_t
    size_t operator()(const c10::AliasInfo& aliasInfo) const {
      // 计算基于 aliasInfo.isWrite() 的布尔值哈希
      auto hash = std::hash<bool>()(aliasInfo.isWrite());

      // NOTE: 对于无序集合的哈希值，我们无法使用 hash_combine，
      // 因为 hash_combine 是依赖顺序的。因此，我们选择使用 XOR 作为组合函数，
      // 因为 XOR 是可交换的。
      // 初始化 before_set_hash_seed 为 0
      size_t before_set_hash_seed = 0;
      // 遍历 aliasInfo.beforeSets() 中的每个元素 e
      for (auto &e: aliasInfo.beforeSets()) {
        // 计算 c10::Symbol 类型 e 的哈希值
        auto symbol_hash = std::hash<c10::Symbol>()(e);
        // 将 symbol_hash 与 before_set_hash_seed 进行 XOR 操作
        before_set_hash_seed = before_set_hash_seed ^ symbol_hash;
      }
      // 初始化 after_set_hash_seed 为 0
      size_t after_set_hash_seed = 0;
      // 遍历 aliasInfo.afterSets() 中的每个元素 e
      for (auto &e: aliasInfo.afterSets()) {
        // 计算 c10::Symbol 类型 e 的哈希值
        auto symbol_hash = std::hash<c10::Symbol>()(e);
        // 将 symbol_hash 与 after_set_hash_seed 进行 XOR 操作
        after_set_hash_seed = after_set_hash_seed ^ symbol_hash;
      }

      // 使用 c10::hash_combine 函数将 before_set_hash_seed 合并到 hash 中
      hash = c10::hash_combine(hash, before_set_hash_seed);
      // 使用 c10::hash_combine 函数将 after_set_hash_seed 合并到 hash 中
      hash = c10::hash_combine(hash, after_set_hash_seed);
      // 遍历 aliasInfo.containedTypes() 中的每个元素 e
      for (auto &e: aliasInfo.containedTypes()) {
        // 计算 c10::AliasInfo 类型 e 的哈希值
        auto contained_type_hash = std::hash<c10::AliasInfo>()(e);
        // 使用 c10::hash_combine 函数将 contained_type_hash 合并到 hash 中
        hash = c10::hash_combine(hash, contained_type_hash);
      }
      // 返回最终计算得到的哈希值
      return hash;
    }
  };
}
```