# `.\pytorch\aten\src\ATen\core\qualified_name.h`

```
#pragma once

#include <c10/util/ArrayRef.h>
#include <c10/util/Exception.h>
#include <c10/util/StringUtil.h>
#include <c10/util/irange.h>
#include <string>

namespace c10 {

// 表示形如 "foo.bar.baz" 的名称结构体
struct QualifiedName {
  QualifiedName() = default;

  // `name` 可以是点分隔的字符串，例如 "foo.bar.baz"，或者单独的名称。
  /* implicit */ QualifiedName(const std::string& name) {
    // 检查名称不为空
    TORCH_CHECK(!name.empty());
    // 使用特定的分隔符来分割字符串成单个部分。
    size_t startSearchFrom = 0;
    size_t pos = name.find(delimiter_, startSearchFrom);

    while (pos != std::string::npos) {
      auto atom = name.substr(startSearchFrom, pos - startSearchFrom);
      // 检查每个部分不为空
      TORCH_INTERNAL_ASSERT(
          !atom.empty(), "Invalid name for qualified name: '", name, "'");
      // 将每个部分存入 atoms_ 中
      atoms_.push_back(std::move(atom));
      startSearchFrom = pos + 1;
      pos = name.find(delimiter_, startSearchFrom);
    }

    auto finalAtom = name.substr(startSearchFrom);
    // 检查最后一个部分不为空
    TORCH_INTERNAL_ASSERT(
        !finalAtom.empty(), "Invalid name for qualified name: '", name, "'");
    // 将最后一个部分存入 atoms_ 中
    atoms_.emplace_back(std::move(finalAtom));

    // 生成缓存的访问器
    cacheAccessors();
  }

  // 使用 vector<string> 初始化 QualifiedName 对象
  explicit QualifiedName(std::vector<std::string> atoms) : atoms_(std::move(atoms)) {
    // 检查每个原子部分不为空，并且不包含分隔符
    for (const auto& atom : atoms_) {
      TORCH_CHECK(!atom.empty(), "Atom cannot be empty");
      TORCH_CHECK(
          atom.find(delimiter_) == std::string::npos,
          "Delimiter not allowed in atom");
    }

    // 生成缓存的访问器
    cacheAccessors();
  }

  // 无需复制。理想情况下，应使用 std::string_view 等。
  /* implicit */ QualifiedName(const char* name)
      : QualifiedName(std::string(name)) {}

  // 使用前缀和名称初始化 QualifiedName 对象
  // `name` 必须是单个名称（不包含点！）
  explicit QualifiedName(const QualifiedName& prefix, std::string name) {
    // 检查名称不为空，并且不包含分隔符
    TORCH_INTERNAL_ASSERT(!name.empty());
    TORCH_INTERNAL_ASSERT(name.find(delimiter_) == std::string::npos);
    // 将前缀的 atoms_ 部分和名称部分整合成新的 atoms_
    atoms_.insert(atoms_.begin(), prefix.atoms_.begin(), prefix.atoms_.end());
    atoms_.push_back(std::move(name));

    // 生成缓存的访问器
    cacheAccessors();
  }

  // 检查 `this` 是否是 `other` 的前缀？
  // 例如，"foo.bar" 是 "foo.bar.baz" 的前缀
  bool isPrefixOf(const QualifiedName& other) const {
    const auto& thisAtoms = atoms_;
    const auto& otherAtoms = other.atoms_;

    if (thisAtoms.size() > otherAtoms.size()) {
      // 如果 atoms_ 大小超过 otherAtoms，则不可能是前缀
      return false;
    }
    for (const auto i : c10::irange(thisAtoms.size())) {
      if (thisAtoms[i] != otherAtoms[i]) {
        return false;
      }
    }
    return true;
  }

  // 完整的限定名称，例如 "foo.bar.baz"
  const std::string& qualifiedName() const {
    return qualifiedName_;
  }

  // 前导限定符，例如 "foo.bar"
  const std::string& prefix() const {
    return prefix_;
  }

  // 基本名称，例如 "baz"
  const std::string& name() const {
    return name_;
  }

  // 原子部分的 vector
  const std::vector<std::string>& atoms() const {
    return atoms_;
  }

  // 比较运算符重载，比较两个 QualifiedName 是否相等
  bool operator==(const QualifiedName& other) const {
  // 检查当前对象的限定名是否等于另一个对象的限定名
  return this->qualifiedName_ == other.qualifiedName_;
}

// 检查当前对象的限定名是否不等于另一个对象的限定名
bool operator!=(const QualifiedName& other) const {
  return !(*this == other);
}

private:
static constexpr char delimiter_ = '.';

// 下面是 cacheAccessors() 的辅助函数
template<typename T>
// 使用指定的分隔符将容器中的元素连接成一个字符串
std::string join(char delimiter, const T& v) {
  std::string out;
  size_t reserve = 0;
  for (const auto& e : v) {
    reserve += e.size() + 1;
  }
  out.reserve(reserve);
  for (const auto i : c10::irange(v.size())) {
    if (i != 0) {
      out.push_back(delimiter);
    }
    out.append(v[i]);
  }
  return out;
}

// 缓存访问器函数，根据 atoms_ 更新缓存的限定名、前缀和名称
void cacheAccessors() {
  // 更新限定名为 atoms_ 的连接结果，使用 delimiter_ 分隔符
  qualifiedName_ = join(delimiter_, atoms_);
  if (atoms_.size() > 1) {
    ArrayRef<std::string> view(atoms_);
    // 根据 atoms_ 的前缀部分更新缓存的前缀
    const auto prefixView = view.slice(0, view.size() - 1);
    prefix_ = join(delimiter_, prefixView);
  }

  // 如果 atoms_ 非空，更新缓存的名称为 atoms_ 的最后一个元素
  if (!atoms_.empty()) {
    name_ = atoms_.back();
  }
}

// 实际的名称列表，如 "{foo, bar, baz}"
std::vector<std::string> atoms_;

/*
 * 从 `atoms_` 派生的缓存访问器。
 */
std::string qualifiedName_; // 缓存的限定名
std::string prefix_;        // 缓存的前缀
std::string name_;          // 缓存的名称
};
} // namespace c10



// 结束 c10 命名空间的定义

namespace std {
template <>
struct hash<c10::QualifiedName> {
  // 定义模板特化：计算 c10::QualifiedName 类型对象的哈希值
  size_t operator()(const c10::QualifiedName& n) const noexcept {
    // 调用 std::string 类型的哈希函数，计算 QualifiedName 对象的 qualifiedName() 的哈希值并返回
    return std::hash<std::string>()(n.qualifiedName());
  }
};
} // namespace std



// 结束 std 命名空间的定义


这段代码定义了一个特化的哈希函数模板 `std::hash<c10::QualifiedName>`，用于计算 `c10::QualifiedName` 类型对象的哈希值。
```