# `.\pytorch\aten\src\ATen\TensorNames.cpp`

```py
// 引入 ATen 库中的头文件，包括 TensorNames 和 WrapDimUtils
#include <ATen/TensorNames.h>
#include <ATen/WrapDimUtils.h>
// 引入 c10 库中的实用工具 irange
#include <c10/util/irange.h>

// 命名空间 at::namedinference 下的实现
namespace at::namedinference {

// 实现 TensorName 类的 toDimname 方法
Dimname TensorName::toDimname() const {
  // 直接返回成员变量 name_
  return name_;
}

// 实现 TensorName 类的 unify 方法
const TensorName& TensorName::unify(const TensorName& other, const char* op_name) const {
  // 处理 unify(None, None) 的情况
  if (name_.isWildcard() && other.name_.isWildcard()) {
    return *this;
  }

  // 处理 unify(A, A) 的情况
  if (name_ == other.name_) {
    return *this;
  }

  // 处理 unify(A, None) 的情况
  if (other.name_.isWildcard()) {
    // 在 other.origin_ 中查找 name_ 是否存在
    const auto it = std::find(other.origin_.begin(), other.origin_.end(), name_);
    // 如果找到，抛出错误
    TORCH_CHECK(it == other.origin_.end(),
        op_name, ":", " Cannot match ", *this, " with ", other,
        " because the latter names already have ", name_, ".",
        " Are your tensors misaligned?");
    return *this;
  }

  // 处理 unify(None, A) 的情况
  if (name_.isWildcard()) {
    // 调用 other 对象的 unify 方法
    return other.unify(*this, op_name);
  }

  // 处理 unify(A, B) 的情况，此时 A != B
  TORCH_CHECK(name_ == other.name_,
      op_name, ":", " Expected ", *this,
      " to match ", other,
      " but they do not match.");
  return *this;
}

// 实现 TensorNames 类的构造函数，接受 Dimname 数组作为参数
TensorNames::TensorNames(ArrayRef<Dimname> names) {
  // 预留足够的空间以容纳 names 的大小
  names_.reserve(names.size());
  // 遍历 names 数组，用每个元素初始化 names_
  for (const auto idx : c10::irange(names.size())) {
    names_.emplace_back(names, idx);
  }
}

// 实现 TensorNames 类的构造函数，接受 Dimname 数组和两个整数作为参数
TensorNames::TensorNames(ArrayRef<Dimname> names, int64_t start, int64_t end) {
  // 将 names 数组的大小转换为 int64_t 类型
  int64_t names_size = static_cast<int64_t>(names.size());
  // 对 start 和 end 进行可能的封装
  start = maybe_wrap_dim(start, names_size);
  end = maybe_wrap_dim(end, names_size);
  // 预留足够的空间以容纳从 start 到 end 的元素数量
  names_.reserve(end - start);
  // 遍历 names 数组的子范围，用每个元素初始化 names_
  for (const auto idx : c10::irange(start, end)) {
    names_.emplace_back(names, idx);
  }
}

// 实现 TensorNames 类的 unifyFromRightInplace 方法，就地对 names_ 进行右侧的统一操作
TensorNames& TensorNames::unifyFromRightInplace(const TensorNames& other, const char* op_name) {
  // 如果 names_ 的大小大于 other.names_ 的大小
  if (names_.size() > other.names_.size()) {
    const auto size_diff = names_.size() - other.names_.size();
    // 对 names_ 的后部分进行统一操作
    for (const auto idx : c10::irange(size_diff, names_.size())) {
      names_[idx] = names_[idx].unify(other.names_[idx - size_diff], op_name);
    }
  } else {
    const auto size_diff = other.names_.size() - names_.size();
    // 将 names_ 扩展到与 other.names_ 相同的长度，然后再进行统一操作
    names_.insert(
        names_.begin(),
        other.names_.begin(),
        other.names_.begin() + size_diff);
    // 对 names_ 的后部分进行统一操作
    for (const auto idx : c10::irange(size_diff, names_.size())) {
      names_[idx] = names_[idx].unify(other.names_[idx], op_name);
    }
  }

  return *this;
}

// 实现 TensorNames 类的 append 方法，向 names_ 中添加新的 TensorName
void TensorNames::append(TensorName name) {
  names_.emplace_back(name);
}

// 实现 TensorNames 类的 checkUnique 方法，检查 names_ 中的元素是否唯一
void TensorNames::checkUnique(const char* op_name) const {
  // O(N^2) 的复杂度，但由于命名张量最多有 N = 64 个维度，因此这里的性能不是问题
  // 如果需要优化，可以使用一种集合数据结构，但对于小规模的情况，这可能会带来额外的开销
  for (auto it = names_.begin(); it != names_.end(); ++it) {
    // 将当前 TensorName 转换为 Dimname
    const auto name = it->toDimname();
    // 如果是通配符，则继续下一个循环
    if (name.isWildcard()) continue;
    # 在 names_ 列表中查找从 it 的下一个元素开始，第一个满足条件的元素
    auto dup = std::find_if(it + 1, names_.end(),
        [&](const TensorName& other) { return other.toDimname() == name; });
    # 检查是否找到重复的名称
    TORCH_CHECK(dup == names_.end(),
        op_name, ": ",
        "Attempted to propagate dims ", *it, " and ", *dup, " to the output, ",
        "but that would create a tensor with duplicate names [", toDimnameVec(),
        "]. Please rename your inputs with Tensor.rename to prevent this.");
  }
}

// 结束 at::namedinference 命名空间

// 输出重载运算符 <<，用于打印 TensorName 对象信息
// 输出格式为：'C' (index 1 of ['N', 'C', 'H', 'W'])
std::ostream& operator<<(std::ostream& out, const TensorName& tensorname) {
  // 输出 TensorName 对象的名称和索引信息
  out << tensorname.name_ << " (index ";
  out << tensorname.origin_idx_ << " of ";
  out << tensorname.origin_ << ")";
  return out;
}

// 将 TensorNames 对象转换为 Dimname 类型的向量
std::vector<Dimname> TensorNames::toDimnameVec() const {
  // 创建结果向量，预留足够空间以容纳所有的 Dimname
  std::vector<Dimname> result;
  result.reserve(names_.size());
  // 遍历所有的 TensorName 对象，将其转换为 Dimname 对象并添加到结果向量中
  for (const auto& tensor_name : names_) {
    result.emplace_back(tensor_name.toDimname());
  }
  // 返回包含所有 Dimname 的结果向量
  return result;
}

} // namespace at::namedinference
```