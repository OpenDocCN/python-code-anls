# `.\pytorch\aten\src\ATen\TensorNames.h`

```py
#pragma once
// 预处理指令，确保头文件只被包含一次

#include <ATen/WrapDimUtils.h>
// 包含 ATen 库中的 WrapDimUtils.h 头文件

namespace at::namedinference {
// 进入 at::namedinference 命名空间

// TensorName and TensorNames are wrappers around Dimname and DimnameList
// that contain helper functions to make writing name inference rules easier.
// TensorName 和 TensorNames 是 Dimname 和 DimnameList 的封装，
// 提供了辅助函数以便于编写名称推断规则。

// A TensorName represents a Dimname associated with some DimnameList (from a
// Tensor). This encapsulates all the information that is needed to check if
// names *match* and to *unify* names.
// TensorName 表示与某个 DimnameList（来自 Tensor）相关联的 Dimname。
// 它封装了检查名称是否匹配和统一名称所需的所有信息。

// Definition: Two names in two tensors *match* if they are equal, or if at
// least one of them is a wildcard that can be *refined* to the other name.
// 定义：两个张量中的两个名称匹配，如果它们相等，或者如果它们中至少有一个是可以被细化为另一个名称的通配符。

// Definition: unify(name, other) fails if the names do not match. Otherwise,
// it returns the most refined of name and other.
// 定义：如果名称不匹配，则 unify(name, other) 失败。否则，它返回 name 和 other 中最细化的那个。

// Here is an example of checking if two names match.
// tensor: Tensor[A, None]
// other: Tensor[A]

// Let's say we wish to check if tensor.names[-1] matches other.names[-1].
// None (in tensor) cannot match A (in other) because if the None were refined
// to A, `tensor` would have duplicate names [A, A]. Therefore we need to check
// tensor.names [A, None] for the existence of A.
// 这里是检查两个名称是否匹配的示例。
// tensor: Tensor[A, None]
// other: Tensor[A]
// 假设我们希望检查 tensor.names[-1] 是否与 other.names[-1] 匹配。
// None（在 tensor 中）无法与 A（在 other 中）匹配，因为如果将 None 细化为 A，
// `tensor` 将具有重复的名称 [A, A]。因此，我们需要检查 tensor.names [A, None] 是否存在 A。

struct TORCH_API TensorName {
  explicit TensorName(ArrayRef<Dimname> origin, int origin_idx)
      : origin_(origin),
        name_(origin[maybe_wrap_dim(
            origin_idx,
            static_cast<int64_t>(origin.size()))]),
        origin_idx_(origin_idx) {}
  // 构造函数，用于创建 TensorName 对象

  // op_name is only used for error reporting.
  // op_name 仅用于错误报告。

  // unify 方法用于统一两个 TensorName 对象
  const TensorName& unify(const TensorName& other, const char* op_name) const;
  // 返回 TensorName 对象的 Dimname
  Dimname toDimname() const;

 private:
  ArrayRef<Dimname> origin_; // 原始 Dimname 数组的引用
  Dimname name_; // Dimname 对象
  int origin_idx_; // 原始索引，命名张量最多可以有 64 个维度

  TORCH_API friend std::ostream& operator<<(
      std::ostream& out,
      const TensorName& tensorname);
  // 友元声明，允许 ostream 使用 TensorName 对象
};

// 使用 SmallVector 封装的 TensorNameVec 类型
using TensorNameVec = SmallVector<TensorName, 10>;

// TensorNames 结构体，封装了一组 Dimname
struct TORCH_API TensorNames {
  explicit TensorNames(ArrayRef<Dimname> names);
  // 构造函数，从 Dimname 数组创建 TensorNames 对象

  // Create TensorNames from names[start:end]. Each individual TensorName stores
  // `names`, NOT names[start:end], because the original tensor's names are
  // `names`.
  // 从 names[start:end] 创建 TensorNames 对象。每个 TensorName 存储 `names`，
  // 而不是 names[start:end]，因为原始张量的名称是 `names`。

  explicit TensorNames(ArrayRef<Dimname> names, int64_t start, int64_t end);
  // 构造函数，从 names[start:end] 创建 TensorNames 对象

  // op_name is only used for error reporting.
  // op_name 仅用于错误报告。

  // 从右边就地统一两个 TensorNames 对象
  TensorNames& unifyFromRightInplace(
      const TensorNames& other,
      const char* op_name = "unify");
  // 检查 TensorNames 对象中的名称是否唯一
  void checkUnique(const char* op_name) const;

  // 添加一个 TensorName 到 TensorNames 对象
  void append(TensorName name);
  // 返回 Dimname 数组的 vector 表示
  std::vector<Dimname> toDimnameVec() const;

 private:
  explicit TensorNames(TensorNameVec&& names) : names_(std::move(names)){};
  // 私有构造函数，从 TensorNameVec 移动构造 TensorNames 对象

  TensorNameVec names_; // 使用 SmallVector 封装的 TensorNameVec
};

} // namespace at::namedinference
// 结束 at::namedinference 命名空间
```