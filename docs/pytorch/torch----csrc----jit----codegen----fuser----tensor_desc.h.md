# `.\pytorch\torch\csrc\jit\codegen\fuser\tensor_desc.h`

```py
#pragma once
// 预处理指令：确保本头文件只被编译一次

#include <ATen/ATen.h>
// 包含 ATen 库的头文件

#include <ATen/core/jit_type.h>
// 包含 ATen 库的 jit_type.h 头文件

#include <c10/util/Exception.h>
// 包含 c10 库的 Exception 头文件

#include <c10/util/hash.h>
// 包含 c10 库的 hash 头文件

#include <torch/csrc/Export.h>
// 包含 torch 库的 Export 头文件

#include <algorithm>
// 包含算法库，用于 STL 算法

#include <ostream>
// 包含输出流库，用于输出流操作

#include <vector>
// 包含向量库，用于定义和操作向量

namespace torch {
namespace jit {
namespace fuser {

// type information needed by the compiler for input/outputs
// 编译器需要的输入/输出的类型信息
// contiguity[i] is true if the dim i is contiguous with dim i + 1.
// contiguity[i] 为 true 表示维度 i 与维度 i + 1 是连续的。
// contiguity.back() == true means strides.back() == 1.
// contiguity.back() 为 true 表示 strides.back() == 1。

struct TORCH_API TensorDesc {
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  at::ScalarType scalar_type;
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  std::vector<bool> contiguity;

  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  // 构造函数：根据标量类型和连续性向量初始化张量描述对象
  TensorDesc(const at::ScalarType& type, const std::vector<bool>& contiguity)
      : scalar_type{type}, contiguity{contiguity} {
    if (contiguity.empty()) {
      nDim_ = 0;
    } else {
      nDim_ = std::count(contiguity.begin(), contiguity.end(), false) +
          (lastIsContiguous() ? 1 : 0);
    }
  }

  // Delegating constructors
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  // 委托构造函数：根据标量类型、大小和步长初始化张量描述对象
  TensorDesc(
      const at::ScalarType& type,
      const at::IntArrayRef& sizes,
      const at::IntArrayRef& strides)
      : TensorDesc(type, TensorDesc::findContiguous(sizes, strides)) {}

  // 构造函数：根据 Tensor 对象初始化张量描述对象
  TensorDesc(const at::Tensor& t)
      : TensorDesc(t.scalar_type(), t.sizes(), t.strides()) {}

  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  // 构造函数：根据 TensorTypePtr 对象初始化张量描述对象
  TensorDesc(const c10::TensorTypePtr& type)
      : TensorDesc(
            type->scalarType().value(),
            type->sizes().concrete_sizes().value(),
            type->strides().concrete_sizes().value()) {}

  // 返回压缩连续性后的维度数目
  size_t nDim() const {
    return nDim_;
  }

  // 返回内部维度是否是连续的
  bool lastIsContiguous() const {
    return (contiguity.empty() || contiguity.back());
  }

  // 静态方法：根据大小和步长找到连续性向量
  static std::vector<bool> findContiguous(
      const at::IntArrayRef& sizes,
      const at::IntArrayRef& strides) {
    AT_ASSERT(sizes.size() == strides.size());
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    std::vector<bool> cont(sizes.size());
    for (size_t i = 0; i < sizes.size(); ++i) {
      const auto expected_stride =
          (i + 1 < sizes.size()) ? sizes[i + 1] * strides[i + 1] : 1;
      cont[i] = (strides[i] == expected_stride);
    }
    return cont;
  }

  // 比较运算符重载：判断两个张量描述对象是否相等
  bool operator==(const TensorDesc& desc) const {
    return scalar_type == desc.scalar_type && contiguity == desc.contiguity;
  }

  // 比较运算符重载：判断两个张量描述对象是否不相等
  bool operator!=(const TensorDesc& desc) const {
    return !(*this == desc);
  }

  // 静态方法：计算张量描述对象的哈希值
  static size_t hash(const TensorDesc& spec) {
    return c10::get_hash(
        spec.scalar_type,
        spec.nDim_,
        std::hash<std::vector<bool>>{}(spec.contiguity));
  }

 private:
  size_t nDim_;
};

} // namespace fuser
} // namespace jit
} // namespace torch
// 定义了一个重载的流输出操作符 <<，用于将 TensorDesc 对象的描述输出到流中
inline std::ostream& operator<<(std::ostream& out, const TensorDesc& d) {
  // 输出 TensorDesc 对象的标量类型到流中
  out << d.scalar_type << "[";
  // 遍历 TensorDesc 对象的连续性属性，并输出到流中
  for (const auto b : d.contiguity)
    out << b << ";";
  // 输出连续性属性的结束标记到流中
  out << "]";
  // 返回输出流
  return out;
}

// 命名空间 fuser 结束
} // namespace fuser
// 命名空间 jit 结束
} // namespace jit
// 命名空间 torch 结束
} // namespace torch
```