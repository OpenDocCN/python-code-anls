# `.\pytorch\torch\csrc\lazy\core\shape.h`

```
#pragma once
// 预处理指令：确保头文件只被编译一次

#include <ostream>
// 包含标准输出流的头文件

#include <vector>
// 包含向量（动态数组）的头文件

#include <c10/core/Scalar.h>
// 包含 C10 库中的 Scalar 类的头文件

#include <torch/csrc/jit/passes/symbolic_shape_analysis.h>
// 包含 Torch 库中的符号形状分析的头文件

#include <torch/csrc/lazy/core/hash.h>
// 包含 Torch 懒加载模块中的哈希计算的头文件

C10_DECLARE_FLAGS(ltc_enable_symbolic_shapes);
// 定义 C10 的标志位 ltc_enable_symbolic_shapes

namespace torch {
namespace lazy {

class TORCH_API Shape {
 public:
  Shape() = default;
  // 默认构造函数

  Shape(
      at::ScalarType scalar_type,
      c10::ArrayRef<int64_t> sizes,
      std::optional<std::vector<bool>> is_symbolic = c10::nullopt);
  // 构造函数：接受标量类型、大小数组和可选的符号化布尔向量

  std::string to_string() const;
  // 返回形状对象的字符串表示

  c10::ScalarType scalar_type() const {
    return scalar_type_;
  }
  // 返回形状对象的标量类型

  void set_scalar_type(at::ScalarType value) {
    scalar_type_ = value;
  }
  // 设置形状对象的标量类型

  int64_t dim() const {
    return sizes_.size();
  }
  // 返回形状对象的维度数量

  c10::ArrayRef<int64_t> sizes() const {
    return sizes_;
  }
  // 返回形状对象的大小数组的引用

  int64_t size(int64_t dim) const {
    return sizes_.at(dim);
  }
  // 返回指定维度的大小

  void set_size(int64_t dim, int64_t size) {
    sizes_.at(dim) = size;
  }
  // 设置指定维度的大小

  const std::optional<std::vector<bool>>& is_symbolic() const {
    return is_symbolic_;
  }
  // 返回形状对象是否包含符号维度的可选布尔向量

  // Makes a copy with symbolic dims applied
  Shape with_symbolic_dims(
      std::optional<std::vector<bool>> symbolic_dims) const;
  // 创建包含符号维度的副本形状对象

  size_t numel() const;
  // 返回形状对象的元素数量

  hash_t hash(bool bakeInSizes) const;
  // 返回形状对象的哈希值，是否包含大小信息作为参数

  bool operator==(const Shape& other) const;
  // 比较运算符重载：判断形状对象是否相等

 private:
  c10::ScalarType scalar_type_{c10::ScalarType::Undefined};
  // 私有成员变量：标量类型，默认未定义

  // Sizes are the upper bound sizes for a tensor, used by XLA.
  std::vector<int64_t> sizes_;
  // 私有成员变量：大小数组，张量的上限大小，由 XLA 使用

  // Stores which dimmensions are symbolic
  // If nullopt, either it hasn't been initialized or the symbolic
  // dimmensions are not calculatable
  std::optional<std::vector<bool>> is_symbolic_ = c10::nullopt;
  // 私有成员变量：存储符号维度的可选布尔向量，如果为 nullopt，表示未初始化或符号维度不可计算
};

TORCH_API std::ostream& operator<<(std::ostream& out, const Shape& shape);
// 声明输出流运算符重载：输出形状对象到流

TORCH_API bool symbolicShapeEnabled();
// 声明函数：检查符号形状是否启用

// Calculate and applies symbolic shapes onto the
// Shape objects passed to result_shapes
TORCH_API void applySymbolicShapesOnLT(
    const char* schema_str,
    std::vector<c10::IValue> args,
    std::vector<Shape>& result_shapes);
// 声明函数：计算并应用符号形状到传递给 result_shapes 的形状对象数组

} // namespace lazy
} // namespace torch
// 命名空间结束注释
```