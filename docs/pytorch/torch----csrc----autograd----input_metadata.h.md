# `.\pytorch\torch\csrc\autograd\input_metadata.h`

```
#pragma once

// 包含 ATen 库中的各种头文件，用于张量操作和设备管理
#include <ATen/ExpandUtils.h>
#include <ATen/NestedTensorImpl.h>
#include <ATen/core/Tensor.h>
#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>
#include <c10/core/Stream.h>
#include <c10/core/SymIntArrayRef.h>
#include <c10/core/TensorImpl.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/util/DimVector.h>
#include <c10/util/Exception.h>
#include <c10/util/SmallVector.h>

// 根据是否定义了 AT_PER_OPERATOR_HEADERS 宏来选择不同的 ATen 头文件
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/zeros.h>
#endif

// 定义了 torch::autograd 命名空间，包含了相关的数据结构和函数
namespace torch::autograd {

// 使用 c10::SmallVector 来定义 SymInt 类型的小向量
using SymIntSmallVec = c10::SmallVector<c10::SymInt, c10::kDimVectorStaticSize>;

// MetadataShape 是一个 variant 类型，可以是 SymIntSmallVec 或者 at::Tensor
using MetadataShape = std::variant<SymIntSmallVec, at::Tensor>;

/**
 * InputMetadata 记录了 TensorOptions、张量的形状、是否设置了 Python dispatch key
 * （即张量子类），以及操作发生的流（stream）。
 *
 * 如果 is_valid() 返回 false，则对应的输入数据没有使用，可能是未定义的张量。
 */
struct TORCH_API InputMetadata {
  // 默认构造函数
  InputMetadata() = default;

  // 构造函数，接受张量选项、输入形状、是否为张量子类、是否为嵌套张量作为参数
  InputMetadata(
      const at::TensorOptions& options,
      MetadataShape input_shape,
      bool is_tensor_subclass,
      bool is_nested);

  // 构造函数，接受一个张量作为参数
  InputMetadata(const at::Tensor& t);

  // 返回张量的选项
  const at::TensorOptions& options() const {
    return options_;
  }

  // 返回张量的数据类型
  caffe2::TypeMeta dtype() const {
    return options_.dtype();
  }

  // 返回张量的设备
  at::Device device() const {
    return options_.device();
  }

  // 返回张量的布局
  at::Layout layout() const {
    return options_.layout();
  }

  // 返回操作发生的流
  c10::Stream stream() const {
    return stream_;
  }

  // 返回是否为张量子类
  bool is_tensor_subclass() const {
    return is_tensor_subclass_;
  }

  // 返回一个与当前张量形状相同的零张量
  at::Tensor zeros_like() const;

  // 检查梯度张量是否与当前张量的形状相同
  bool is_same_shape(const at::Tensor& grad) const;

  // 检查梯度张量是否可以扩展到与当前张量相同的形状
  bool is_expandable_to_shape(const at::Tensor& grad) const;

  // 减少梯度张量的维度
  at::Tensor reduce_grad(at::Tensor& grad) const;

  // 根据索引可能减少梯度张量的维度，并返回处理错误的函数
  at::Tensor maybe_reduce(
      const size_t index,
      at::Tensor grad,
      const std::function<std::string(const std::string&)>& format_error) const;

  // 返回不兼容形状错误信息的字符串流
  std::stringstream incompatible_shape_error_message(
      const size_t index,
      const at::Tensor& grad) const;

  // 返回是否使用默认构造
  bool was_default_constructed() const {
    return was_default_constructed_;
  }

  // 返回是否为 C++ 嵌套张量
  bool is_cpp_nested_tensor() const;

  // 返回是否为嵌套张量
  bool is_nested_tensor() const {
    return is_nested_;
  }

  // 返回形状作为 SymIntArrayRef
  c10::SymIntArrayRef shape_as_dim_vector() const;

  // 可能会修改形状的 SymIntSmallVec，不是线程安全的，调用者必须保护使用锁
  SymIntSmallVec& mutable_shape_as_dim_vector();

 private:
  // 返回形状作为张量
  at::Tensor shape_as_tensor() const;

  // 检查嵌套性是否与梯度张量相同
  bool is_nestedness_same(const at::Tensor& grad) const;

  // 检查是否可能将梯度张量扩展到相同的形状
  bool maybe_expandable_to(const at::Tensor& grad) const;

  // 不可变的张量选项
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  const at::TensorOptions options_;

  // 形状的元数据
  MetadataShape shape_;

  // 操作的流，默认为张量的设备流
  c10::Stream stream_ = c10::Stream(c10::Stream::Default::DEFAULT, device());

  // 是否为张量子类的标志
  bool is_tensor_subclass_ = false;

  // 是否为嵌套张量的标志
  bool is_nested_ = false;

  // 是否使用默认构造的标志
  bool was_default_constructed_ = true;
};

} // namespace torch::autograd
```