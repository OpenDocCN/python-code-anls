# `.\pytorch\torch\csrc\lazy\core\tensor_impl.h`

```
#pragma once
// 预处理指令，确保本文件只被编译一次

#include <ATen/Tensor.h>
// 引入 ATen 库中的 Tensor 类

#include <c10/core/SymIntArrayRef.h>
#include <c10/core/TensorImpl.h>
// 引入 C10 库中的 SymIntArrayRef 类和 TensorImpl 类

#include <torch/csrc/lazy/core/tensor.h>
// 引入 Torch 惰性计算模块中的 tensor.h 文件

namespace torch {
namespace lazy {

// Torch 惰性计算模块的命名空间

// LTCTensorImpl 类，继承自 c10::TensorImpl，用于处理 LazyTensor
class TORCH_API LTCTensorImpl final : public c10::TensorImpl {
 public:
  // 显式构造函数，接受 LazyTensorPtr 类型参数
  explicit LTCTensorImpl(const LazyTensorPtr& tensor);
  // 显式构造函数，接受 LazyTensor 类型参数
  explicit LTCTensorImpl(const LazyTensor& tensor);
  // 显式构造函数，接受 LazyTensor 类型右值参数
  explicit LTCTensorImpl(LazyTensor&& tensor);

  // 获取内部 LazyTensorPtr 的访问方法
  LazyTensorPtr tensor() {
    return tensor_;
  }

  // 设置内部 LazyTensorPtr 的方法
  void set_tensor(const LazyTensorPtr& lazy_tensor);

  // 强制刷新大小的方法，将代数（generation）置为 0
  void force_refresh_sizes() {
    generation_ = 0;
  }

  // 浅拷贝和分离方法，传入版本计数器和是否允许改变张量元数据标志
  c10::intrusive_ptr<TensorImpl> shallow_copy_and_detach(
      const c10::VariableVersion& version_counter,
      bool allow_tensor_metadata_change) const override;

  // 浅拷贝和分离方法，传入版本计数器右值和是否允许改变张量元数据标志
  c10::intrusive_ptr<TensorImpl> shallow_copy_and_detach(
      c10::VariableVersion&& version_counter,
      bool allow_tensor_metadata_change) const override;

  // 从另一个 TensorImpl 类型对象浅拷贝方法
  void shallow_copy_from(const c10::intrusive_ptr<TensorImpl>& impl) override;

  // 自定义方法，返回张量大小的整数数组引用
  at::IntArrayRef sizes_custom() const override;

  // 自定义方法，返回张量步幅的整数数组引用
  at::IntArrayRef strides_custom() const override;

  // 自定义方法，返回张量元素总数
  int64_t numel_custom() const override;

  // 自定义方法，返回张量存储偏移量
  int64_t storage_offset_custom() const override;

  // 自定义方法，返回张量维度数
  int64_t dim_custom() const override;

  // 自定义方法，检查张量是否在指定内存格式下连续
  bool is_contiguous_custom(at::MemoryFormat memory_format) const override;

  // 自定义方法，检查张量是否具有指定内存格式下的步幅
  bool is_strides_like_custom(at::MemoryFormat memory_format) const override;

  // 自定义方法，检查张量是否非重叠且密集
  bool is_non_overlapping_and_dense_custom() const override;

  // 自定义方法，返回符号化大小的 SymIntArrayRef 引用
  c10::SymIntArrayRef sym_sizes_custom() const override;

  // 自定义方法，返回符号化步幅的 SymIntArrayRef 引用
  c10::SymIntArrayRef sym_strides_custom() const override;

  // 自定义方法，返回符号化元素总数的 SymInt
  c10::SymInt sym_numel_custom() const override;

 private:
  // 设置大小属性的私有方法
  void setup_size_properties();

  // 内部的 LazyTensorPtr 对象
  LazyTensorPtr tensor_;

  // 可选的符号化大小向量的可选值
  mutable std::optional<std::vector<c10::SymInt>> sym_sizes_;

  // 代数（generation）计数器
  size_t generation_{0};
};

} // namespace lazy
} // namespace torch
```