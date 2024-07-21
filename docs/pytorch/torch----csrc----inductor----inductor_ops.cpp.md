# `.\pytorch\torch\csrc\inductor\inductor_ops.cpp`

```py
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/mm.h>
#endif

#include <torch/csrc/autograd/functions/accumulate_grad.h>
#include <torch/csrc/inductor/inductor_ops.h>
#include <torch/library.h>

#include <ATen/FunctionalTensorWrapper.h>

namespace torch {
namespace inductor {
using namespace at;

// 对两个矩阵进行矩阵乘法，结果保存在给定的输出张量中
Tensor _mm_plus_mm(
    const Tensor& a,
    const Tensor& b,
    const Tensor& c,
    const Tensor& d,
    Tensor& out) {
  // 使用mm函数计算矩阵a和b的乘积，并将结果存储到out中
  at::mm_out(out, a, b);
  // 对输出张量out执行addmm_操作，将矩阵c和d的乘积添加到out中（原地操作）
  out.addmm_(c, d);
  // 返回计算后的输出张量out
  return out;
}

// 从内存池中分配张量，设置偏移字节、数据类型、大小和步长
Tensor _alloc_from_pool(
    const Tensor& self,
    int64_t offset_bytes,
    ScalarType dtype,
    IntArrayRef size,
    IntArrayRef stride) {
  // 检查张量self的存储偏移是否为0
  TORCH_CHECK(self.storage_offset() == 0);
  // 创建一个新的张量self_，其存储使用self的存储，数据类型为dtype
  Tensor self_ = at::detail::make_tensor<TensorImpl>(
      Storage(self.storage()),
      self.key_set(),
      caffe2::TypeMeta::fromScalarType(dtype));
  auto* self_tmp_ = self_.unsafeGetTensorImpl();
  // 设置张量的存储偏移，单位为dtype的元素大小
  self_tmp_->set_storage_offset(offset_bytes / c10::elementSize(dtype));
  // 设置张量的大小和步长
  self_tmp_->set_sizes_and_strides(size, stride);
  // 返回分配后的张量self_
  return self_;
}

// 重新解释张量的视图，增加偏移量并设置新的大小和步长
Tensor _reinterpret_tensor(
    const Tensor& self,
    IntArrayRef size,
    IntArrayRef stride,
    int64_t offset_increment) {
  // 创建一个新的张量self_，其存储使用self的存储，数据类型为self的数据类型
  Tensor self_ = at::detail::make_tensor<TensorImpl>(
      Storage(self.storage()), self.key_set(), self.dtype());
  auto* self_tmp_ = self_.unsafeGetTensorImpl();
  // 增加张量的存储偏移量
  self_tmp_->set_storage_offset(self.storage_offset() + offset_increment);
  // 设置张量的大小和步长
  self_tmp_->set_sizes_and_strides(size, stride);
  // 返回重新解释后的张量self_
  return self_;
}

// 累积梯度更新函数，处理来自Python的梯度更新
static void accumulate_grad_(const Tensor& variable, const Tensor& new_grad) {
  // 获取变量的可变梯度引用
  at::Tensor& grad = variable.mutable_grad();
  // 如果新梯度的设备不是kMeta
  if (new_grad.device() != kMeta) {
    // 从Python调用时，直接调用accumulateGrad函数累积梯度更新
    torch::autograd::AccumulateGrad::accumulateGrad(
        variable,
        grad,
        new_grad,
        2 /* num_expected_refs */,
        [&grad](at::Tensor&& grad_update) { grad = std::move(grad_update); });
  } else {
    // 针对`device="meta"`的情况，无需进行形状检查，直接复制新梯度给grad
    if (!grad.defined()) {
      grad = new_grad;
    }
  }
}

} // namespace inductor
} // namespace torch
TORCH_LIBRARY_FRAGMENT(inductor, m) {
  // 定义名为 "_mm_plus_mm" 的函数，接受五个张量参数并返回一个张量
  m.def(
      "_mm_plus_mm(Tensor a, Tensor b, Tensor c, Tensor d, Tensor(t!) out) -> Tensor(t!)",
      // 使用指定的分发键创建分发规则，与函数 _mm_plus_mm 相关联
      dispatch(c10::DispatchKey::CompositeExplicitAutograd, _mm_plus_mm),
      // 使用 pt2_compliant_tag 标签，指示兼容 PyTorch 2.0 行为
      {at::Tag::pt2_compliant_tag});

  // 定义名为 "_alloc_from_pool" 的函数，从内存池分配张量
  m.def(
      "_alloc_from_pool(Tensor self, int offset_bytes, ScalarType dtype, int[] size, int[] stride) -> Tensor",
      // 将 _alloc_from_pool 与指定的分发键相关联
      _alloc_from_pool,
      // 使用 pt2_compliant_tag 标签，指示兼容 PyTorch 2.0 行为
      {at::Tag::pt2_compliant_tag});

  // 定义名为 "_reinterpret_tensor" 的函数，重新解释张量的大小和步长
  m.def(
      "_reinterpret_tensor(Tensor self, int[] size, int[] stride, int offset_increment=0) -> Tensor",
      // 使用指定的分发键创建分发规则，与函数 _reinterpret_tensor 相关联
      dispatch(
          c10::DispatchKey::CompositeExplicitAutograd, _reinterpret_tensor),
      // 使用 pt2_compliant_tag 标签，指示兼容 PyTorch 2.0 行为
      {at::Tag::pt2_compliant_tag});

  // 定义名为 "accumulate_grad_" 的函数，累积梯度到变量的梯度上
  m.def(
      "accumulate_grad_(Tensor variable, Tensor new_grad) -> ()",
      // 使用指定的分发键创建分发规则，与函数 accumulate_grad_ 相关联
      dispatch(c10::DispatchKey::CompositeExplicitAutograd, accumulate_grad_),
      // 使用 pt2_compliant_tag 标签，指示兼容 PyTorch 2.0 行为
      {at::Tag::pt2_compliant_tag});
}

} // namespace inductor
} // namespace torch
```