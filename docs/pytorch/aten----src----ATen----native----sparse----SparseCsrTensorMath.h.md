# `.\pytorch\aten\src\ATen\native\sparse\SparseCsrTensorMath.h`

```py
// 声明代码在此之后只能被包含一次，防止头文件被多次包含而引起的重定义问题
#pragma once

// 包含 ATen 库的头文件
#include <ATen/Tensor.h>
#include <ATen/core/Scalar.h>
#include <ATen/TensorUtils.h>
#include <ATen/native/ReductionType.h>
#include <ATen/native/cpu/SpmmReduceKernel.h>

// ATen 命名空间中嵌套的命名空间 at::native::sparse::impl
namespace at::native::sparse::impl {

// 返回 true 如果 self 张量的所有元素都是零
// TODO: 这个函数有潜力成为一个通用的辅助函数
inline bool _is_sparse_and_zero(const Tensor& self) {
    // 检查 self 张量是否是稀疏格式之一，并且非零元素个数为零
    if (self.layout() == kSparse || self.layout() == kSparseCsr ||
        self.layout() == kSparseCsc || self.layout() == kSparseBsr ||
        self.layout() == kSparseBsc) {
        if (self._nnz() == 0) {
            return true;
        }
    }
    return false;
}

// 检查 self 张量是否在 CPU 上，并给出相应的错误消息
inline void _check_is_cpu(const Tensor& self, c10::string_view name) {
    TORCH_CHECK(
        self.is_cpu(),
        "Expected all tensors to be on the same device. addmm expected '",
        name,
        "' to be CPU tensor, but got ",
        self.device(),
        " tensor");
}

// 检查 self 张量是否在 CUDA 上，并给出相应的错误消息
inline void _check_is_cuda(const Tensor& self, c10::string_view name) {
    TORCH_CHECK(
        self.is_cuda(),
        "Expected all tensors to be on the same device. addmm expected '",
        name,
        "' to be CUDA tensor, but got ",
        self.device(),
        " tensor");
}

// 检查 self 张量的维度是否符合预期，并给出相应的错误消息
inline void _check_dim(const Tensor& self, int64_t target_dim, c10::string_view name) {
    if (target_dim == 2) {
        TORCH_CHECK(
            self.dim() == target_dim,
            name, " must be a matrix, ",
            "got ", self.dim(), "-D tensor");
    }
    TORCH_CHECK(
        self.dim() == target_dim,
        "Expected ",
        name,
        " to be of dimension ",
        target_dim,
        " but got ",
        self.dim(),
        " instead.");
}

// 检查稀疏矩阵乘法 reduce 函数的输入张量是否符合预期
template <bool train>
inline void check_sparse_mm_reduce_impl_inputs(
    const Tensor& self,
    const Tensor& grad_out,
    const Tensor& other) {
  // 内部断言：self 张量必须是稀疏 CSR 格式
  TORCH_INTERNAL_ASSERT(self.is_sparse_csr());

  // 获取输入的标量类型
  const auto input_scalar_type = self.values().scalar_type();
  // 检查布局和标量类型，给出相应的错误消息
  CheckedFrom c = train ? "sparse_mm_reduce_backward" : "sparse_mm_reduce";
  if (train) {
    checkLayout(c, grad_out, kStrided);
    checkScalarType(c, {grad_out, "grad_out", 1}, input_scalar_type);
    check_dim_size(grad_out, 2, 0, self.size(0));
    check_dim_size(grad_out, 2, 1, other.size(1));
  }

  // 确定 other 张量的位置（如果在训练过程中为 2，否则为 1），并检查其布局、标量类型和尺寸
  int pos = train ? 2 : 1;
  checkLayout(c, other, kStrided);
  checkScalarType(c, {other, "other", pos}, input_scalar_type);
  check_dim_size(other, 2, 0, self.size(1));
}

} // at::native::sparse::impl


这段代码是一个 C++ 的头文件，定义了一些用于稀疏张量计算的实用函数和模板函数。
```