# `.\pytorch\aten\src\ATen\native\sparse\cuda\SparseBlas.cpp`

```
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/ExpandUtils.h>
#include <ATen/SparseCsrTensorUtils.h>
#include <ATen/native/Resize.h>
#include <ATen/native/sparse/cuda/SparseBlasImpl.h>
#include <ATen/native/sparse/SparseBlas.h>
#include <ATen/native/sparse/SparseCsrTensorMath.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/addmm_native.h>
#include <ATen/ops/addmv_native.h>
#include <ATen/ops/copy_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/mul.h>
#include <ATen/ops/resize_as_sparse_native.h>
#include <ATen/ops/scalar_tensor_native.h>
#include <ATen/ops/sparse_sampled_addmm_native.h>
#include <ATen/ops/triangular_solve_native.h>
#endif

#include <c10/util/MaybeOwned.h>

namespace at::native {

/*
  Computes `result` <- α*(A @ B) * spy(C) + β*C, where spy(C) is the sparsity pattern matrix of C.

  Args:
  * `mat1` - [in] dense Tensor A of size m × k.
  * `mat2` - [in] dense Tensor B of size k × n.
  * `self` - [in] sparse Tensor C of size m × n.
  * `result` - [out] sparse Tensor of size m × n.
*/
// 在 CUDA 环境下实现稀疏矩阵乘法加法运算
Tensor& sparse_sampled_addmm_out_sparse_csr_cuda(
    const Tensor& self,
    const Tensor& mat1,
    const Tensor& mat2,
    const Scalar& beta,
    const Scalar& alpha,
    Tensor& result) {
  // 检查输入的稀疏矩阵乘法加法运算的合法性
  at::native::sparse::sparse_sampled_addmm_check_inputs(
      self, mat1, mat2, beta, alpha, result);

  // 如果 result 和 self 不是同一个 Tensor，进行大小调整并拷贝 self 的值
  if (&result != &self) {
    // 如果 mat1 和 mat2 是批处理的，允许 self 是单个矩阵
    auto result_sizes = DimVector(mat1.sizes().slice(0, mat1.dim() - 2));
    result_sizes.push_back(self.size(-2));
    result_sizes.push_back(self.size(-1));
    at::sparse_csr::get_sparse_csr_impl(result)->resize_(self._nnz(), result_sizes);
    result.copy_(self);
  }

  // 当 mat1 或 mat2 为空矩阵时，避免调用 cuSPARSE 导致的段错误
  if (mat1.numel() == 0 || mat2.numel() == 0) {
    // 对 result 执行乘法运算
    result.mul_(beta);
    return result;
  }

  // 调用 CUDA 实现的稀疏矩阵乘法加法运算
  sparse::impl::cuda::sampled_addmm_out_sparse_csr(mat1, mat2, beta, alpha, result);
  return result;
}

// 返回稀疏矩阵乘法加法运算的结果，不修改输入的 result Tensor
Tensor sparse_sampled_addmm_sparse_csr_cuda(
    const Tensor& self,
    const Tensor& mat1,
    const Tensor& mat2,
    const Scalar& beta,
    const Scalar& alpha) {
  // 创建一个空的 Tensor 作为结果
  auto result = at::empty({0, 0}, self.options());
  // 调用实现稀疏矩阵乘法加法运算的函数
  at::native::sparse_sampled_addmm_out_sparse_csr_cuda(self, mat1, mat2, beta, alpha, result);
  // 返回结果 Tensor
  return result;
}

// 实现稀疏压缩格式的加法乘法运算：result = beta * self + alpha * (mat1 @ mat2)
Tensor& addmm_out_sparse_compressed_cuda(
    const Tensor& self,
    const Tensor& mat1,
    const Tensor& mat2,
    const Scalar& beta,
    const Scalar& alpha,
    Tensor& result) {
    // 检查输入张量是否在 CUDA 设备上，并在不满足条件时抛出错误
    sparse::impl::_check_is_cuda(self, "self");
    sparse::impl::_check_is_cuda(mat1, "mat1");
    sparse::impl::_check_is_cuda(mat2, "mat2");
    sparse::impl::_check_is_cuda(result, "result");
    
    // 检查 mat1 和 mat2 的维度是否为2，不是则抛出错误
    sparse::impl::_check_dim(mat1, 2, "mat1");
    sparse::impl::_check_dim(mat2, 2, "mat2");
    
    // 检查 mat1 和 mat2 的维度是否允许进行矩阵乘法操作，不允许则抛出错误
    TORCH_CHECK(
        mat1.size(1) == mat2.size(0), "mat1 and mat2 shapes cannot be multiplied (",
        mat1.size(0), "x", mat1.size(1), " and ", mat2.sizes()[0], "x", mat2.sizes()[1], ")");
    
    // 初始化一个 MaybeOwned<Tensor> 对象 self_，用于存储处理后的 self 张量
    // 如果 result 是 self 的引用（即 in-place 操作），则不需要扩展 self
    if (&result == &self) {
       self_ = c10::MaybeOwned<Tensor>::borrowed(self);
    } else {
       // 如果不是 in-place 操作，根据 mat1 和 mat2 的尺寸扩展 self 张量的尺寸
       self_ = expand_size(self, {mat1.size(0), mat2.size(1)}, "addmm");
    }
    
    // 检查扩展后的 self_ 张量是否是二维的，且尺寸与 mat1 和 mat2 匹配，不匹配则抛出错误
    sparse::impl::_check_dim(*self_, 2, "self");
    TORCH_CHECK(((self_->dim() == 2) &&
                 (self_->size(0) == mat1.size(0)) &&
                 (self_->size(1) == mat2.size(1))),
                "The input tensor must be a matrix with size ",
                mat1.size(0),
                "x",
                mat2.size(1),
                ", but got a ",
                self_->dim(),
                "-D tensor with size ",
                self_->size(0),
                "x",
                self_->size(1));
    
    // 如果 result 不是 self，根据 result 的布局调整其尺寸以匹配 self_ 的尺寸
    if (!result.is_same(self)) {
      if (result.layout() == kStrided) {
        at::native::resize_output(result, self_->sizes());
      } else {
        result.resize_as_sparse_(*self_);
      }
    }
    
    // 如果 result 的元素数量为0，直接返回 result
    if (result.numel() == 0) {
      return result;
    }
    
    // 如果 mat1 或 mat2 是稀疏张量且全为零，根据 beta 的值对 result 进行处理
    if (sparse::impl::_is_sparse_and_zero(mat1) || sparse::impl::_is_sparse_and_zero(mat2)) {
      // 根据文档，当 beta==0 时忽略 self 中的值，且不传播 NaN 和 Inf
      const auto beta_val = beta.toComplexDouble();
      if (beta_val == 0.) {
        result.zero_();
      } else {
        // 如果 result 不是 self，复制 self 的值到 result
        if (!result.is_same(self)) {
          result.copy_(*self_);
        }
        // 如果 beta_val 不等于 1，则将 result 的值乘以 beta_val
        if (beta_val != 1.) {
          result.mul_(beta);
        }
      }
      return result;
    }
    
    // 使用 CUDA 实现的稀疏矩阵乘法操作，将结果存储到 result 中
    sparse::impl::cuda::addmm_out_sparse_csr(*self_, mat1, mat2, beta, alpha, result);
    return result;
}

Tensor& baddbmm_out_sparse_csr_cuda(
    const Tensor& self,
    const Tensor& mat1,
    const Tensor& mat2,
    const Scalar& beta,
    const Scalar& alpha,
    Tensor& result) {
  // 断言 mat1 是稀疏 CSR 格式的张量
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(mat1.is_sparse_csr());

  // 检查 self 的布局是否为 kStrided，若不是则抛出错误信息
  TORCH_CHECK(self.layout() == kStrided, "torch.baddbmm: Expected self to be strided, but got layout ", self.layout());
  // 检查 mat2 的布局是否为 kStrided，若不是则抛出错误信息
  TORCH_CHECK(mat2.layout() == kStrided, "torch.baddbmm: Expect mat2 to be strided, but got ", mat2.layout());
  // 检查 result 的布局是否为 kStrided，若不是则抛出错误信息
  TORCH_CHECK(result.layout() == kStrided, "torch.baddbmm: Expect result to be strided, but got ", result.layout());

  // 若 result 不是 self 引用，调整 result 的大小以匹配 self
  if (!result.is_same(self)) {
    at::native::resize_output(result, self.sizes());
  }

  // 如果 mat1 的非零元素数量为 0
  if (mat1._nnz() == 0) {
    // 根据文档说明，当 beta == 0 时，应忽略 self 中的值，不应传播 NaN 和 Inf
    if (beta.toComplexDouble() == 0.) {
      // 将 result 的值置为零
      result.zero_();
    } else {
      // 如果 result 不与 self 相同，则将 self 的值复制到 result
      if (!result.is_same(self)) {
        result.copy_(self);
      }
      // 如果 beta 不等于 1，则将 result 中的值乘以 beta
      if (beta.toComplexDouble() != 1.) {
        result.mul_(beta);
      }
    }
    // 返回 result
    return result;
  }

  // 调用 sparse::impl::cuda::addmm_out_sparse_csr 函数执行稀疏矩阵乘法运算
  sparse::impl::cuda::addmm_out_sparse_csr(self, mat1, mat2, beta, alpha, result);
  // 返回 result
  return result;
}

Tensor& bmm_out_sparse_csr_cuda(
    const Tensor& mat1,
    const Tensor& mat2,
    Tensor& result) {
  // 设置 beta 和 alpha 的值
  Scalar beta(0.0);
  Scalar alpha(1.0);
  // 调用 baddbmm_out_sparse_csr_cuda 函数执行稀疏矩阵乘法运算
  return at::native::baddbmm_out_sparse_csr_cuda(result, mat1, mat2, beta, alpha, result);
}

Tensor& addmv_out_sparse_compressed_cuda(
    const Tensor& self,
    const Tensor& mat,
    const Tensor& vec,
    const Scalar& beta,
    const Scalar& alpha,
    Tensor& result) {

  // 如果 mat 的布局为 kSparseCsc
  if (mat.layout() == kSparseCsc) {
    // 将 mat 转换为稀疏 CSR 格式，然后调用当前函数以处理
    return addmv_out_sparse_compressed_cuda(self, mat.to_sparse_csr(), vec,
        beta, alpha, result);
  }
  // 如果 mat 的布局为 kSparseBsc，抛出错误信息，因为当前函数不支持 SparseBsc 布局的输入 mat
  TORCH_CHECK(mat.layout() != kSparseBsc, "addmm_out_sparse_csr_cuda currently does not support layout SparseBsc for input mat.");

  // 检查 mat 的维度是否为 2，若不是则抛出错误信息
  TORCH_CHECK(mat.dim() == 2, "addmv: Expected mat to be 2-D");
  // 检查 vec 的维度是否为 1，若不是则抛出错误信息
  TORCH_CHECK(vec.dim() == 1, "addmv: Expected vec to be 1-D");

  // 从 TORCH_IMPL_FUNC(addmv_out_cuda) 的 CUDA 实现中复制的预处理代码
  // 在 CUDA 和 SparseCsrCUDA 调度键以及结构化内核中使用相同的函数可能更好，
  // 但当使用相同函数时，存在未定义符号问题

  // 将 self 的大小扩展为与 mat 的第一维度大小相同
  c10::MaybeOwned<Tensor> self_ = expand_size(self, {mat.size(0)});
  auto betaval = beta.toComplexDouble();

  // 如果 result 不是 self 的引用
  if (&result != &self) {
    // 调整 result 的大小以匹配 self 的大小
    at::native::resize_output(result, self_->sizes());
    // 如果 betaval 不等于 0.0，则将 self_ 的值复制到 result
    if (betaval != 0.0) {
      at::native::copy_(result, *self_);
    }
  }

  // 如果 mat 的非零元素数量为 0
  if (mat._nnz() == 0) {
    // 当矩阵为空时的快捷方式
    // 根据定义，当 beta == 0 时，应忽略 self 中的值。NaN 和 Inf 不应传播
    if (betaval == 0.0) {
      // 将 result 的值置为零
      return result.zero_();
    }
    } else {
      # 如果不是稀疏矩阵-向量乘法，执行以下操作：
      # 调用 at::mul_out 函数，将 self 与 result 相乘，并将结果存储在 result 中
      # 使用 at::native::scalar_tensor 创建一个标量 Tensor，其数值为 beta，类型与 self 相同，
      # 在 CPU 上执行，不设置特定的布局和内存固定选项
      return at::mul_out(
          const_cast<Tensor&>(result),  # 将 result 转换为可变的 Tensor 引用
          self,                         # 第一个乘数 self
          at::native::scalar_tensor(    # 第二个乘数，一个标量 Tensor，值为 beta
              beta,
              self.scalar_type(),
              c10::nullopt /* layout */,
              at::kCPU,
              c10::nullopt /* pin_memory */));
    }
  }

  # 调用 CUDA 实现的稀疏矩阵-向量乘法函数，计算 mat 与 vec 的乘积，并将结果存储在 result 中
  sparse::impl::cuda::addmv_out_sparse_csr(mat, vec, beta, alpha, result);
  # 返回结果 Tensor result
  return result;
}



/*
  Solves a system of linear equations whose coefficients are represented in a sparse triangular matrix A:
  op(A) X = B.

  Args:
  * `B` - dense Tensor of size m × nrhs.
  * `A` - sparse Tensor of size m × m.
  * `upper` - controls whether upper or lower triangular part of A is considered in computations.
  * `transpose` - if true then op(A) = A^T.
  * `unitriangular` - if true then the diagonal elements of A are assumed to be one.
  * `X` - dense Tensor of size m × nrhs.
  * `clone_A` - cloned matrix A, required only for compatibility with strided layout interface.
*/
std::tuple<Tensor&, Tensor&> triangular_solve_out_sparse_csr_cuda(
    const Tensor& B,
    const Tensor& A,
    bool upper,
    bool transpose,
    bool unitriangular,
    Tensor& X,
    Tensor& clone_A) {
  // 调用 CUDA 实现的稀疏 CSR 格式的三角求解器
  sparse::impl::cuda::triangular_solve_out_sparse_csr(A, B, X, upper, transpose, unitriangular);
  // 返回解 X 和克隆的 A
  return std::tuple<Tensor&, Tensor&>(X, clone_A);
}

} // namespace at::native



// 结束 at::native 命名空间
```