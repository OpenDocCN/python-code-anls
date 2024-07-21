# `.\pytorch\aten\src\ATen\native\sparse\cuda\SparseBlasLegacy.cpp`

```
/*
Functions here use deprecated cuSPARSE API that was removed in CUDA 11.
This file will be removed eventually.
*/
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/native/SparseTensorUtils.h>
#include <ATen/native/sparse/cuda/SparseBlasLegacy.h>
#include <ATen/native/sparse/cuda/SparseCUDABlas.h>

namespace at::native {

// 定义了一个名为 s_addmm_out_csr_sparse_dense_cuda_worker 的函数，用于在 CUDA 上执行 CSR 稀疏矩阵与密集矩阵的乘法加法操作
void s_addmm_out_csr_sparse_dense_cuda_worker(int64_t nnz, int64_t m, int64_t n, int64_t k, const Tensor& r_, const Scalar& beta, const Tensor& t, const Scalar& alpha, const Tensor& crow_indices, const Tensor& col_indices, const Tensor& values, const Tensor& dense) {
  TORCH_INTERNAL_ASSERT(nnz > 0);

  // No half support, so we don't have to use CUDATypeConversion
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      values.scalar_type(), "addmm_sparse_cuda", [&] {
        scalar_t cast_beta = beta.to<scalar_t>();  // 将 beta 转换为与 values 相同的数据类型
        scalar_t cast_alpha = alpha.to<scalar_t>();  // 将 alpha 转换为与 values 相同的数据类型
        Tensor r__;  // 声明一个名为 r__ 的张量
        if (cast_beta == scalar_t(0)) {  // 如果 beta 等于零
          r_.zero_();  // 将 r_ 清零
        } else if (!at::sparse::is_same_tensor(t, r_)) {  // 如果 t 与 r_ 不是同一个张量
          r_.copy_(t);  // 将 t 的值复制到 r_
        }
        if (r_.stride(0) == 1 && r_.stride(1) == r_.size(0)) {  // 如果 r_ 是行优先存储
          r__ = r_;  // 直接使用 r_
        } else {
          // Note: This storage arrangement is preferred due to most of the CUDA kernels handle only contiguous tensors
          // 否则，创建一个行优先存储的副本
          r__ = r_.transpose(0, 1).clone(at::MemoryFormat::Contiguous);  // 对 r_ 进行转置，并克隆成连续存储格式的张量
          r__.transpose_(0, 1);  // 将副本再次转置为原来的形状
        }
        TORCH_INTERNAL_ASSERT(r__.mT().is_contiguous());  // 断言 r__.mT() 是连续存储的

        Tensor dense_;  // 声明一个名为 dense_ 的张量
        char transpose_dense;  // 声明一个用于表示 dense 是否转置的字符
        if (dense.stride(0) == 1 && dense.stride(1) == dense.size(0)) {  // 如果 dense 是行优先存储
          transpose_dense = 'n';  // 不进行转置
          dense_ = dense;  // 直接使用 dense_
        } else if (dense.stride(1) == 1 && dense.stride(0) == dense.size(1)) {  // 如果 dense 是列优先存储
          transpose_dense = 't';  // 进行转置
          dense_ = dense;  // 直接使用 dense_
        } else {
          transpose_dense = 't';  // 需要转置
          dense_ = dense.contiguous();  // 将 dense 转换为连续存储格式的张量
        }

        // 调用 sparse::cuda::csrmm2 函数执行 CSR 稀疏矩阵乘法加法操作
        sparse::cuda::csrmm2(
          'n',  // 不转置稀疏矩阵
          transpose_dense,  // 根据 dense 是否转置决定如何处理
          m,  // 稀疏矩阵的行数
          n,  // 稀疏矩阵的列数
          k,  // 稀疏矩阵的列数或者 dense 矩阵的行数
          nnz,  // 非零元素个数
          cast_alpha,  // alpha 参数
          values.data_ptr<scalar_t>(),  // 非零元素的数据指针
          crow_indices.data_ptr<int32_t>(),  // 行索引的数据指针
          col_indices.data_ptr<int32_t>(),  // 列索引的数据指针
          dense_.data_ptr<scalar_t>(),  // dense 矩阵的数据指针
          (transpose_dense == 'n' ? dense_.stride(1) : dense_.stride(0)),  // dense 矩阵的 stride
          cast_beta,  // beta 参数
          r__.data_ptr<scalar_t>(),  // 结果矩阵 r__ 的数据指针
          r__.stride(1));  // 结果矩阵 r__ 的 stride

        if (!at::sparse::is_same_tensor(r__, r_)) {  // 如果 r__ 与 r_ 不是同一个张量
          r_.copy_(r__);  // 将 r__ 的值复制到 r_
        }
      }
    );
}

} // namespace at::native
```