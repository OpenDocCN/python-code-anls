# `.\pytorch\aten\src\ATen\native\sparse\cuda\SparseBlasImpl.h`

```py
#pragma once
// 预处理指令，确保头文件只被编译一次

#include <ATen/SparseCsrTensorUtils.h>
// 包含稀疏 CSR 张量的实用函数头文件
#include <ATen/Tensor.h>
// 包含张量的头文件
#include <ATen/core/Scalar.h>
// 包含标量的头文件

namespace at::native::sparse::impl::cuda {

void addmm_out_sparse_csr(
    const Tensor& input,
    // 输入张量
    const at::sparse_csr::SparseCsrTensor& mat1,
    // 稀疏 CSR 格式的第一个输入矩阵
    const Tensor& mat2,
    // 第二个输入矩阵
    const Scalar& beta,
    // 乘以输入矩阵 mat1 的标量 beta
    const Scalar& alpha,
    // 乘以输入矩阵 mat2 的标量 alpha
    const Tensor& result);
    // 输出结果张量

void addmv_out_sparse_csr(
    const at::sparse_csr::SparseCsrTensor& mat,
    // 稀疏 CSR 格式的矩阵
    const Tensor& vec,
    // 输入向量
    const Scalar& beta,
    // 乘以输入向量的标量 beta
    const Scalar& alpha,
    // 乘以矩阵 mat 的标量 alpha
    const Tensor& result);
    // 输出结果张量

void add_out_sparse_csr(
    const at::sparse_csr::SparseCsrTensor& mat1,
    // 第一个输入稀疏 CSR 格式的矩阵
    const at::sparse_csr::SparseCsrTensor& mat2,
    // 第二个输入稀疏 CSR 格式的矩阵
    const Scalar& alpha,
    // 第一个矩阵的乘法标量 alpha
    const Scalar& beta,
    // 第二个矩阵的乘法标量 beta
    const at::sparse_csr::SparseCsrTensor& result);
    // 输出结果稀疏 CSR 格式的矩阵

void triangular_solve_out_sparse_csr(
    const at::sparse_csr::SparseCsrTensor& A,
    // 稀疏 CSR 格式的矩阵 A
    const Tensor& B,
    // 输入张量 B
    const Tensor& X,
    // 输出张量 X
    bool upper,
    // 是否解上三角部分
    bool transpose,
    // 是否转置
    bool unitriangular);
    // 是否单位三角化

void sampled_addmm_out_sparse_csr(
    const Tensor& mat1,
    // 输入张量 mat1
    const Tensor& mat2,
    // 输入张量 mat2
    const Scalar& beta,
    // 乘以输入张量 mat1 的标量 beta
    const Scalar& alpha,
    // 乘以输入张量 mat2 的标量 alpha
    const at::sparse_csr::SparseCsrTensor& result);
    // 输出结果稀疏 CSR 格式的张量 result

} // namespace at::native::sparse::impl::cuda
// 结束命名空间声明
```