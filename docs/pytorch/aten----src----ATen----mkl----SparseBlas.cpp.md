# `.\pytorch\aten\src\ATen\mkl\SparseBlas.cpp`

```
/*
  Provides the implementations of MKL Sparse BLAS function templates.
*/
#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/mkl/Exceptions.h>
#include <ATen/mkl/SparseBlas.h>

namespace at::mkl::sparse {

namespace {

/*
  Converts a C++ complex number to MKL complex format.
  This function extracts real and imaginary parts of the input complex scalar
  and returns an MKL_Complex structure.
*/
template <typename scalar_t, typename MKL_Complex>
MKL_Complex to_mkl_complex(c10::complex<scalar_t> scalar) {
  MKL_Complex mkl_scalar;
  mkl_scalar.real = scalar.real();
  mkl_scalar.imag = scalar.imag();
  return mkl_scalar;
}

} // namespace

/*
  Template specialization for creating CSR format sparse matrices of type float.
  Calls MKL Sparse BLAS function mkl_sparse_s_create_csr with provided arguments.
*/
template <>
void create_csr<float>(MKL_SPARSE_CREATE_CSR_ARGTYPES(float)) {
  TORCH_MKLSPARSE_CHECK(mkl_sparse_s_create_csr(
      A, indexing, rows, cols, rows_start, rows_end, col_indx, values));
}

/*
  Template specialization for creating CSR format sparse matrices of type double.
  Calls MKL Sparse BLAS function mkl_sparse_d_create_csr with provided arguments.
*/
template <>
void create_csr<double>(MKL_SPARSE_CREATE_CSR_ARGTYPES(double)) {
  TORCH_MKLSPARSE_CHECK(mkl_sparse_d_create_csr(
      A, indexing, rows, cols, rows_start, rows_end, col_indx, values));
}

/*
  Template specialization for creating CSR format sparse matrices of type complex<float>.
  Calls MKL Sparse BLAS function mkl_sparse_c_create_csr with provided arguments,
  converting 'values' to MKL_Complex8*.
*/
template <>
void create_csr<c10::complex<float>>(
    MKL_SPARSE_CREATE_CSR_ARGTYPES(c10::complex<float>)) {
  TORCH_MKLSPARSE_CHECK(mkl_sparse_c_create_csr(
      A,
      indexing,
      rows,
      cols,
      rows_start,
      rows_end,
      col_indx,
      reinterpret_cast<MKL_Complex8*>(values)));
}

/*
  Template specialization for creating CSR format sparse matrices of type complex<double>.
  Calls MKL Sparse BLAS function mkl_sparse_z_create_csr with provided arguments,
  converting 'values' to MKL_Complex16*.
*/
template <>
void create_csr<c10::complex<double>>(
    MKL_SPARSE_CREATE_CSR_ARGTYPES(c10::complex<double>)) {
  TORCH_MKLSPARSE_CHECK(mkl_sparse_z_create_csr(
      A,
      indexing,
      rows,
      cols,
      rows_start,
      rows_end,
      col_indx,
      reinterpret_cast<MKL_Complex16*>(values)));
}

/*
  Template specialization for creating BSR format sparse matrices of type float.
  Calls MKL Sparse BLAS function mkl_sparse_s_create_bsr with provided arguments.
*/
template <>
void create_bsr<float>(MKL_SPARSE_CREATE_BSR_ARGTYPES(float)) {
  TORCH_MKLSPARSE_CHECK(mkl_sparse_s_create_bsr(
      A,
      indexing,
      block_layout,
      rows,
      cols,
      block_size,
      rows_start,
      rows_end,
      col_indx,
      values));
}

/*
  Template specialization for creating BSR format sparse matrices of type double.
  Calls MKL Sparse BLAS function mkl_sparse_d_create_bsr with provided arguments.
*/
template <>
void create_bsr<double>(MKL_SPARSE_CREATE_BSR_ARGTYPES(double)) {
  TORCH_MKLSPARSE_CHECK(mkl_sparse_d_create_bsr(
      A,
      indexing,
      block_layout,
      rows,
      cols,
      block_size,
      rows_start,
      rows_end,
      col_indx,
      values));
}

/*
  Template specialization for creating BSR format sparse matrices of type complex<float>.
  Calls MKL Sparse BLAS function mkl_sparse_c_create_bsr with provided arguments,
  converting 'values' to MKL_Complex8*.
*/
template <>
void create_bsr<c10::complex<float>>(
    MKL_SPARSE_CREATE_BSR_ARGTYPES(c10::complex<float>)) {
  TORCH_MKLSPARSE_CHECK(mkl_sparse_c_create_bsr(
      A,
      indexing,
      block_layout,
      rows,
      cols,
      block_size,
      rows_start,
      rows_end,
      col_indx,
      reinterpret_cast<MKL_Complex8*>(values)));
}

/*
  Template specialization for creating BSR format sparse matrices of type complex<double>.
  Calls MKL Sparse BLAS function mkl_sparse_z_create_bsr with provided arguments,
  converting 'values' to MKL_Complex16*.
*/
template <>
void create_bsr<c10::complex<double>>(
    MKL_SPARSE_CREATE_BSR_ARGTYPES(c10::complex<double>)) {
  TORCH_MKLSPARSE_CHECK(mkl_sparse_z_create_bsr(
      A,
      indexing,
      block_layout,
      rows,
      cols,
      block_size,
      rows_start,
      rows_end,
      col_indx,
      reinterpret_cast<MKL_Complex16*>(values)));
}

/*
  Template specialization for performing matrix-vector multiplication of type float.
  Calls MKL Sparse BLAS function mkl_sparse_s_mv with provided arguments.
*/
template <>
void mv<float>(MKL_SPARSE_MV_ARGTYPES(float)) {
  TORCH_MKLSPARSE_CHECK(
      mkl_sparse_s_mv(operation, alpha, A, descr, x, beta, y));
}
// 定义模板特化，实现稀疏矩阵向量乘法，针对 double 类型的操作
template <>
void mv<double>(MKL_SPARSE_MV_ARGTYPES(double)) {
  // 调用 MKL 库函数执行稀疏矩阵向量乘法，结果存储在 y 中
  TORCH_MKLSPARSE_CHECK(
      mkl_sparse_d_mv(operation, alpha, A, descr, x, beta, y));
}

// 定义模板特化，实现稀疏矩阵向量乘法，针对复数浮点类型 c10::complex<float> 的操作
template <>
void mv<c10::complex<float>>(MKL_SPARSE_MV_ARGTYPES(c10::complex<float>)) {
  // 调用 MKL 库函数执行复数稀疏矩阵向量乘法，将 x 和 y 转换为 MKL 库支持的复数格式
  TORCH_MKLSPARSE_CHECK(mkl_sparse_c_mv(
      operation,
      to_mkl_complex<float, MKL_Complex8>(alpha),
      A,
      descr,
      reinterpret_cast<const MKL_Complex8*>(x),
      to_mkl_complex<float, MKL_Complex8>(beta),
      reinterpret_cast<MKL_Complex8*>(y)));
}

// 定义模板特化，实现稀疏矩阵向量乘法，针对复数双精度类型 c10::complex<double> 的操作
template <>
void mv<c10::complex<double>>(MKL_SPARSE_MV_ARGTYPES(c10::complex<double>)) {
  // 调用 MKL 库函数执行复数稀疏矩阵向量乘法，将 x 和 y 转换为 MKL 库支持的复数格式
  TORCH_MKLSPARSE_CHECK(mkl_sparse_z_mv(
      operation,
      to_mkl_complex<double, MKL_Complex16>(alpha),
      A,
      descr,
      reinterpret_cast<const MKL_Complex16*>(x),
      to_mkl_complex<double, MKL_Complex16>(beta),
      reinterpret_cast<MKL_Complex16*>(y)));
}

// 定义模板特化，实现稀疏矩阵加法，针对单精度浮点数 float 的操作
template <>
void add<float>(MKL_SPARSE_ADD_ARGTYPES(float)) {
  // 调用 MKL 库函数执行稀疏矩阵加法，结果存储在 C 中
  TORCH_MKLSPARSE_CHECK(mkl_sparse_s_add(operation, A, alpha, B, C));
}

// 定义模板特化，实现稀疏矩阵加法，针对双精度浮点数 double 的操作
template <>
void add<double>(MKL_SPARSE_ADD_ARGTYPES(double)) {
  // 调用 MKL 库函数执行稀疏矩阵加法，结果存储在 C 中
  TORCH_MKLSPARSE_CHECK(mkl_sparse_d_add(operation, A, alpha, B, C));
}

// 定义模板特化，实现稀疏矩阵加法，针对复数浮点类型 c10::complex<float> 的操作
template <>
void add<c10::complex<float>>(MKL_SPARSE_ADD_ARGTYPES(c10::complex<float>)) {
  // 调用 MKL 库函数执行复数稀疏矩阵加法，将 alpha 转换为 MKL 库支持的复数格式
  TORCH_MKLSPARSE_CHECK(mkl_sparse_c_add(
      operation, A, to_mkl_complex<float, MKL_Complex8>(alpha), B, C));
}

// 定义模板特化，实现稀疏矩阵加法，针对复数双精度类型 c10::complex<double> 的操作
template <>
void add<c10::complex<double>>(MKL_SPARSE_ADD_ARGTYPES(c10::complex<double>)) {
  // 调用 MKL 库函数执行复数稀疏矩阵加法，将 alpha 转换为 MKL 库支持的复数格式
  TORCH_MKLSPARSE_CHECK(mkl_sparse_z_add(
      operation, A, to_mkl_complex<double, MKL_Complex16>(alpha), B, C));
}

// 定义模板特化，导出稀疏矩阵的 CSR 格式，针对单精度浮点数 float 的操作
template <>
void export_csr<float>(MKL_SPARSE_EXPORT_CSR_ARGTYPES(float)) {
  // 调用 MKL 库函数导出单精度浮点数稀疏矩阵的 CSR 格式数据
  TORCH_MKLSPARSE_CHECK(mkl_sparse_s_export_csr(
      source, indexing, rows, cols, rows_start, rows_end, col_indx, values));
}

// 定义模板特化，导出稀疏矩阵的 CSR 格式，针对双精度浮点数 double 的操作
template <>
void export_csr<double>(MKL_SPARSE_EXPORT_CSR_ARGTYPES(double)) {
  // 调用 MKL 库函数导出双精度浮点数稀疏矩阵的 CSR 格式数据
  TORCH_MKLSPARSE_CHECK(mkl_sparse_d_export_csr(
      source, indexing, rows, cols, rows_start, rows_end, col_indx, values));
}

// 定义模板特化，导出稀疏矩阵的 CSR 格式，针对复数浮点类型 c10::complex<float> 的操作
template <>
void export_csr<c10::complex<float>>(
    MKL_SPARSE_EXPORT_CSR_ARGTYPES(c10::complex<float>)) {
  // 调用 MKL 库函数导出复数浮点类型稀疏矩阵的 CSR 格式数据，将 values 转换为 MKL 库支持的复数格式
  TORCH_MKLSPARSE_CHECK(mkl_sparse_c_export_csr(
      source,
      indexing,
      rows,
      cols,
      rows_start,
      rows_end,
      col_indx,
      reinterpret_cast<MKL_Complex8**>(values)));
}

// 定义模板特化，导出稀疏矩阵的 CSR 格式，针对复数双精度类型 c10::complex<double> 的操作
template <>
void export_csr<c10::complex<double>>(
    MKL_SPARSE_EXPORT_CSR_ARGTYPES(c10::complex<double>)) {
  // 调用 MKL 库函数导出复数双精度类型稀疏矩阵的 CSR 格式数据，将 values 转换为 MKL 库支持的复数格式
  TORCH_MKLSPARSE_CHECK(mkl_sparse_z_export_csr(
      source,
      indexing,
      rows,
      cols,
      rows_start,
      rows_end,
      col_indx,
      reinterpret_cast<MKL_Complex16**>(values)));
}

// 定义模板特化，实现稀疏矩阵乘法，针对单精度浮点数 float 的操作
template <>
void mm<float>(MKL_SPARSE_MM_ARGTYPES(float)) {
  // 调用 MKL 库函数执行稀疏矩阵乘法，结果存储在 C 中
  TORCH_MKLSPARSE_CHECK(mkl_sparse_s_mm(
      operation, alpha, A, descr, layout, B, columns, ldb, beta, C, ldc));
}

// 定义模板特化，实现稀疏矩阵乘法，针对双精度浮点数 double 的操作
template <>
void mm<double>(MKL_SPARSE_MM_ARGTYPES(double)) {
  // 调用 MKL 库函数执行稀疏矩阵乘法，结果存储在 C 中
  TORCH_MKLSPARSE_CHECK(mkl_sparse_d_mm(
      operation, alpha, A, descr, layout, B, columns, ldb, beta, C, ldc));
}
// 实现稀疏矩阵乘法的模板特化，处理复数类型为 c10::complex<float>
template <>
void mm<c10::complex<float>>(MKL_SPARSE_MM_ARGTYPES(c10::complex<float>)) {
  // 调用 MKL 库中的稀疏矩阵乘法函数 mkl_sparse_c_mm
  TORCH_MKLSPARSE_CHECK(mkl_sparse_c_mm(
      operation,                                          // 操作类型 (稀疏矩阵乘法)
      to_mkl_complex<float, MKL_Complex8>(alpha),         // 转换 alpha 为 MKL_Complex8 类型
      A,                                                  // 稀疏矩阵 A
      descr,                                              // 稀疏矩阵描述符
      layout,                                             // 稀疏矩阵布局
      reinterpret_cast<const MKL_Complex8*>(B),           // 强制类型转换 B 到 const MKL_Complex8*
      columns,                                            // 列数
      ldb,                                                // B 的 leading dimension
      to_mkl_complex<float, MKL_Complex8>(beta),          // 转换 beta 为 MKL_Complex8 类型
      reinterpret_cast<MKL_Complex8*>(C),                 // 强制类型转换 C 到 MKL_Complex8*
      ldc));                                              // C 的 leading dimension
}

// 实现稀疏矩阵乘法的模板特化，处理复数类型为 c10::complex<double>
template <>
void mm<c10::complex<double>>(MKL_SPARSE_MM_ARGTYPES(c10::complex<double>)) {
  // 调用 MKL 库中的稀疏矩阵乘法函数 mkl_sparse_z_mm
  TORCH_MKLSPARSE_CHECK(mkl_sparse_z_mm(
      operation,                                          // 操作类型 (稀疏矩阵乘法)
      to_mkl_complex<double, MKL_Complex16>(alpha),       // 转换 alpha 为 MKL_Complex16 类型
      A,                                                  // 稀疏矩阵 A
      descr,                                              // 稀疏矩阵描述符
      layout,                                             // 稀疏矩阵布局
      reinterpret_cast<const MKL_Complex16*>(B),          // 强制类型转换 B 到 const MKL_Complex16*
      columns,                                            // 列数
      ldb,                                                // B 的 leading dimension
      to_mkl_complex<double, MKL_Complex16>(beta),        // 转换 beta 为 MKL_Complex16 类型
      reinterpret_cast<MKL_Complex16*>(C),                // 强制类型转换 C 到 MKL_Complex16*
      ldc));                                              // C 的 leading dimension
}

// 实现稀疏矩阵稠密矩阵乘法的模板特化，处理单精度浮点数
template <>
void spmmd<float>(MKL_SPARSE_SPMMD_ARGTYPES(float)) {
  // 调用 MKL 库中的稀疏矩阵稠密矩阵乘法函数 mkl_sparse_s_spmmd
  TORCH_MKLSPARSE_CHECK(mkl_sparse_s_spmmd(
      operation,                                          // 操作类型 (稀疏矩阵稠密矩阵乘法)
      A,                                                  // 稀疏矩阵 A
      B,                                                  // 稠密矩阵 B
      layout,                                             // 矩阵布局
      C,                                                  // 稠密矩阵 C
      ldc));                                              // C 的 leading dimension
}

// 实现稀疏矩阵稠密矩阵乘法的模板特化，处理双精度浮点数
template <>
void spmmd<double>(MKL_SPARSE_SPMMD_ARGTYPES(double)) {
  // 调用 MKL 库中的稀疏矩阵稠密矩阵乘法函数 mkl_sparse_d_spmmd
  TORCH_MKLSPARSE_CHECK(mkl_sparse_d_spmmd(
      operation,                                          // 操作类型 (稀疏矩阵稠密矩阵乘法)
      A,                                                  // 稀疏矩阵 A
      B,                                                  // 稠密矩阵 B
      layout,                                             // 矩阵布局
      C,                                                  // 稠密矩阵 C
      ldc));                                              // C 的 leading dimension
}

// 实现稀疏矩阵稠密矩阵乘法的模板特化，处理复数类型为 c10::complex<float>
template <>
void spmmd<c10::complex<float>>(MKL_SPARSE_SPMMD_ARGTYPES(c10::complex<float>)) {
  // 调用 MKL 库中的稀疏矩阵稠密矩阵乘法函数 mkl_sparse_c_spmmd
  TORCH_MKLSPARSE_CHECK(mkl_sparse_c_spmmd(
      operation,                                          // 操作类型 (稀疏矩阵稠密矩阵乘法)
      A,                                                  // 稀疏矩阵 A
      B,                                                  // 稠密矩阵 B
      layout,                                             // 矩阵布局
      reinterpret_cast<MKL_Complex8*>(C),                 // 强制类型转换 C 到 MKL_Complex8*
      ldc));                                              // C 的 leading dimension
}

// 实现稀疏矩阵稠密矩阵乘法的模板特化，处理复数类型为 c10::complex<double>
template <>
void spmmd<c10::complex<double>>(MKL_SPARSE_SPMMD_ARGTYPES(c10::complex<double>)) {
  // 调用 MKL 库中的稀疏矩阵稠密矩阵乘法函数 mkl_sparse_z_spmmd
  TORCH_MKLSPARSE_CHECK(mkl_sparse_z_spmmd(
      operation,                                          // 操作类型 (稀疏矩阵稠密矩阵乘法)
      A,                                                  // 稀疏矩阵 A
      B,                                                  // 稠密矩阵 B
      layout,                                             // 矩阵布局
      reinterpret_cast<MKL_Complex16*>(C),                // 强制类型转换 C 到 MKL_Complex16*
      ldc));                                              // C 的 leading dimension
}

// 实现稀疏矩阵向量三角解的模板特化，处理单精度浮点数
template <>
void trsv<float>(MKL_SPARSE_TRSV_ARGTYPES(float)) {
  // 调用 MKL 库中的稀疏矩阵向量三角解函数 mkl_sparse_s_trsv
  TORCH_MKLSPARSE_CHECK(mkl_sparse_s_trsv(
      operation,                                          // 操作类型 (稀疏矩阵向量三角解)
      alpha,                                              // α 系数
      A,                                                  // 稀疏矩阵 A
      descr,                                              // 稀疏矩阵描述符
      x,                                                  // 向量 x
      y));                                                // 向量 y
}

// 实现稀疏矩阵向量三角解的模板特化，处理双精度浮点数
template <>
void trsv<double>(MKL_SPARSE_TRSV_ARGTYPES(double)) {
  // 调用 MKL 库中的稀疏矩阵向量三角解函数 mkl_sparse_d_trsv
  TORCH_MKLSPARSE_CHECK(mkl_sparse_d_trsv(
      operation,                                          // 操作类型 (稀疏矩阵向量三角解)
      alpha,                                              // α 系数
      A,                                                  // 稀疏矩阵 A
      descr,                                              // 稀疏矩阵描述符
      x,                                                  // 向量 x
      y));                                                // 向量 y
}

// 实现稀疏矩阵向量三角解的模板特化，处理复数类型为 c10::complex<float>
template <>
void trsv<c10::complex<float>>(MK
// 实现稀疏矩阵的 TRSM 操作，针对复数类型 float 的特化版本
void trsm<c10::complex<float>>(MKL_SPARSE_TRSM_ARGTYPES(c10::complex<float>)) {
  // 调用 MKL 库函数进行稀疏矩阵的左下角或右上角解算
  TORCH_MKLSPARSE_CHECK(mkl_sparse_c_trsm(
      operation,                                     // 操作类型：解算左下角或右上角
      to_mkl_complex<float, MKL_Complex8>(alpha),    // 将 alpha 转换为 MKL_Complex8 类型
      A,                                             // 稀疏矩阵 A
      descr,                                         // 矩阵描述符
      layout,                                        // 矩阵布局
      reinterpret_cast<const MKL_Complex8*>(x),      // 输入向量 x 的复数表示
      columns,                                       // 列数
      ldx,                                           // x 的 leading dimension
      reinterpret_cast<MKL_Complex8*>(y),            // 输出向量 y 的复数表示
      ldy));                                         // y 的 leading dimension
}

// 实现稀疏矩阵的 TRSM 操作，针对复数类型 double 的特化版本
template <>
void trsm<c10::complex<double>>(
    MKL_SPARSE_TRSM_ARGTYPES(c10::complex<double>)) {
  // 调用 MKL 库函数进行稀疏矩阵的左下角或右上角解算
  TORCH_MKLSPARSE_CHECK(mkl_sparse_z_trsm(
      operation,                                     // 操作类型：解算左下角或右上角
      to_mkl_complex<double, MKL_Complex16>(alpha),  // 将 alpha 转换为 MKL_Complex16 类型
      A,                                             // 稀疏矩阵 A
      descr,                                         // 矩阵描述符
      layout,                                        // 矩阵布局
      reinterpret_cast<const MKL_Complex16*>(x),     // 输入向量 x 的复数表示
      columns,                                       // 列数
      ldx,                                           // x 的 leading dimension
      reinterpret_cast<MKL_Complex16*>(y),           // 输出向量 y 的复数表示
      ldy));                                         // y 的 leading dimension
}

} // namespace at::mkl::sparse
```