# `.\pytorch\aten\src\ATen\mkl\SparseBlas.h`

```
#pragma once

/*
  提供 MKL 稀疏 BLAS 函数的子集作为模板：

    mv<scalar_t>(operation, alpha, A, descr, x, beta, y)

  其中 scalar_t 可以是 double、float、c10::complex<double> 或 c10::complex<float>。
  这些函数位于 at::mkl::sparse 命名空间中。
*/

#include <c10/util/Exception.h>
#include <c10/util/complex.h>

#include <mkl_spblas.h>

namespace at::mkl::sparse {

// 定义宏，用于创建 CSR 格式稀疏矩阵所需的参数类型
#define MKL_SPARSE_CREATE_CSR_ARGTYPES(scalar_t)                              \
  sparse_matrix_t *A, const sparse_index_base_t indexing, const MKL_INT rows, \
      const MKL_INT cols, MKL_INT *rows_start, MKL_INT *rows_end,             \
      MKL_INT *col_indx, scalar_t *values

// 创建 CSR 格式稀疏矩阵的模板函数声明
template <typename scalar_t>
inline void create_csr(MKL_SPARSE_CREATE_CSR_ARGTYPES(scalar_t)) {
  // 抛出异常，表示此函数未实现
  TORCH_INTERNAL_ASSERT(
      false,
      "at::mkl::sparse::create_csr: not implemented for ",
      typeid(scalar_t).name());
}

// 显式实例化模板函数，针对不同的 scalar_t 类型
template <>
void create_csr<float>(MKL_SPARSE_CREATE_CSR_ARGTYPES(float));
template <>
void create_csr<double>(MKL_SPARSE_CREATE_CSR_ARGTYPES(double));
template <>
void create_csr<c10::complex<float>>(
    MKL_SPARSE_CREATE_CSR_ARGTYPES(c10::complex<float>));
template <>
void create_csr<c10::complex<double>>(
    MKL_SPARSE_CREATE_CSR_ARGTYPES(c10::complex<double>));

// 定义宏，用于创建 BSR 格式稀疏矩阵所需的参数类型
#define MKL_SPARSE_CREATE_BSR_ARGTYPES(scalar_t)                   \
  sparse_matrix_t *A, const sparse_index_base_t indexing,          \
      const sparse_layout_t block_layout, const MKL_INT rows,      \
      const MKL_INT cols, MKL_INT block_size, MKL_INT *rows_start, \
      MKL_INT *rows_end, MKL_INT *col_indx, scalar_t *values

// 创建 BSR 格式稀疏矩阵的模板函数声明
template <typename scalar_t>
inline void create_bsr(MKL_SPARSE_CREATE_BSR_ARGTYPES(scalar_t)) {
  // 抛出异常，表示此函数未实现
  TORCH_INTERNAL_ASSERT(
      false,
      "at::mkl::sparse::create_bsr: not implemented for ",
      typeid(scalar_t).name());
}

// 显式实例化模板函数，针对不同的 scalar_t 类型
template <>
void create_bsr<float>(MKL_SPARSE_CREATE_BSR_ARGTYPES(float));
template <>
void create_bsr<double>(MKL_SPARSE_CREATE_BSR_ARGTYPES(double));
template <>
void create_bsr<c10::complex<float>>(
    MKL_SPARSE_CREATE_BSR_ARGTYPES(c10::complex<float>));
template <>
void create_bsr<c10::complex<double>>(
    MKL_SPARSE_CREATE_BSR_ARGTYPES(c10::complex<double>));

// 定义宏，用于稀疏矩阵向量乘法所需的参数类型
#define MKL_SPARSE_MV_ARGTYPES(scalar_t)                        \
  const sparse_operation_t operation, const scalar_t alpha,     \
      const sparse_matrix_t A, const struct matrix_descr descr, \
      const scalar_t *x, const scalar_t beta, scalar_t *y

// 稀疏矩阵向量乘法的模板函数声明
template <typename scalar_t>
inline void mv(MKL_SPARSE_MV_ARGTYPES(scalar_t)) {
  // 抛出异常，表示此函数未实现
  TORCH_INTERNAL_ASSERT(
      false,
      "at::mkl::sparse::mv: not implemented for ",
      typeid(scalar_t).name());
}

// 显式实例化模板函数，针对不同的 scalar_t 类型
template <>
void mv<float>(MKL_SPARSE_MV_ARGTYPES(float));
template <>
void mv<double>(MKL_SPARSE_MV_ARGTYPES(double));
template <>
void mv<c10::complex<float>>(MKL_SPARSE_MV_ARGTYPES(c10::complex<float>));
template <>
void mv<c10::complex<double>>(MKL_SPARSE_MV_ARGTYPES(c10::complex<double>));

} // namespace at::mkl::sparse
#define MKL_SPARSE_ADD_ARGTYPES(scalar_t)                      \
  const sparse_operation_t operation, const sparse_matrix_t A, \
      const scalar_t alpha, const sparse_matrix_t B, sparse_matrix_t *C

定义了一个宏 `MKL_SPARSE_ADD_ARGTYPES`，用于声明 `add` 函数的参数列表，包括操作类型 `operation`、稀疏矩阵 `A`、标量 `alpha`、稀疏矩阵 `B` 以及指向稀疏矩阵指针 `C`。


template <typename scalar_t>
inline void add(MKL_SPARSE_ADD_ARGTYPES(scalar_t)) {
  TORCH_INTERNAL_ASSERT(
      false,
      "at::mkl::sparse::add: not implemented for ",
      typeid(scalar_t).name());
}

模板函数 `add`，针对模板类型 `scalar_t`，通过宏展开定义了 `add` 函数，抛出一个断言错误，表明该函数未实现对当前类型 `scalar_t` 的支持。


template <>
void add<float>(MKL_SPARSE_ADD_ARGTYPES(float));
template <>
void add<double>(MKL_SPARSE_ADD_ARGTYPES(double));
template <>
void add<c10::complex<float>>(MKL_SPARSE_ADD_ARGTYPES(c10::complex<float>));
template <>
void add<c10::complex<double>>(MKL_SPARSE_ADD_ARGTYPES(c10::complex<double>));

针对不同的具体类型（`float`、`double`、`c10::complex<float>`、`c10::complex<double>`），显式实例化了 `add` 函数模板，为每种类型生成特定的函数定义。


#define MKL_SPARSE_EXPORT_CSR_ARGTYPES(scalar_t)                              \
  const sparse_matrix_t source, sparse_index_base_t *indexing, MKL_INT *rows, \
      MKL_INT *cols, MKL_INT **rows_start, MKL_INT **rows_end,                \
      MKL_INT **col_indx, scalar_t **values

定义了一个宏 `MKL_SPARSE_EXPORT_CSR_ARGTYPES`，用于声明 `export_csr` 函数的参数列表，包括源稀疏矩阵 `source`、索引基数指针 `indexing`、行和列数组指针、行起始和结束指针、列索引指针以及值数组指针。


template <typename scalar_t>
inline void export_csr(MKL_SPARSE_EXPORT_CSR_ARGTYPES(scalar_t)) {
  TORCH_INTERNAL_ASSERT(
      false,
      "at::mkl::sparse::export_csr: not implemented for ",
      typeid(scalar_t).name());
}

模板函数 `export_csr`，针对模板类型 `scalar_t`，通过宏展开定义了 `export_csr` 函数，抛出一个断言错误，表明该函数未实现对当前类型 `scalar_t` 的支持。


template <>
void export_csr<float>(MKL_SPARSE_EXPORT_CSR_ARGTYPES(float));
template <>
void export_csr<double>(MKL_SPARSE_EXPORT_CSR_ARGTYPES(double));
template <>
void export_csr<c10::complex<float>>(
    MKL_SPARSE_EXPORT_CSR_ARGTYPES(c10::complex<float>));
template <>
void export_csr<c10::complex<double>>(
    MKL_SPARSE_EXPORT_CSR_ARGTYPES(c10::complex<double>));

针对不同的具体类型（`float`、`double`、`c10::complex<float>`、`c10::complex<double>`），显式实例化了 `export_csr` 函数模板，为每种类型生成特定的函数定义。


#define MKL_SPARSE_MM_ARGTYPES(scalar_t)                                      \
  const sparse_operation_t operation, const scalar_t alpha,                   \
      const sparse_matrix_t A, const struct matrix_descr descr,               \
      const sparse_layout_t layout, const scalar_t *B, const MKL_INT columns, \
      const MKL_INT ldb, const scalar_t beta, scalar_t *C, const MKL_INT ldc

定义了一个宏 `MKL_SPARSE_MM_ARGTYPES`，用于声明 `mm` 函数的参数列表，包括操作类型 `operation`、标量 `alpha`、稀疏矩阵 `A`、矩阵描述符 `descr`、布局类型 `layout`、输入矩阵 `B`、列数 `columns`、输入矩阵的列偏移 `ldb`、标量 `beta`、输出矩阵 `C` 以及输出矩阵的列偏移 `ldc`。


template <typename scalar_t>
inline void mm(MKL_SPARSE_MM_ARGTYPES(scalar_t)) {
  TORCH_INTERNAL_ASSERT(
      false,
      "at::mkl::sparse::mm: not implemented for ",
      typeid(scalar_t).name());
}

模板函数 `mm`，针对模板类型 `scalar_t`，通过宏展开定义了 `mm` 函数，抛出一个断言错误，表明该函数未实现对当前类型 `scalar_t` 的支持。


template <>
void mm<float>(MKL_SPARSE_MM_ARGTYPES(float));
template <>
void mm<double>(MKL_SPARSE_MM_ARGTYPES(double));
template <>
void mm<c10::complex<float>>(MKL_SPARSE_MM_ARGTYPES(c10::complex<float>));
template <>
void mm<c10::complex<double>>(MKL_SPARSE_MM_ARGTYPES(c10::complex<double>));

针对不同的具体类型（`float`、`double`、`c10::complex<float>`、`c10::complex<double>`），显式实例化了 `mm` 函数模板，为每种类型生成特定的函数定义。


#define MKL_SPARSE_SPMMD_ARGTYPES(scalar_t)                               \
  const sparse_operation_t operation, const sparse_matrix_t A,            \
      const sparse_matrix_t B, const sparse_layout_t layout, scalar_t *C, \
      const MKL_INT ldc

定义了一个宏 `MKL_SPARSE_SPMMD_ARGTYPES`，用于声明 `spmmd` 函数的参数列表，包括操作类型 `operation`、稀疏矩阵 `A`、稀疏矩阵 `B`、布局类型 `layout`、输出矩阵 `C` 以及输出矩阵的列偏移 `ldc`。


template <typename scalar_t>
inline void spmmd(MKL_SPARSE_SPMMD_ARGTYPES(scalar_t)) {
  TORCH_INTERNAL_ASSERT(
      false,
      "at::mkl::sparse::spmmd: not implemented for ",
      typeid(scalar_t).name());
}

模板函数 `spmmd`，针对模板类型 `scalar_t`，通过宏展开定义了 `spmmd` 函数，抛出一个断言错误，表明该函数未实现对当前类型 `scalar_t` 的支持。


template <>
void spmmd<float>(MKL_SPARSE_SPMMD_ARGTYPES(float));
template <>
void spmmd<double>(MKL_SPARSE_SPMMD_ARGTYPES(double));
template <>
void spmmd<c10::complex<float>>(MKL_SPARSE_SPMMD_ARGTYPES(c10::complex<float>));
template <>
void spmmd<c10::complex<double>>(MKL_SPARSE_SPMMD_ARGTYPES(c10::complex<double>));

针对不同的具体类型（`float`、`double`、`c10::complex<float>`、`c10::complex<double>`），显式实例化了 `spmmd` 函数模板，为每种类型生成特定的函数定义。
// 定义一个模板函数 spmmd，接受 float 类型的参数，使用 MKL 的稀疏矩阵乘法运算接口
void spmmd<float>(MKL_SPARSE_SPMMD_ARGTYPES(float));

// 定义一个模板函数 spmmd，接受 double 类型的参数，使用 MKL 的稀疏矩阵乘法运算接口
template <>
void spmmd<double>(MKL_SPARSE_SPMMD_ARGTYPES(double));

// 定义一个模板函数 spmmd，接受 c10::complex<float> 类型的参数，使用 MKL 的稀疏矩阵乘法运算接口
template <>
void spmmd<c10::complex<float>>(MKL_SPARSE_SPMMD_ARGTYPES(c10::complex<float>));

// 定义一个模板函数 spmmd，接受 c10::complex<double> 类型的参数，使用 MKL 的稀疏矩阵乘法运算接口
template <>
void spmmd<c10::complex<double>>(
    MKL_SPARSE_SPMMD_ARGTYPES(c10::complex<double>));

// 定义宏 MKL_SPARSE_TRSV_ARGTYPES，用于生成模板函数 trsv 的参数类型列表，支持不同类型的稀疏矩阵向量三角求解
#define MKL_SPARSE_TRSV_ARGTYPES(scalar_t)                      \
  const sparse_operation_t operation, const scalar_t alpha,     \
      const sparse_matrix_t A, const struct matrix_descr descr, \
      const scalar_t *x, scalar_t *y

// 定义模板函数 trsv，用于稀疏矩阵向量三角求解，如果调用了该函数，会触发断言错误提示
template <typename scalar_t>
inline void trsv(MKL_SPARSE_TRSV_ARGTYPES(scalar_t)) {
  TORCH_INTERNAL_ASSERT(
      false,
      "at::mkl::sparse::trsv: not implemented for ",
      typeid(scalar_t).name());
}

// 显式实例化模板函数 trsv，支持 float 类型的稀疏矩阵向量三角求解
template <>
void trsv<float>(MKL_SPARSE_TRSV_ARGTYPES(float));

// 显式实例化模板函数 trsv，支持 double 类型的稀疏矩阵向量三角求解
template <>
void trsv<double>(MKL_SPARSE_TRSV_ARGTYPES(double));

// 显式实例化模板函数 trsv，支持 c10::complex<float> 类型的稀疏矩阵向量三角求解
template <>
void trsv<c10::complex<float>>(MKL_SPARSE_TRSV_ARGTYPES(c10::complex<float>));

// 显式实例化模板函数 trsv，支持 c10::complex<double> 类型的稀疏矩阵向量三角求解
template <>
void trsv<c10::complex<double>>(MKL_SPARSE_TRSV_ARGTYPES(c10::complex<double>));

// 定义宏 MKL_SPARSE_TRSM_ARGTYPES，用于生成模板函数 trsm 的参数类型列表，支持不同类型的稀疏矩阵矩阵三角求解
#define MKL_SPARSE_TRSM_ARGTYPES(scalar_t)                                    \
  const sparse_operation_t operation, const scalar_t alpha,                   \
      const sparse_matrix_t A, const struct matrix_descr descr,               \
      const sparse_layout_t layout, const scalar_t *x, const MKL_INT columns, \
      const MKL_INT ldx, scalar_t *y, const MKL_INT ldy

// 定义模板函数 trsm，用于稀疏矩阵矩阵三角求解，如果调用了该函数，会触发断言错误提示
template <typename scalar_t>
inline void trsm(MKL_SPARSE_TRSM_ARGTYPES(scalar_t)) {
  TORCH_INTERNAL_ASSERT(
      false,
      "at::mkl::sparse::trsm: not implemented for ",
      typeid(scalar_t).name());
}

// 显式实例化模板函数 trsm，支持 float 类型的稀疏矩阵矩阵三角求解
template <>
void trsm<float>(MKL_SPARSE_TRSM_ARGTYPES(float));

// 显式实例化模板函数 trsm，支持 double 类型的稀疏矩阵矩阵三角求解
template <>
void trsm<double>(MKL_SPARSE_TRSM_ARGTYPES(double));

// 显式实例化模板函数 trsm，支持 c10::complex<float> 类型的稀疏矩阵矩阵三角求解
template <>
void trsm<c10::complex<float>>(MKL_SPARSE_TRSM'tYPES(c10::complex<float>));

// 显式实例化模板函数 trsm，支持 c10::complex<double> 类型的稀疏矩阵矩阵三角求解
template <>
void trsm<c10::complex<double>>(MKL_SPARSE_TRSM_ARGTYPES(c10::complex<double>));

// 结束命名空间 at::mkl::sparse
} // namespace at::mkl::sparse
```