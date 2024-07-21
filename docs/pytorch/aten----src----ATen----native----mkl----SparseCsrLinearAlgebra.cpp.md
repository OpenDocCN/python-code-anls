# `.\pytorch\aten\src\ATen\native\mkl\SparseCsrLinearAlgebra.cpp`

```
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/mkl/SparseCsrLinearAlgebra.h>
#include <ATen/native/SparseTensorUtils.h>

// Don't compile with MKL for macos since linking the sparse MKL routines
// needs some build fixes.
// Macros source:
// https://web.archive.org/web/20191012035921/http://nadeausoftware.com/articles/2012/01/c_c_tip_how_use_compiler_predefined_macros_detect_operating_system
#if !AT_MKL_ENABLED() || defined(__APPLE__) || \
    defined(__MACH__)

namespace at {
namespace sparse_csr {

// Define a placeholder method _sparse_mm_mkl_ that throws an error message when called on macOS.
Tensor& _sparse_mm_mkl_(
    Tensor& self,                               // Reference to the calling Tensor object
    const SparseCsrTensor& sparse_,             // Reference to the Sparse CSR Tensor object
    const Tensor& dense,                        // Reference to the dense Tensor
    const Tensor& t,                            // Reference to another Tensor object
    const Scalar& alpha,                        // Scalar value alpha
    const Scalar& beta) {                       // Scalar value beta
#if __APPLE__ || __MACH__
  AT_ERROR("sparse_mm_mkl: MKL support is disabled on macos/iOS."); // Throw error for macOS/iOS
#else
  AT_ERROR("sparse_mm_mkl: ATen not compiled with MKL support");    // Throw error for other platforms without MKL support
#endif
  return self; // Return self to avoid compiler warnings
}

} // namespace sparse_csr
} // namespace at

#else // AT_MKL_ENABLED

#include <ATen/mkl/Descriptors.h>
#include <ATen/mkl/Exceptions.h>
#include <ATen/mkl/Limits.h>
#include <mkl.h>
#include <mkl_spblas.h>

#include <ATen/Dispatch.h>
#include <ATen/ExpandUtils.h>
#include <ATen/SparseCsrTensorImpl.h>

namespace at {
namespace sparse_csr {

#ifdef MKL_ILP64
static constexpr ScalarType TORCH_INT_TYPE = at::kLong; // Define TORCH_INT_TYPE based on MKL_ILP64
#else
static constexpr ScalarType TORCH_INT_TYPE = at::kInt;  // Define TORCH_INT_TYPE for default case
#endif

// Define a class SparseCsrMKLInterface to interact with MKL sparse matrices
class SparseCsrMKLInterface {
 private:
  sparse_matrix_t A{nullptr};     // Pointer to the sparse matrix A
  matrix_descr desc;              // Descriptor for matrix properties

 public:
  // Constructor for double precision values
  SparseCsrMKLInterface(
      MKL_INT* col_indices,        // Pointer to column indices
      MKL_INT* crow_indices,       // Pointer to row indices
      double* values,              // Pointer to matrix values
      MKL_INT nrows,               // Number of rows
      MKL_INT ncols) {             // Number of columns
    desc.type = SPARSE_MATRIX_TYPE_GENERAL;  // Set matrix type to general
    int retval = mkl_sparse_d_create_csr(    // Create CSR matrix from double values
        &A,
        SPARSE_INDEX_BASE_ZERO,
        nrows,
        ncols,
        crow_indices,
        crow_indices + 1,
        col_indices,
        values);
    TORCH_CHECK(
        retval == 0,
        "mkl_sparse_d_create_csr failed with error code: ",
        retval);  // Check for creation success
  }

  // Constructor for single precision values
  SparseCsrMKLInterface(
      MKL_INT* col_indices,        // Pointer to column indices
      MKL_INT* crow_indices,       // Pointer to row indices
      float* values,               // Pointer to matrix values
      MKL_INT nrows,               // Number of rows
      MKL_INT ncols) {             // Number of columns
    desc.type = SPARSE_MATRIX_TYPE_GENERAL;  // Set matrix type to general
    int retval = mkl_sparse_s_create_csr(    // Create CSR matrix from single precision values
        &A,
        SPARSE_INDEX_BASE_ZERO,
        nrows,
        ncols,
        crow_indices,
        crow_indices + 1,
        col_indices,
        values);
    TORCH_CHECK(
        retval == 0,
        "mkl_sparse_s_create_csr failed with error code: ",
        retval);  // Check for creation success
  }

  // Method to perform sparse matrix multiplication: res(nrows, dense_ncols) = (sparse(nrows * ncols) @ dense(ncols x dense_ncols))
  inline void sparse_mm(
      float* res,                 // Pointer to the result matrix
      float* dense,               // Pointer to the dense matrix
      float alpha,                // Scalar alpha
      float beta,                 // Scalar beta
      MKL_INT nrows,              // Number of rows
      MKL_INT ncols,              // Number of columns
      MKL_INT dense_ncols) {      // Number of dense columns
    int stat;
  // 如果稠密矩阵列数为1，则执行稀疏矩阵向量乘法
  if (dense_ncols == 1) {
    // 调用 MKL 库执行稀疏矩阵向量乘法操作
    stat = mkl_sparse_s_mv(
      SPARSE_OPERATION_NON_TRANSPOSE,  // 不转置操作
      alpha,                           // 系数 alpha
      A,                               // 稀疏矩阵 A
      desc,                            // 矩阵描述符
      dense,                           // 稠密向量 dense
      beta,                            // 系数 beta
      res);                            // 结果存储在 res 中
    // 检查操作是否成功，若不成功则输出错误信息和错误码
    TORCH_CHECK(stat == 0, "mkl_sparse_s_mv failed with error code: ", stat);
  } else {
    // 否则执行稀疏矩阵乘法
    stat = mkl_sparse_s_mm(
      SPARSE_OPERATION_NON_TRANSPOSE,  // 不转置操作
      alpha,                           // 系数 alpha
      A,                               // 稀疏矩阵 A
      desc,                            // 矩阵描述符
      SPARSE_LAYOUT_ROW_MAJOR,         // 稀疏矩阵以行优先布局
      dense,                           // 稠密矩阵 dense
      nrows,                           // 稀疏矩阵的行数
      ncols,                           // 稀疏矩阵的列数
      beta,                            // 系数 beta
      res,                             // 结果存储在 res 中
      dense_ncols);                    // 稠密矩阵的列数
    // 检查操作是否成功，若不成功则输出错误信息和错误码
    TORCH_CHECK(stat == 0, "mkl_sparse_s_mm failed with error code: ", stat);
  }
}

// 定义一个内联函数 sparse_mm，执行稀疏矩阵和稠密矩阵之间的乘法
inline void sparse_mm(
    double* res,                       // 结果存储在 res 中
    double* dense,                     // 稠密矩阵 dense
    double alpha,                      // 系数 alpha
    double beta,                       // 系数 beta
    MKL_INT nrows,                     // 稀疏矩阵的行数
    MKL_INT ncols,                     // 稀疏矩阵的列数
    MKL_INT dense_ncols) {             // 稠密矩阵的列数
  int stat;                            // 定义一个整型变量 stat，用于存储操作的状态
  // 如果稠密矩阵列数为1，则执行双精度稀疏矩阵向量乘法
  if (dense_ncols == 1) {
    // 调用 MKL 库执行双精度稀疏矩阵向量乘法操作
    stat = mkl_sparse_d_mv(
      SPARSE_OPERATION_NON_TRANSPOSE,  // 不转置操作
      alpha,                           // 系数 alpha
      A,                               // 稀疏矩阵 A
      desc,                            // 矩阵描述符
      dense,                           // 稠密向量 dense
      beta,                            // 系数 beta
      res);                            // 结果存储在 res 中
    // 检查操作是否成功，若不成功则输出错误信息和错误码
    TORCH_CHECK(stat == 0, "mkl_sparse_d_mv failed with error code: ", stat);
  }
  else {
    // 否则执行双精度稀疏矩阵乘法
    stat = mkl_sparse_d_mm(
      SPARSE_OPERATION_NON_TRANSPOSE,  // 不转置操作
      alpha,                           // 系数 alpha
      A,                               // 稀疏矩阵 A
      desc,                            // 矩阵描述符
      SPARSE_LAYOUT_ROW_MAJOR,         // 稀疏矩阵以行优先布局
      dense,                           // 稠密矩阵 dense
      nrows,                           // 稀疏矩阵的行数
      ncols,                           // 稀疏矩阵的列数
      beta,                            // 系数 beta
      res,                             // 结果存储在 res 中
      dense_ncols);                    // 稠密矩阵的列数
    // 检查操作是否成功，若不成功则输出错误信息和错误码
    TORCH_CHECK(stat == 0, "mkl_sparse_d_mm failed with error code: ", stat);
  }
}

// 稀疏矩阵 CSR 格式的 MKL 接口的析构函数，销毁稀疏矩阵 A
~SparseCsrMKLInterface() {
  mkl_sparse_destroy(A);  // 使用 MKL 函数销毁稀疏矩阵 A
}
// 以下代码为一个 C++ 函数模板，用于执行稀疏矩阵与密集矩阵的乘法，基于 MKL 接口实现。

template <typename scalar_t>
static inline void sparse_mm_mkl_template(
    Tensor& res,                            // 结果张量
    const Tensor& col_indices,              // 稀疏矩阵的列索引张量
    const Tensor& crow_indices,             // 稀疏矩阵的行指针张量
    const Tensor& values,                   // 稀疏矩阵的值张量
    const Tensor& dense,                    // 密集矩阵
    const Tensor& t,                        // 未使用的张量
    const Scalar& alpha,                    // 乘法的标量因子 alpha
    const Scalar& beta,                     // 乘法的标量因子 beta
    IntArrayRef size,                       // 稀疏矩阵的大小
    IntArrayRef dense_size) {               // 密集矩阵的大小

  // 使用 SparseCsrMKLInterface 创建 MKL 接口实例，传入稀疏矩阵的相关数据
  SparseCsrMKLInterface mkl_impl(
      col_indices.data_ptr<MKL_INT>(),     // 稀疏矩阵列索引的数据指针
      crow_indices.data_ptr<MKL_INT>(),    // 稀疏矩阵行指针的数据指针
      values.data_ptr<scalar_t>(),         // 稀疏矩阵值的数据指针
      size[0],                             // 稀疏矩阵的行数
      size[1]);                            // 稀疏矩阵的列数

  // 调用 MKL 接口实例的稀疏矩阵乘法函数，计算结果存储在 res 中
  mkl_impl.sparse_mm(
      res.data_ptr<scalar_t>(),            // 结果张量的数据指针
      dense.data_ptr<scalar_t>(),          // 密集矩阵的数据指针
      alpha.to<scalar_t>(),                // alpha 转换为 scalar_t 类型
      beta.to<scalar_t>(),                 // beta 转换为 scalar_t 类型
      size[0],                             // 稀疏矩阵的行数
      size[1],                             // 稀疏矩阵的列数
      dense_size[1]);                      // 密集矩阵的列数
}

// 检查是否使用 MKL 的 int32 索引类型
static bool inline constexpr is_mkl_int32_index() {
#ifdef MKL_ILP64
  return false;
#else
  return true;
#endif
}

// 执行稀疏矩阵与密集矩阵的乘法，根据当前编译设置可能进行类型转换警告
Tensor& _sparse_mm_mkl_(
    Tensor& self,                           // 自身张量，存储乘法结果
    const SparseCsrTensor& sparse_,         // 稀疏矩阵张量
    const Tensor& dense,                    // 密集矩阵
    const Tensor& t,                        // 未使用的张量
    const Scalar& alpha,                    // 乘法的标量因子 alpha
    const Scalar& beta) {                   // 乘法的标量因子 beta

  // 如果使用 int32 索引类型，检查并转换稀疏矩阵的索引类型
  if (is_mkl_int32_index()) {
    if (sparse_.crow_indices().scalar_type() != kInt) {
      TORCH_WARN(
          "Pytorch is compiled with MKL LP64 and will convert crow_indices to int32.");
    }
    if (sparse_.col_indices().scalar_type() != kInt) {
      TORCH_WARN(
          "Pytorch is compiled with MKL LP64 and will convert col_indices to int32.");
    }
  } else { // 为了将来使用 MKL ILP64 的情况进行未来保护
    if (sparse_.crow_indices().scalar_type() != kLong) {
      TORCH_WARN(
          "Pytorch is compiled with MKL ILP64 and will convert crow_indices dtype to int64.");
    }
    if (sparse_.col_indices().scalar_type() != kLong) {
      TORCH_WARN(
          "Pytorch is compiled with MKL ILP64 and will convert col_indices dtype to int64.");
    }
  }

  // 使用 AT_DISPATCH_FLOATING_TYPES 宏来处理不同的浮点类型，调用对应的模板函数进行稀疏矩阵乘法
  AT_DISPATCH_FLOATING_TYPES(
      dense.scalar_type(), "addmm_sparse_csr_dense", [&] {
        sparse_mm_mkl_template<scalar_t>(
            self,                           // 结果张量
            sparse_.col_indices().to(TORCH_INT_TYPE),    // 稀疏矩阵的列索引，转换为 int 类型
            sparse_.crow_indices().to(TORCH_INT_TYPE),   // 稀疏矩阵的行指针，转换为 int 类型
            sparse_.values(),                           // 稀疏矩阵的值
            dense,                                      // 密集矩阵
            t,                                          // 未使用的张量
            alpha,                                      // 乘法的标量因子 alpha
            beta,                                       // 乘法的标量因子 beta
            sparse_.sizes(),                            // 稀疏矩阵的大小
            dense.sizes());                             // 密集矩阵的大小
      });

  // 返回乘法结果张量
  return self;
}

// 命名空间结束标记
} // namespace native
} // namespace at

#endif // AT_MKL_ENABLED
```