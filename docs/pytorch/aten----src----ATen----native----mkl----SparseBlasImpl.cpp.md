# `.\pytorch\aten\src\ATen\native\mkl\SparseBlasImpl.cpp`

```py
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/SparseCsrTensorImpl.h>
#include <ATen/Tensor.h>
#include <ATen/mkl/Sparse.h>
#include <ATen/native/LinearAlgebraUtils.h>
#include <ATen/SparseCsrTensorUtils.h>
#include <ATen/native/mkl/SparseBlasImpl.h>

#include <c10/core/ScalarType.h>
#include <c10/util/MaybeOwned.h>

#if AT_USE_MKL_SPARSE()
#include <ATen/mkl/SparseBlas.h>
#include <ATen/mkl/SparseDescriptors.h>
#include <ATen/mkl/Utils.h>
#endif

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/cat.h>
#include <ATen/ops/sparse_coo_tensor.h>
#endif

namespace at {
namespace native {
namespace sparse {
namespace impl {
namespace mkl {

namespace {

#if AT_USE_MKL_SPARSE()
/*
  Prepares a dense matrix for MKL operations.

  Args:
  * `tensor` - Input tensor to be prepared.

  Returns:
  * A `MaybeOwned<Tensor>` object wrapping the prepared tensor.
*/
c10::MaybeOwned<Tensor> prepare_dense_matrix_for_mkl(
    const Tensor& tensor) {
  // Checks if the tensor is non-overlapping and dense, or if it's already in a compatible order
  if (tensor.is_non_overlapping_and_dense() ||
      is_blas_compatible_row_major_order(tensor) ||
      is_blas_compatible_column_major_order(tensor)) {
    return at::native::expect_resolved_conj(tensor);  // Returns the tensor with resolved conjugation if necessary
  } else {
    // Clones the tensor to ensure it is contiguous
    return c10::MaybeOwned<Tensor>::owned(
        tensor.clone(at::MemoryFormat::Contiguous));
  }
}

/*
  Prepares a dense matrix for MKL operations with specified memory layout.

  Args:
  * `tensor` - Input tensor to be prepared.
  * `row_major` - Boolean flag indicating if row-major order is preferred.

  Returns:
  * A `MaybeOwned<Tensor>` object wrapping the prepared tensor.
*/
c10::MaybeOwned<Tensor> prepare_dense_matrix_for_mkl(
    const Tensor& tensor,
    bool row_major) {
  // Checks if the tensor is in the desired order based on `row_major` flag
  if (is_blas_compatible_row_major_order(tensor) && row_major) {
    return at::native::expect_resolved_conj(tensor);  // Returns the tensor with resolved conjugation if necessary
  } else {
    if (row_major) {
      // Clones the tensor to ensure it is contiguous
      return c10::MaybeOwned<Tensor>::owned(
          tensor.clone(at::MemoryFormat::Contiguous));
    } else {
      // Clones the tensor in batched column-major format
      return c10::MaybeOwned<Tensor>::owned(cloneBatchedColumnMajor(tensor));
    }
  }
}

/*
  Prepares a dense vector for MKL operations.

  Args:
  * `tensor` - Input tensor to be prepared.

  Returns:
  * A `MaybeOwned<Tensor>` object wrapping the prepared tensor.
*/
c10::MaybeOwned<Tensor> inline prepare_dense_vector_for_mkl(
    const Tensor& tensor) {
  // Checks if the tensor is non-overlapping and dense
  if (tensor.is_non_overlapping_and_dense()) {
    return c10::MaybeOwned<Tensor>::borrowed(tensor);  // Returns a borrowed reference to the tensor
  } else {
    // Clones the tensor to ensure it is contiguous
    return c10::MaybeOwned<Tensor>::owned(
        tensor.clone(at::MemoryFormat::Contiguous));
  }
}

/*
  Converts indices of a sparse CSR tensor to MKL-compatible format inplace.

  Args:
  * `input` - Sparse CSR tensor whose indices need conversion.

  Notes:
  * Uses MKL_ILP64 for 64-bit API or LP64 for 32-bit API.
*/
void inline indices_to_mkl_compatible_inplace(const Tensor& input) {
#ifdef MKL_ILP64
  // If using ILP64 (64-bit API), indices must be of type Long
  static_cast<SparseCsrTensorImpl*>(input.unsafeGetTensorImpl())
      ->set_member_tensors(
          input.crow_indices().to(kLong),
          input.col_indices().to(kLong),
          input.values(),
          input.sizes());
#else
  // If using LP64 (32-bit API), indices must be of type Int
  static_cast<SparseCsrTensorImpl*>(input.unsafeGetTensorImpl())
      ->set_member_tensors(
          input.crow_indices().to(kInt),
          input.col_indices().to(kInt),
          input.values(),
          input.sizes());
#endif
}

#endif // AT_USE_MKL_SPARSE()

} // namespace mkl
} // namespace impl
} // namespace sparse
} // namespace native
} // namespace at
void inline col_indices_and_values_resize_(const Tensor& input, int64_t nnz) {
  // 获取输入 Tensor 的 SparseCsrTensorImpl 实现指针，用于设置成员张量
  static_cast<SparseCsrTensorImpl*>(input.unsafeGetTensorImpl())
      ->set_member_tensors(
          input.crow_indices(),  // 获取输入 Tensor 的行索引
          input.col_indices().resize_({nnz}),  // 调整输入 Tensor 的列索引大小为 nnz
          input.values().resize_({nnz}),  // 调整输入 Tensor 的值大小为 nnz
          input.sizes());  // 保留输入 Tensor 的尺寸信息
}

/*
  Resizes `input` tensor and fills it with the data from MKL.
*/
template <typename scalar_t>
void mkl_result_copy_(const Tensor& input, sparse_matrix_t mkl_desc) {
  // 设置稀疏矩阵的索引基准为零
  sparse_index_base_t indexing = SPARSE_INDEX_BASE_ZERO;
  MKL_INT rows, cols;
  MKL_INT *rows_start = nullptr, *rows_end = nullptr, *columns = nullptr;
  scalar_t* values = nullptr;
  // 导出 MKL 稀疏矩阵的 CSR 格式数据到对应的变量中
  at::mkl::sparse::export_csr(
      mkl_desc,
      &indexing,
      &rows,
      &cols,
      &rows_start,
      &rows_end,
      &columns,
      &values);

  // 使用从 MKL 获得的 nnz 信息调整 input Tensor 的大小
  MKL_INT nnz = rows_end[rows - 1];
  col_indices_and_values_resize_(input, nnz);

  auto crow_indices = input.crow_indices();
  auto col_indices = input.col_indices();
  auto input_values = input.values();

  // 注意：当 nnz 为零时，input_values.data_ptr<scalar_t> 可能为 nullptr，
  // 因此需要检查 nnz 不为零，避免将 nullptr 传递给 std::memcpy。对 crow_indices.data_ptr<MKL_INT>
  // 也采取同样的预防措施，以避免 ASAN 报告问题。

  if (nnz > 0) {
    // 使用 std::memcpy 从 values 复制数据到 input_values
    std::memcpy(
        input_values.mutable_data_ptr<scalar_t>(), values, nnz * sizeof(scalar_t));
    // 使用 std::memcpy 从 columns 复制数据到 col_indices
    std::memcpy(
        col_indices.mutable_data_ptr<MKL_INT>(), columns, nnz * sizeof(MKL_INT));
  }
  if (rows > 0) {
    // 使用 std::memcpy 从 rows_start 复制数据到 crow_indices
    std::memcpy(
        crow_indices.mutable_data_ptr<MKL_INT>(), rows_start, rows * sizeof(MKL_INT));
  }
  // 将 crow_indices 的最后一个元素设为 nnz
  crow_indices.mutable_data_ptr<MKL_INT>()[rows] = nnz;
}
#endif

/*
  Computes a sparse matrix-dense matrix product defined as
  C <- alpha*(A*B) + beta*C

  Args:
  * `A` - Sparse Tensor storing m x k matrix.
  * `B` - Dense Tensor storing k x n matrix.
  * `C` - [in] Dense Tensor storing matrix of size m x n.
          [out] result of the operation.
*/
void addmm_dense_result(
    const Tensor& A,
    const Tensor& B,
    const Scalar& beta,
    const Scalar& alpha,
    const Tensor& C) {
#if !AT_USE_MKL_SPARSE()
  // 如果不使用 MKL 稀疏计算，则抛出错误信息
  TORCH_CHECK(
      false,
      "Calling addmm on a sparse CPU tensor requires Linux platform. ",
      "Please use PyTorch built with MKL on Linux.");
#else
  // 准备将稠密矩阵 C 转换为 MKL 可用格式
  c10::MaybeOwned<Tensor> C_ = prepare_dense_matrix_for_mkl(C);
  // 获取 C 的步幅信息
  IntArrayRef C_strides = C_->strides();
  // 获取 C 的维度数
  auto ndim = C_->dim();
  // 判断 C 是否是行优先存储
  bool is_C_row_major = (C_strides[ndim - 1] == 1);

  // MKL 要求矩阵使用相同的存储布局
  // 准备将稠密矩阵 B 转换为 MKL 可用格式，并根据 C 的存储布局调整
  c10::MaybeOwned<Tensor> B_ = prepare_dense_matrix_for_mkl(B, is_C_row_major);
  // 获取 B 的步幅信息
  IntArrayRef B_strides = B_->strides();
  // 判断 B 是否是行优先存储
  bool is_B_row_major = (B_strides[ndim - 1] == 1);

  // 断言 C 和 B 必须具有相同的存储布局
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(!(is_C_row_major ^ is_B_row_major));

  // 确定稀疏矩阵乘法的存储布局
  auto order =
      is_C_row_major ? SPARSE_LAYOUT_ROW_MAJOR : SPARSE_LAYOUT_COLUMN_MAJOR;
  // 获取 C 的列步幅或行步幅，根据其存储布局决定
  auto ldc = is_C_row_major ? C_strides[ndim - 2] : C_strides[ndim - 1];
  // 获取 B 的列步幅或行步幅，根据其存储布局决定
  auto ldb = is_B_row_major ? B_strides[ndim - 2] : B_strides[ndim - 1];
  // 获取 C 的列数，并转换为 MKL 使用的整型类型
  auto columns_C = mkl_int_cast(C.size(-1), "columns_C");

  // 定义稀疏矩阵描述符
  matrix_descr descrA;
  descrA.type = SPARSE_MATRIX_TYPE_GENERAL;

  // 根据 C 的数据类型进行分发处理
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      C.scalar_type(), "addmm_out_sparse_csr_impl_mkl", [&] {
        // 将 beta 和 alpha 转换为当前数据类型的标量
        auto beta_ = beta.to<scalar_t>();
        auto alpha_ = alpha.to<scalar_t>();

        // 创建 MKL 稀疏矩阵描述符
        auto mkl_sparse_mat =
            at::mkl::sparse::MklSparseCsrDescriptor<scalar_t>(A);
        // 调用 MKL 的稀疏矩阵乘法函数 mm
        at::mkl::sparse::mm<scalar_t>(
            SPARSE_OPERATION_NON_TRANSPOSE,
            alpha_,
            mkl_sparse_mat.descriptor(),
            descrA,
            order,
            B_->data_ptr<scalar_t>(),
            columns_C,
            ldb,
            beta_,
            C_->data_ptr<scalar_t>(),
            ldc);
      });

  // 如果 C 和 C_ 不是同一个 Tensor，则将 C_ 的数据拷贝到 C 中
  if (!C.is_same(*C_)) {
    C.copy_(*C_);
  }
#endif
}

/*
  Computes a sparse matrix-sparse matrix product with dense result defined as
  C <- alpha*(A*B) + beta*C

  Args:
  * `A` - Sparse Tensor storing m x k matrix.
  * `B` - Sparse Tensor storing k x n matrix.
  * `C` - [in] Dense Tensor storing matrix of size m x n.
          [out] result of the operation.
*/
void addmm_sparse_input_dense_result(
    const Tensor& A,
    const Tensor& B,
    const Scalar& beta,
    const Scalar& alpha,
    const Tensor& C) {
#if !AT_USE_MKL_SPARSE()
  // 如果未使用 MKL 的稀疏库，抛出错误信息
  TORCH_CHECK(
      false,
      "Calling addmm on a sparse CPU tensor requires Linux platform. ",
      "Please use PyTorch built with MKL on Linux.");
#endif
}
#else
  // 如果未定义 AT_USE_MKL_SPARSE，则以下代码块被执行
  // MKL 函数计算 C <- A*B
  // 因此我们需要一个临时矩阵来存储结果
  // 然后将其加到 C 上
  auto C_ = at::empty(C.sizes(), C.options());
  // 设置稀疏矩阵布局为行主序
  auto order = SPARSE_LAYOUT_ROW_MAJOR;
  // 计算 C 的列跨度
  auto ldc = C_.stride(-2);

  // 根据 C 的数据类型进行派发，执行稀疏矩阵乘法
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      C.scalar_type(), "addmm_sparse_input_dense_result", [&] {
        // 创建 A 和 B 的 MKL 稀疏 CSR 描述符
        auto mkl_A = at::mkl::sparse::MklSparseCsrDescriptor<scalar_t>(A);
        auto mkl_B = at::mkl::sparse::MklSparseCsrDescriptor<scalar_t>(B);
        // 调用 MKL 的稀疏矩阵乘法
        at::mkl::sparse::spmmd<scalar_t>(
            SPARSE_OPERATION_NON_TRANSPOSE,
            mkl_A.descriptor(),
            mkl_B.descriptor(),
            order,
            C_.data_ptr<scalar_t>(),
            ldc);
      });

  // 如果 beta 为零，则不应将 NaN 和 Inf 传播到结果中
  if (beta.toComplexDouble() == 0.) {
    // 将 C 的值置零
    C.zero_();
  } else {
    // 将 C 乘以 beta
    C.mul_(beta);
  }
  // 将 C 加上临时矩阵 C_
  C.add_(C_, alpha);
#endif
}

/*
  计算稀疏矩阵-稀疏矩阵乘积，定义为 C <- alpha*(A*B) + beta*C

  参数:
  * `mat1` - 存储 m x k 矩阵 A 的稀疏 CSR 张量
  * `mat2` - 存储 k x n 矩阵 B 的稀疏 CSR 张量
  * `beta` - 标量 beta
  * `alpha` - 标量 alpha
  * `result` - [in] 存储大小为 m x n 的矩阵 C 的稀疏 CSR 张量。
               [out] 操作的结果。
*/
void addmm_sparse_result(
    const Tensor& mat1,
    const Tensor& mat2,
    const Scalar& beta,
    const Scalar& alpha,
    const Tensor& result) {
#if !AT_USE_MKL_SPARSE()
  // 如果未启用 MKL 稀疏支持，则抛出错误信息
  TORCH_CHECK(
      false,
      "Calling add on a sparse CPU tensor requires Linux platform. ",
      "Please use PyTorch built with MKL on Linux.");
#else
  // 计算 beta*result，因为 MKL 不会处理这一步
  // 如果 beta 为零，则不应将 NaN 和 Inf 传播到结果中
  if (beta.toComplexDouble() == 0.) {
    // 将 result 的值置零
    result.values().zero_();
  } else {
    // 将 result 乘以 beta
    result.values().mul_(beta);
  }

  // 如果 mat1 或 mat2 是空矩阵，则直接返回，因为 MKL 不能处理空矩阵
  if (mat1._nnz() == 0 || mat2._nnz() == 0) {
    return;
  }

  // MKL 没有接口可以一次计算 alpha*(A*B) + beta*C
  // 创建一个与 result 大小相同的全零张量
  Tensor mat1_mat2 = at::zeros(result.sizes(), result.options());
  // 将其转换为 MKL 兼容格式
  indices_to_mkl_compatible_inplace(mat1_mat2);

  // 根据 result 的数据类型进行派发，执行 MKL 的稀疏矩阵乘法
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      result.scalar_type(), "addmm_out_sparse_csr_impl_mkl_sparse", [&] {
        auto mkl_sparse_mat1 =
            at::mkl::sparse::MklSparseCsrDescriptor<scalar_t>(mat1);
        auto mkl_sparse_mat2 =
            at::mkl::sparse::MklSparseCsrDescriptor<scalar_t>(mat2);
        auto mkl_result = at::mkl::sparse::MklSparseCsrDescriptor<scalar_t>();
        auto result_desc = mkl_result.descriptor();

        // 调用 MKL 的稀疏矩阵乘法
        TORCH_MKLSPARSE_CHECK(mkl_sparse_spmm(
            SPARSE_OPERATION_NON_TRANSPOSE,
            mkl_sparse_mat1.descriptor(),
            mkl_sparse_mat2.descriptor(),
            &result_desc));

        // 从 MKL 复制数据，否则计算结果会随着 `mkl_result` 被销毁而丢失
        mkl_result_copy_<scalar_t>(mat1_mat2, result_desc);
      });

  // 将 mat1_mat2 加到 result 上，乘以 alpha
  result.add_(mat1_mat2, alpha);
#endif
}
} // anonymous namespace


/*
  计算定义为矩阵乘积的操作：
  C <- alpha*(A*B) + beta*C

  Args:
  * `mat1` - 存储 m x k 矩阵 A 的张量。
  * `mat2` - 存储 k x n 矩阵 B 的张量。
  * `beta` - 标量，用于乘法的缩放因子。
  * `alpha` - 标量，用于乘法的缩放因子。
  * `result` - [in] 大小为 m x n 的矩阵 C 的张量。
               [out] 操作结果。

  该函数实现了稀疏 CSR 格式的矩阵乘法。
*/
void addmm_out_sparse_csr(
    const Tensor& mat1,
    const Tensor& mat2,
    const Scalar& beta,
    const Scalar& alpha,
    const Tensor& result) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      mat1.dim() == 2 && mat2.dim() == 2 && result.dim() == 2);
  TORCH_INTERNAL_ASSERT(
      !((mat1.layout() == kStrided) && (mat2.layout() == kStrided) &&
        (result.layout() == kStrided)),
      "Expected at least one sparse input");

  // 对布局进行检查，按照 mat1, mat2, result 的顺序进行嵌套检查
  // 条件按照 strided, csr, csc, bsr, bsc 的顺序进行排列
  // 有效组合会直接返回，无效组合会继续下一个检查，最终由 TORCH 检查并生成错误信息
  if (mat1.layout() == kStrided) {
    if (mat2.layout() == kSparseCsr) {
      if (result.layout() == kStrided) {
        // 如果结果布局为 strided，则调用相应的稠密结果处理函数
        return addmm_dense_result(
            mat2.transpose(0, 1).to_sparse_csr(),
            mat1.transpose(0, 1),
            beta,
            alpha,
            result.transpose(0, 1));
      }
    }
    if (mat2.layout() == kSparseCsc) {
      if (result.layout() == kStrided) {
        // 如果结果布局为 strided，则调用相应的稠密结果处理函数
        return addmm_dense_result(
            mat2.transpose(-2, -1),
            mat1.transpose(-2, -1),
            beta,
            alpha,
            result.transpose(-2, -1));
      }
    }
    if (mat2.layout() == kSparseBsc) {
      if (result.layout() == kStrided) {
        // 如果结果布局为 strided，则调用相应的稠密结果处理函数
        return addmm_dense_result(
            mat2.transpose(-2, -1),
            mat1.transpose(-2, -1),
            beta,
            alpha,
            result.transpose(-2, -1));
      }
    }
  }
  if (mat1.layout() == kSparseCsr) {
    if (mat2.layout() == kStrided) {
      if (result.layout() == kStrided) {
        // 调用稀疏 CSR 输入的稠密结果处理函数
        return addmm_dense_result(mat1, mat2, beta, alpha, result);
      }
    }
    if (mat2.layout() == kSparseCsr) {
      if (result.layout() == kStrided) {
        // 调用稀疏 CSR 输入的稀疏 CSR 结果处理函数
        return addmm_sparse_input_dense_result(mat1, mat2, beta, alpha, result);
      }
      if (result.layout() == kSparseCsr) {
        // 直接调用稀疏 CSR 结果处理函数
        return addmm_sparse_result(mat1, mat2, beta, alpha, result);
      }
    }
    if (mat2.layout() == kSparseCsc) {
      if (result.layout() == kStrided) {
        // 如果结果布局为 strided，则调用稀疏 CSR 输入的稠密结果处理函数
        return addmm_sparse_input_dense_result(
            mat1, mat2.to_sparse_csr(), beta, alpha, result);
      }
      if (result.layout() == kSparseCsr) {
        // 直接调用稀疏 CSR 结果处理函数
        return addmm_sparse_result(
            mat1, mat2.to_sparse_csr(), beta, alpha, result);
      }
    }
  }
  // 检查第一个输入矩阵是否为稀疏的列压缩格式（CSC）
  if (mat1.layout() == kSparseCsc) {
    // 如果第二个输入矩阵是步进布局（strided）
    if (mat2.layout() == kStrided) {
      // 如果结果矩阵也是步进布局
      if (result.layout() == kStrided) {
        // TODO: 使用本地的CSC支持避免CSC到CSR的转换
        return addmm_dense_result(
            mat1.to_sparse_csr(), mat2, beta, alpha, result);
      }
    }
    // 如果第二个输入矩阵是稀疏的行压缩格式（CSR）
    if (mat2.layout() == kSparseCsr) {
      // 如果结果矩阵也是稀疏的行压缩格式（CSR）
      if (result.layout() == kSparseCsr) {
        // TODO: 使用本地的CSC支持避免CSC到CSR的转换
        return addmm_sparse_result(
            mat1.to_sparse_csr(), mat2, beta, alpha, result);
      }
    }
    // 如果第二个输入矩阵是稀疏的列压缩格式（CSC）
    if (mat2.layout() == kSparseCsc) {
      // 如果结果矩阵是步进布局
      if (result.layout() == kStrided) {
        // 返回使用稀疏输入矩阵和稠密结果矩阵进行的加法乘法运算结果
        return addmm_sparse_input_dense_result(
            mat2.transpose(-2, -1),
            mat1.transpose(-2, -1),
            beta,
            alpha,
            result.transpose(-2, -1));
      }
      // 如果结果矩阵是稀疏的行压缩格式（CSR）
      if (result.layout() == kSparseCsr) {
        // TODO: 避免CSC到CSR的转换
        return addmm_sparse_result(
            mat1.to_sparse_csr(), mat2.to_sparse_csr(), beta, alpha, result);
      }
      // 如果结果矩阵是稀疏的列压缩格式（CSC）
      if (result.layout() == kSparseCsc) {
        // 返回使用稀疏输入矩阵进行的加法乘法运算结果
        return addmm_sparse_result(
            mat2.transpose(-2, -1),
            mat1.transpose(-2, -1),
            beta,
            alpha,
            result.transpose(-2, -1));
      }
    }
  }
  // 如果第一个输入矩阵是稀疏的分块稠密行格式（BSR）
  if (mat1.layout() == kSparseBsr) {
    // 如果第二个输入矩阵是步进布局
    if (mat2.layout() == kStrided) {
      // 如果结果矩阵也是步进布局
      if (result.layout() == kStrided) {
        // 返回使用稀疏分块稠密行格式的输入矩阵和步进布局的第二个输入矩阵进行的加法乘法运算结果
        return addmm_dense_result(mat1, mat2, beta, alpha, result);
      }
    }
  }
  // 抛出异常，提示在CPU上不支持所给的矩阵布局进行加法乘法运算
  TORCH_CHECK(
      false,
      "addmm: computation on CPU is not implemented for ",
      result.layout(),
      " + ",
      mat1.layout(),
      " @ ",
      mat2.layout());
/*
  Computes a sparse matrix-dense vector product defined as
  y <- alpha*op(A)*x + beta*y

  Args:
  * `mat` - Tensor storing sparse m x n matrix A.
  * `vec` - Tensor storing dense vector x of size n.
  * `result` - [in] Tensor storing dense vector y of size m.
               [out] result of the operation.
*/
void addmv_out_sparse_csr(
    const Tensor& mat,
    const Tensor& vec,
    const Scalar& beta,
    const Scalar& alpha,
    const Tensor& result) {
#if !AT_USE_MKL_SPARSE()
  // 检查是否在非 Linux 平台上调用稀疏张量的 addmv 操作
  TORCH_CHECK(
      false,
      "Calling addmv on a sparse CPU tensor requires Linux platform. ",
      "Please use PyTorch built with MKL on Linux.");
#else
  // 准备稠密向量用于 MKL
  c10::MaybeOwned<Tensor> result_ = prepare_dense_vector_for_mkl(result);
  c10::MaybeOwned<Tensor> vec_ = prepare_dense_vector_for_mkl(vec);

  // 指定稀疏操作和稀疏矩阵的描述符
  sparse_operation_t opA = SPARSE_OPERATION_NON_TRANSPOSE;
  matrix_descr descrA;
  descrA.type = SPARSE_MATRIX_TYPE_GENERAL;

  // 根据结果张量的数据类型分发处理函数
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      result.scalar_type(), "addmv_out_sparse_csr_impl_mkl", [&] {
        auto beta_ = beta.to<scalar_t>();
        auto alpha_ = alpha.to<scalar_t>();

        // 创建 MKL 稀疏矩阵描述符
        auto mkl_sparse_mat =
            at::mkl::sparse::MklSparseCsrDescriptor<scalar_t>(mat);

        // 调用 MKL 的稀疏-稠密矩阵乘法
        at::mkl::sparse::mv<scalar_t>(
            opA,
            alpha_,
            mkl_sparse_mat.descriptor(),
            descrA,
            vec_->data_ptr<scalar_t>(),
            beta_,
            result_->data_ptr<scalar_t>());
      });

  // 如果结果张量不是原始张量，则复制结果
  if (!result.is_same(*result_)) {
    result.copy_(*result_);
  }
#endif
}

void add_out_sparse_csr(
    const Tensor& mat1,
    const Tensor& mat2,
    const Scalar& alpha,
    const Tensor& result) {
#if !AT_USE_MKL_SPARSE()
  // 检查是否在非 Linux 平台上调用稀疏张量的 add 操作
  TORCH_CHECK(
      false,
      "Calling add on a sparse CPU tensor requires Linux platform. ",
      "Please use PyTorch built with MKL on Linux.");
#else

  // MKL 不适用于空矩阵
  if (mat2._nnz() == 0) {
    // 调整结果张量大小并复制 mat1 到结果
    col_indices_and_values_resize_(result, mat1._nnz());
    result.copy_(mat1);
    return;
  } else if (mat1._nnz() == 0) {
    // 调整结果张量大小并复制 mat2 到结果，然后乘以 alpha
    col_indices_and_values_resize_(result, mat2._nnz());
    result.copy_(mat2);
    result.values().mul_(alpha);

    result.values().mul_(alpha);
  }
#endif
}
    return;
  }

  // Modify `result` tensor in-place to swap indices tensors with 32-bit (or
  // 64-bit) variants
  // 确定输出的稀疏张量的索引数据类型，使用 promoteTypes 函数提升输入张量的索引数据类型
  const auto output_indices_dtype = promoteTypes(mat1.crow_indices().scalar_type(), mat2.crow_indices().scalar_type());
  // 备份 `result` 张量的行索引和列索引
  auto result_crow_indices_backup = result.crow_indices();
  auto result_col_indices_backup = result.col_indices();
  // 将 `result` 张量的索引转换为 MKL 兼容格式，就地操作
  indices_to_mkl_compatible_inplace(result);
  // 定义稀疏操作类型为非转置
  sparse_operation_t opA = SPARSE_OPERATION_NON_TRANSPOSE;

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      result.scalar_type(), "add_out_sparse_csr_impl_mkl", [&] {
        auto alpha_ = alpha.to<scalar_t>();

        // 创建 MKL 稀疏 CSR 描述符，用于 mat1 和 mat2
        auto mkl_mat1 = at::mkl::sparse::MklSparseCsrDescriptor<scalar_t>(mat1);
        auto mkl_mat2 = at::mkl::sparse::MklSparseCsrDescriptor<scalar_t>(mat2);
        auto mkl_result = at::mkl::sparse::MklSparseCsrDescriptor<scalar_t>();

        // 注意，mat1 和 mat2 参数的顺序被交换，因为 MKL 计算 alpha*mat1 + mat2，
        // 而 PyTorch 需要 mat1 + alpha*mat2
        auto result_desc = mkl_result.descriptor();
        // 调用 MKL 函数计算稀疏矩阵相加
        at::mkl::sparse::add<scalar_t>(
            opA,
            mkl_mat2.descriptor(),
            alpha_,
            mkl_mat1.descriptor(),
            &result_desc);

        // 现在将数据从 `result_desc` 复制到 `result`
        mkl_result_copy_<scalar_t>(result, result_desc);
      });

  // 如果输出的索引数据类型为长整型
  if (output_indices_dtype == at::kLong) {
    const auto res_nnz = result._nnz();
    // 获取稀疏张量的实现，并设置成员张量，恢复原始的行索引和调整列索引的大小
    static_cast<SparseCsrTensorImpl*>(result.unsafeGetTensorImpl())->set_member_tensors(
        result_crow_indices_backup.copy_(result.crow_indices()),
        result_col_indices_backup.resize_({res_nnz}).copy_(result.col_indices()),
        result.values(),
        result.sizes());
  }
  // 如果未定义 AT_USE_MKL_SPARSE 宏，则抛出错误，要求在 Linux 平台上使用支持 MKL 的 PyTorch 版本
  TORCH_CHECK(
      false,
      "Calling triangular_solve on a sparse CPU tensor requires Linux platform. ",
      "Please use PyTorch built with MKL on Linux.");
#else
  // 检查输入张量 B、X 和稀疏张量 A_ 是否为空，如果有任一为空，则无法进行求解，将结果张量 X 填充为 NaN 并返回
  if (B.numel() == 0 || X.numel() == 0 || A_._nnz() == 0) {
    X.fill_(NAN);
    return;
  }

  // 定义 lambda 函数 materialize_diagonal_indices 用于生成扩展后的对角线索引
  const auto materialize_diagonal_indices = [](const Tensor& t) -> Tensor {
    // 获取张量 t 的最后一个维度大小作为 n
    const auto n = t.size(-1);
    // 获取压缩后的稀疏张量索引
    const auto compressed_indices = std::get<0>(at::sparse_csr::getCompressedPlainIndices(t));
    // 创建对角线索引，以稀疏张量索引的选项为基础，创建大小为 n 的张量并扩展为二维
    const auto diag_indices = at::arange(n, compressed_indices.options()).unsqueeze(0).expand({2, n});
    // 创建大小为 1 的全零张量并扩展为大小为 n 的对角线值张量
    const auto diag_values = at::zeros({1}, t.values().options()).expand({n});

    // 将稀疏张量转换为 COO 格式
    const auto t_coo = t.to_sparse();
    // 将扩展后的对角线索引和对角线值张量连接到稀疏张量的索引和值后面，维度为 -1
    const auto expanded_indices = at::cat({t_coo._indices(), diag_indices}, /*dim=*/-1);
    const auto expanded_values = at::cat({t_coo._values(), diag_values}, /*dim=*/0);

    // 使用扩展后的索引和值创建 COO 格式的稀疏张量
    const auto t_expanded_coo = at::sparse_coo_tensor(expanded_indices, expanded_values, t_coo.sizes(), t_coo.options());


注释部分已经解释了每一行代码的作用和功能。
    return t_expanded_coo.to_sparse(t.layout());
  };

  // MKL has a bug for inputs with unmaterialized diagonal indices.
  // See https://github.com/pytorch/pytorch/issues/88890 and
  // the comments within.
  // 使用条件表达式根据 `unitriangular` 变量选择是否材料化对角线索引
  const auto A = unitriangular ? materialize_diagonal_indices(A_) : A_;

  // 准备 `X` 稠密矩阵以供 MKL 使用
  c10::MaybeOwned<Tensor> X_ = prepare_dense_matrix_for_mkl(X);
  // 获取 `X` 矩阵的步长信息
  IntArrayRef X_strides = X_->strides();
  auto ndim = X_->dim();
  // 检查 `X` 是否是行优先存储
  bool is_X_row_major = (ndim > 1) ? (X_strides[ndim - 1] == 1) : true;

  // MKL 要求矩阵使用相同的存储布局
  // 根据 `is_X_row_major` 准备 `B` 稠密矩阵以供 MKL 使用
  c10::MaybeOwned<Tensor> B_ = prepare_dense_matrix_for_mkl(B, is_X_row_major);

  // 设置稀疏矩阵操作类型为转置或非转置
  sparse_operation_t opA = transpose ? SPARSE_OPERATION_TRANSPOSE : SPARSE_OPERATION_NON_TRANSPOSE;
  // 设置稀疏矩阵描述符
  matrix_descr descrA;
  descrA.type = SPARSE_MATRIX_TYPE_TRIANGULAR;
  descrA.mode = upper ? SPARSE_FILL_MODE_UPPER : SPARSE_FILL_MODE_LOWER;
  descrA.diag = unitriangular ? SPARSE_DIAG_UNIT : SPARSE_DIAG_NON_UNIT;

  // 根据 `X` 的数据类型进行分发，执行稀疏求解操作
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      X.scalar_type(), "triangular_solve_out_sparse_csr_impl_mkl", [&] {
        // 创建 MKL 稀疏矩阵描述符
        auto mkl_sparse_mat =
            at::mkl::sparse::MklSparseCsrDescriptor<scalar_t>(A);
        scalar_t alpha = 1;

        // 检查 `B` 是否为列向量
        if (B.size(-1) == 1) {
          // 调用 MKL 的向前/向后替代解算法
          at::mkl::sparse::trsv<scalar_t>(
              opA,
              alpha,
              mkl_sparse_mat.descriptor(),
              descrA,
              B_->data_ptr<scalar_t>(),
              X_->data_ptr<scalar_t>());
        } else {
          // 获取 `B` 矩阵的步长信息和存储布局
          IntArrayRef B_strides = B_->strides();
          bool is_B_row_major = (B_strides[ndim - 1] == 1);
          // 断言 `X` 和 `B` 使用相同的存储布局
          TORCH_INTERNAL_ASSERT_DEBUG_ONLY(!(is_X_row_major ^ is_B_row_major));

          // 确定稀疏求解算法的存储顺序、右手边的数量以及 `X` 和 `B` 的步长
          auto order = is_X_row_major ? SPARSE_LAYOUT_ROW_MAJOR : SPARSE_LAYOUT_COLUMN_MAJOR;
          auto nrhs = mkl_int_cast(B.size(-1), "nrhs");
          auto ldx = is_X_row_major ? X_strides[ndim - 2] : X_strides[ndim - 1];
          auto ldb = is_B_row_major ? B_strides[ndim - 2] : B_strides[ndim - 1];
          // 调用 MKL 的稀疏矩阵左乘解算法
          at::mkl::sparse::trsm<scalar_t>(
              opA,
              alpha,
              mkl_sparse_mat.descriptor(),
              descrA,
              order,
              B_->data_ptr<scalar_t>(),
              nrhs,
              ldb,
              X_->data_ptr<scalar_t>(),
              ldx);
        }
      });

  // 如果 `X` 和 `X_` 不是同一个对象，则将计算结果拷贝到 `X` 中
  if (!X.is_same(*X_)) {
    X.copy_(*X_);
  }
#endif



}



} // namespace mkl



} // namespace impl



} // namespace sparse



} // namespace native



} // namespace at


这段代码是C++中的命名空间闭合语句。每一行的作用如下：

1. `#endif`: 结束条件编译指令的作用域。
2. `}`: 结束当前的函数或代码块。
3. `} // namespace mkl`: 结束命名空间 `mkl` 的定义。
4. `} // namespace impl`: 结束命名空间 `impl` 的定义。
5. `} // namespace sparse`: 结束命名空间 `sparse` 的定义。
6. `} // namespace native`: 结束命名空间 `native` 的定义。
7. `} // namespace at`: 结束命名空间 `at` 的定义。

这些语句用于定义命名空间的层次结构，确保在不同的命名空间中声明的变量、函数和类名不会与其他命名空间中的冲突。
```