# `.\pytorch\aten\src\ATen\native\sparse\cuda\SparseBlasImpl.cpp`

```py
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/OpMathType.h>
#include <ATen/cuda/CUDADataType.h>
#include <ATen/cuda/CUDASparse.h>
#include <ATen/cuda/CUDASparseBlas.h>
#include <ATen/cuda/CUDASparseDescriptors.h>
#include <ATen/native/LinearAlgebraUtils.h>
#include <ATen/native/cuda/MiscUtils.h>
#include <ATen/native/sparse/SparseBlasImpl.h>
#include <ATen/native/sparse/cuda/SparseBlasImpl.h>
#include <ATen/native/sparse/cuda/SparseBlasLegacy.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_sparse_csr_tensor_unsafe_native.h>
#include <ATen/ops/empty_strided.h>
#endif

#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/util/MaybeOwned.h>

namespace at::native::sparse::impl::cuda {

namespace {

// 准备列主序矩阵以供 cuSPARSE 使用
c10::MaybeOwned<Tensor> prepare_column_major_matrix_for_cusparse(
    const Tensor& tensor) {
  // 如果输入张量已经是 BLAS 兼容的列主序，则直接返回其共轭视图
  if (is_blas_compatible_column_major_order(tensor)) {
    return at::native::expect_resolved_conj(tensor);
  } else {
    // 否则，克隆成批次列主序张量并返回
    return c10::MaybeOwned<Tensor>::owned(cloneBatchedColumnMajor(tensor));
  }
}

// 准备稠密矩阵以供 cuSPARSE 使用
c10::MaybeOwned<Tensor> inline prepare_dense_matrix_for_cusparse(
    const Tensor& tensor) {
#if defined(USE_ROCM)
  // 对于 ROCm 平台，CUDA < 11.0 不支持行主序布局，因此在这种情况下返回列主序
  return prepare_column_major_matrix_for_cusparse(tensor);
#else
  // 对于 CUDA 平台，如果张量是 BLAS 兼容的行主序或列主序，则返回其共轭视图
  if (is_blas_compatible_row_major_order(tensor) ||
      is_blas_compatible_column_major_order(tensor)) {
    return at::native::expect_resolved_conj(tensor);
  } else {
    // 否则，克隆成连续存储格式的张量并返回
    return c10::MaybeOwned<Tensor>::owned(
        tensor.clone(at::MemoryFormat::Contiguous));
  }
#endif
}

// 复制张量按指定步长
Tensor copy_strided(const Tensor& tensor, IntArrayRef strides) {
  // 根据指定的步长创建一个空张量
  Tensor result = at::empty_strided(tensor.sizes(), strides, tensor.options());
  // 将输入张量的数据复制到新创建的张量中
  result.copy_(tensor);
  return result;
}

// 根据指定步长准备稠密矩阵以供 cuSPARSE 使用
c10::MaybeOwned<Tensor> prepare_dense_matrix_for_cusparse(
    const Tensor& tensor,
    IntArrayRef strides) {
  // 如果输入张量的步长与指定步长相同，则返回 borrowed 类型的张量
  if (tensor.strides().equals(strides)) {
    return c10::MaybeOwned<Tensor>::borrowed(tensor);
  } else {
    // 否则，复制成指定步长的新张量并返回
    return c10::MaybeOwned<Tensor>::owned(copy_strided(tensor, strides));
  }
}

// 为不支持新 cuSPARSE 通用 API 的旧 CUDA Toolkit 版本提供的函数
void addmm_out_legacy(
    const at::sparse_csr::SparseCsrTensor& mat1,
    const Tensor& mat2,
    const Scalar& beta,
    const Scalar& alpha,
    // 断言 mat1 是稀疏 CSR 格式的张量（仅用于调试模式）
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(mat1.is_sparse_csr());
    
    // 获取 mat1 的非零元素个数
    auto nnz = mat1._nnz();
    
    // 获取 mat1 的行数 m
    auto m = mat1.size(0);
    
    // 获取 mat1 的列数 k
    auto k = mat1.size(1);
    
    // 获取 mat2 的列数 n
    auto n = mat2.size(1);
    
    // 将 mat1 的行偏移数组转换为 kInt 类型
    auto crow_indices = mat1.crow_indices().to(kInt);
    
    // 将 mat1 的列索引数组转换为 kInt 类型
    auto col_indices = mat1.col_indices().to(kInt);
    
    // 获取 mat1 的值数组
    auto values = mat1.values();
    
    // 对 mat2 进行共轭处理（如果有必要）
    auto mat2_ = at::native::expect_resolved_conj(mat2);
    
    // 对 result 进行共轭处理（如果有必要）
    auto result_ = at::native::expect_resolved_conj(result);
    
    // 调用 CUDA 的稀疏矩阵与稠密矩阵相乘的工作函数，计算结果存储在 result 中
    at::native::s_addmm_out_csr_sparse_dense_cuda_worker(nnz, m, n, k, result, beta, *result_, alpha, crow_indices, col_indices, values, *mat2_);
    
    // 如果 result 和 *result_ 不是同一个张量，则将 *result_ 的值拷贝到 result 中
    if (!result.is_same(*result_)) {
        result.copy_(*result_);
    }
}

// 将稠密向量准备为适用于 cusparse 的形式
c10::MaybeOwned<Tensor> inline prepare_dense_vector_for_cusparse(
    const Tensor& tensor) {
  // 如果张量是非重叠且稠密的，则直接借用该张量
  if (tensor.is_non_overlapping_and_dense()) {
    return c10::MaybeOwned<Tensor>::borrowed(tensor);
  } else {
    // 否则，克隆张量并保证是连续内存格式
    return c10::MaybeOwned<Tensor>::owned(
        tensor.clone(at::MemoryFormat::Contiguous));
  }
}

// 将输入张量中的索引转换为 32 位整数格式（原地操作）
void inline indices_to_32_bit_inplace(const Tensor& input) {
  // 强制转换为稀疏 CSR 张量实现类，并设置其成员张量
  static_cast<SparseCsrTensorImpl*>(input.unsafeGetTensorImpl())->set_member_tensors(
      input.crow_indices().to(kInt),
      input.col_indices().to(kInt),
      input.values(),
      input.sizes());
}

// 调整输入张量中列索引和值的大小（原地操作）
void inline col_indices_and_values_resize_(const Tensor& input, int64_t nnz) {
  // 强制转换为稀疏 CSR 张量实现类，并设置其成员张量
  static_cast<SparseCsrTensorImpl*>(input.unsafeGetTensorImpl())->set_member_tensors(
      input.crow_indices(),
      input.col_indices().resize_({nnz}),
      input.values().resize_({nnz}),
      input.sizes());
}

// 处理 cuSPARSE 版本低于 11.7.3 的同步问题
void inline bsrsv2_bsrsm2_may_need_to_sync() {
#if defined(CUSPARSE_VERSION) && CUSPARSE_VERSION < 11703
  // cuSPARSE 的 bsrsv2 和 bsrsm2 存在同步问题，可能导致 CUDA <= 11.6.x 的非法内存访问
  // 参见 https://github.com/pytorch/pytorch/issues/71297
  ::c10::cuda::device_synchronize();
#endif
  // 否则不做任何操作！
}

// 解决稀疏块三角求解的向量形式
void block_sparse_triangular_solve_vec(
    const at::sparse_csr::SparseCsrTensor& A,
    const Tensor& B,
    const Tensor& X,
    bool upper,
    bool transpose,
    bool unitriangular) {
#if !AT_USE_HIPSPARSE_TRIANGULAR_SOLVE()
  // 使用稀疏 BSR 布局进行三角求解需要编译 PyTorch 支持 ROCm 4.5.0+
  TORCH_CHECK(
      false,
      "Calling triangular solver with block sparse GPU tensors requires compiling ",
      "PyTorch with ROCm 4.5.0+. ",
      "Please use PyTorch built with newer ROCm version.");
#else
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(A.layout() == kSparseBsr);
  // values 应该是稀疏矩阵的块
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(A.values().dim() == 3);
  // 块应该是方形的
  TORCH_INTERNAL_ASSERT(A.values().size(2) == A.values().size(1));
  // cuSPARSE 仅支持大于 1 的块大小
  TORCH_INTERNAL_ASSERT(A.values().size(-1) > 1);
  // 块应该按行或列主序排列
  TORCH_INTERNAL_ASSERT(
      A.values().is_contiguous() ||
      A.values().transpose(-2, -1).is_contiguous());

  // cuSPARSE 不能处理空的稀疏矩阵
  if (A._nnz() == 0) {
    X.fill_(NAN);
    X.copy_(*X_);
  }
#endif
}

// 解决稀疏块三角求解的矩阵形式
void block_sparse_triangular_solve_mat(
    const at::sparse_csr::SparseCsrTensor& A,
    const Tensor& B,
    const Tensor& X,
    bool upper,
    bool transpose,
    bool unitriangular) {
#if !AT_USE_HIPSPARSE_TRIANGULAR_SOLVE()
  // 使用稀疏 BSR 布局进行三角求解需要编译 PyTorch 支持 ROCm 4.5.0+
  TORCH_CHECK(
      false,
      "Calling triangular solver with block sparse GPU tensors requires compiling ",
      "PyTorch with ROCm 4.5.0+. ",
      "Please use PyTorch built with newer ROCm version.");
#endif
#else
  // 断言：A 的布局应为 kSparseBsr
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(A.layout() == kSparseBsr);
  // 注释：values 应该是稀疏矩阵的块
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(A.values().dim() == 3);
  // 注释：块应该是正方形的
  TORCH_INTERNAL_ASSERT(A.values().size(2) == A.values().size(1));
  // 注释：仅支持大小大于 1 的块在 cuSPARSE 中
  TORCH_INTERNAL_ASSERT(A.values().size(-1) > 1);
  // 注释：块应按行或列主序排列
  TORCH_INTERNAL_ASSERT(
      A.values().is_contiguous() ||
      A.values().transpose(-2, -1).is_contiguous());

  // 注释：cuSPARSE 无法处理空稀疏矩阵
  if (A._nnz() == 0) {
    // 注释：将 X 填充为 NaN
    X.fill_(NAN);
    // 注释：将 X_ 的内容复制到 X 中
    X.copy_(*X_);
  }
#endif
}

void block_sparse_mv(
    const at::sparse_csr::SparseCsrTensor& mat,
    const Tensor& vec,
    const Scalar& beta,
    const Scalar& alpha,
    const Tensor& result) {
```  

// 对 result 参数的引用，表示该函数操作的结果将会存储在这个 Tensor 中
TORCH_INTERNAL_ASSERT_DEBUG_ONLY(mat.layout() == kSparseBsr);
// 断言 mat 的布局是稀疏块压缩稀疏矩阵 (Sparse Block Sparse Row)
TORCH_INTERNAL_ASSERT_DEBUG_ONLY(mat.values().dim() == 3);
// 断言 mat 的 values 张量维度为 3，即预期其为稀疏矩阵的块
TORCH_INTERNAL_ASSERT(mat.values().size(2) == mat.values().size(1));
// 断言块是方形的，即列数等于行数
TORCH_INTERNAL_ASSERT(mat.values().size(-1) > 1);
// 断言每个块的尺寸大于 1，在 cuSPARSE 中仅支持这种情况
TORCH_INTERNAL_ASSERT(
    mat.values().is_contiguous() ||
    mat.values().transpose(-2, -1).is_contiguous());
// 断言块按行或列的顺序存储，要么是连续的，要么进行了转置后是连续的

const cusparseDirection_t block_layout = mat.values().is_contiguous()
    ? CUSPARSE_DIRECTION_ROW
    : CUSPARSE_DIRECTION_COLUMN;
// 根据 values 张量是否连续，确定块的布局顺序是行主还是列主

c10::MaybeOwned<Tensor> result_ = prepare_dense_vector_for_cusparse(result);
// 准备 result 的稠密向量，以备 cuSPARSE 使用
c10::MaybeOwned<Tensor> vec_ = prepare_dense_vector_for_cusparse(vec);
// 准备 vec 的稠密向量，以备 cuSPARSE 使用

auto block_size = cuda_int_cast(mat.values().size(2), "block_size");
// 计算块的尺寸大小，并转换为 CUDA 中的整数类型
auto nnzb = cuda_int_cast(mat._nnz(), "nnzb");
// 计算非零元素的数量，并转换为 CUDA 中的整数类型
auto mb = cuda_int_cast(mat.size(0), "mb") / block_size;
// 计算行的块数，并转换为 CUDA 中的整数类型
auto nb = cuda_int_cast(mat.size(1), "nb") / block_size;
// 计算列的块数，并转换为 CUDA 中的整数类型

AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
    result.scalar_type(), "block_sparse_mv", [&] {
      auto beta_ = beta.to<scalar_t>();
      // 将 beta 转换为当前标量类型的值
      auto alpha_ = alpha.to<scalar_t>();
      // 将 alpha 转换为当前标量类型的值
      auto handle = at::cuda::getCurrentCUDASparseHandle();
      // 获取当前 CUDA 的稀疏句柄
      auto desc = at::cuda::sparse::CuSparseMatDescriptor();
      // 创建 cuSPARSE 矩阵描述符
      auto values = mat.values();
      // 获取 mat 的 values 张量
      auto values_data_ptr = values.data_ptr<scalar_t>();
      // 获取 values 张量的数据指针，并转换为当前标量类型
      auto crow_indices = mat.crow_indices().to(kInt);
      // 获取 mat 的行索引，转换为整数类型
      auto crow_indices_data_ptr = crow_indices.data_ptr<int>();
      // 获取行索引的数据指针
      auto col_indices = mat.col_indices().to(kInt);
      // 获取 mat 的列索引，转换为整数类型
      auto col_indices_data_ptr = col_indices.data_ptr<int>();
      // 获取列索引的数据指针
      at::cuda::sparse::bsrmv(
          handle,
          block_layout,
          CUSPARSE_OPERATION_NON_TRANSPOSE,
          mb,
          nb,
          nnzb,
          &alpha_,
          desc.descriptor(),
          values_data_ptr,
          crow_indices_data_ptr,
          col_indices_data_ptr,
          block_size,
          vec_->data_ptr<scalar_t>(),
          &beta_,
          result_->data_ptr<scalar_t>());
      // 使用 cuSPARSE 进行块稀疏矩阵向量乘法运算
    });
if (!result.is_same(*result_)) {
  result.copy_(*result_);
}
// 如果 result 和 result_ 不是同一个 Tensor，就将 result_ 的值复制给 result
}
    result.copy_(input);
  }

  // 确定 mat1 是否连续，选择块布局的方向
  const cusparseDirection_t block_layout = mat1.values().is_contiguous()
      ? CUSPARSE_DIRECTION_ROW  // 如果连续，则按行布局
      : CUSPARSE_DIRECTION_COLUMN;  // 否则按列布局

  // 准备 mat2 以供 cuSPARSE 使用
  c10::MaybeOwned<Tensor> mat2_ = prepare_dense_matrix_for_cusparse(mat2);

  // 准备 result 以在 cuSPARSE 中使用列主要存储格式
  c10::MaybeOwned<Tensor> result_ =
      prepare_column_major_matrix_for_cusparse(result);

  // 获取 result 和 mat2 的步长信息
  IntArrayRef result_strides = result_->strides();
  IntArrayRef mat2_strides = mat2_->strides();
  auto ndim = result_->dim();

  // 运行时断言，确保维度正确
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(ndim == 2);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(mat1.dim() == 2);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(mat2.dim() == 2);

  // 检查 mat2 是否是行主要存储
  bool is_mat2_row_major = (mat2_strides[ndim - 1] == 1);

  // 设置 ldb 和 ldc 参数
  int ldb = is_mat2_row_major ? cuda_int_cast(mat2_strides[ndim - 2], "ldb")
                              : cuda_int_cast(mat2_strides[ndim - 1], "ldb");
  int ldc = cuda_int_cast(result_strides[ndim - 1], "ldc");

  // 计算其他需要的参数
  auto block_size = cuda_int_cast(mat1.values().size(2), "block_size");
  auto nnzb = cuda_int_cast(mat1._nnz(), "nnzb");
  auto mb = cuda_int_cast(mat1.size(0), "mb") / block_size;
  auto kb = cuda_int_cast(mat1.size(1), "nb") / block_size;
  auto n = cuda_int_cast(mat2.size(1), "n");

  // 根据 cuSPARSE 文档，设置 opA 和 opB 的操作类型
  cusparseOperation_t opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
  cusparseOperation_t opB = is_mat2_row_major
      ? CUSPARSE_OPERATION_TRANSPOSE  // 如果 mat2 是行主要存储，则对 mat2 进行转置操作
      : CUSPARSE_OPERATION_NON_TRANSPOSE;  // 否则不进行转置操作

  // 使用 AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES 宏处理浮点和复数类型
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      result.scalar_type(), "block_sparse_mm", [&] {
        auto beta_ = beta.to<scalar_t>();
        auto alpha_ = alpha.to<scalar_t>();
        auto handle = at::cuda::getCurrentCUDASparseHandle();
        auto desc = at::cuda::sparse::CuSparseMatDescriptor();

        // 获取 mat1 的值、行指针和列指针的数据指针
        auto values = mat1.values();
        auto values_data_ptr = values.data_ptr<scalar_t>();
        auto crow_indices = mat1.crow_indices().to(kInt);
        auto crow_indices_data_ptr = crow_indices.data_ptr<int>();
        auto col_indices = mat1.col_indices().to(kInt);
        auto col_indices_data_ptr = col_indices.data_ptr<int>();

        // 调用 cuSPARSE 的 BSRMM 函数进行块矩阵乘法计算
        at::cuda::sparse::bsrmm(
            handle,
            block_layout,
            opA,
            opB,
            mb,
            n,
            kb,
            nnzb,
            &alpha_,
            desc.descriptor(),
            values_data_ptr,
            crow_indices_data_ptr,
            col_indices_data_ptr,
            block_size,
            mat2_->data_ptr<scalar_t>(),
            ldb,
            &beta_,
            result_->data_ptr<scalar_t>(),
            ldc);
      });

  // 如果 result 和 result_ 不是同一个 Tensor，则将 result_ 的数据复制到 result 中
  if (!result.is_same(*result_)) {
    result.copy_(*result_);
  }
}

void spmm(
    const at::sparse_csr::SparseCsrTensor& mat1, // 第一个稀疏 CSR 格式的输入张量
    const Tensor& mat2, // 第二个稠密格式的输入张量
    const Scalar& beta, // 乘以结果的缩放因子
    const Scalar& alpha, // 乘以矩阵乘法结果的缩放因子
    const Tensor& result) { // 输出结果张量
#if !(AT_USE_CUSPARSE_GENERIC_API() || AT_USE_HIPSPARSE_GENERIC_API())
  addmm_out_legacy(mat1, mat2, beta, alpha, result); // 使用 legacy 方法执行矩阵乘法并将结果加到输出张量中
#else
  c10::MaybeOwned<Tensor> result_ = prepare_dense_matrix_for_cusparse(result); // 准备用于 cuSPARSE 的稠密矩阵形式的输出张量
  c10::MaybeOwned<Tensor> mat2_ = prepare_dense_matrix_for_cusparse(mat2); // 准备用于 cuSPARSE 的稠密矩阵形式的输入张量

  // 这里下标 "c" 表示列优先顺序，"r" 表示行优先顺序，cuSPARSE 支持这两种顺序。
  // 对于混合输入，需要将 'mat2' 转换为 'result' 的顺序。我们计算 result = mat1 @ op(mat2) + result。
  // 如果 'mat2' 的顺序与 'result' 匹配，则 op 是恒等的，op(mat2) == mat2。
  // 如果 'result' 是列优先的而 'mat2' 是行优先的，我们将 'mat2' 作为列优先传递，并计算
  // result_c = mat1 @ transpose(mat2_c) + result_c；其中 mat2_r == transpose(mat2_c)。
  // 如果 'result' 是行优先的而 'mat2' 是列优先的，我们将 'mat2' 作为行优先传递，并计算
  // result_r = mat1 @ transpose(mat2_r) + result_r；其中 mat2_c == transpose(mat2_r)。
  IntArrayRef result_strides = result_->strides(); // 获取输出张量的步幅
  IntArrayRef mat2_strides = mat2_->strides(); // 获取输入张量 mat2 的步幅
  auto ndim = result_->dim(); // 获取输出张量的维度数
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(ndim == 2 || ndim == 3); // 断言输出张量的维度为 2 或 3
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(mat1.dim() == 2 || mat1.dim() == 3); // 断言输入稀疏张量 mat1 的维度为 2 或 3
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(mat2.dim() == 2 || mat2.dim() == 3); // 断言输入张量 mat2 的维度为 2 或 3
  bool is_result_row_major = (result_strides[ndim - 1] == 1); // 检查输出张量是否是行优先顺序
  bool is_mat2_row_major = (mat2_strides[ndim - 1] == 1); // 检查输入张量 mat2 是否是行优先顺序
  bool transpose_B = (is_result_row_major ^ is_mat2_row_major); // 计算是否需要对输入张量 mat2 进行转置

  cusparseOperation_t opA = CUSPARSE_OPERATION_NON_TRANSPOSE; // cuSPARSE 操作类型，不进行转置
  cusparseOperation_t opB = transpose_B ? CUSPARSE_OPERATION_TRANSPOSE
                                        : CUSPARSE_OPERATION_NON_TRANSPOSE; // 根据 transpose_B 确定是否需要转置 mat2

  // CUDA < 11.0 不支持 64 位索引，并且不会报错，但可能导致结果不正确
#if defined(USE_ROCM)
  auto mat1_32 = at::native::_sparse_csr_tensor_unsafe(
      mat1.crow_indices().to(kInt),
      mat1.col_indices().to(kInt),
      mat1.values(),
      mat1.sizes(),
      mat1.scalar_type(),
      mat1.layout(),
      mat1.device());
  auto descA = at::cuda::sparse::CuSparseSpMatCsrDescriptor(mat1_32); // 创建稀疏 CSR 描述符
  auto algorithm = CUSPARSE_MM_ALG_DEFAULT; // cuSPARSE 矩阵乘法的算法选择
#else
  // TODO: update this to support COO sparse layout
  auto descA = at::cuda::sparse::CuSparseSpMatCsrDescriptor(mat1); // 创建稀疏 CSR 描述符
  auto algorithm = CUSPARSE_SPMM_CSR_ALG2; // cuSPARSE CSR 矩阵乘法的算法选择
#endif
#endif

  // 创建CuSparseConstDnMatDescriptor对象descB，根据transpose_B条件选择mat2_->mT()或*mat2_作为参数
  auto descB = at::cuda::sparse::CuSparseConstDnMatDescriptor(
      transpose_B ? mat2_->mT() : *mat2_);
  // 创建CuSparseDnMatDescriptor对象descC，使用*result_作为参数
  auto descC = at::cuda::sparse::CuSparseDnMatDescriptor(*result_);

  // 根据result的数据类型，调用AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2处理函数
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      kHalf,
      kBFloat16,
      result.scalar_type(),
      "spmm",
      [&] {
        // 定义scalar_t的opmath_t别名
        using opmath_t = at::opmath_type<scalar_t>;
        // 将beta和alpha转换为opmath_t类型
        auto beta_ = beta.to<opmath_t>();
        auto alpha_ = alpha.to<opmath_t>();
        // 获取opmath_t的CUDA数据类型
        cudaDataType compute_type = at::cuda::getCudaDataType<opmath_t>();
        // 获取当前CUDA稀疏操作句柄
        auto handle = at::cuda::getCurrentCUDASparseHandle();

        // 计算CUSPARSE稀疏矩阵乘法所需的缓冲区大小
        size_t buffer_size;
        TORCH_CUDASPARSE_CHECK(cusparseSpMM_bufferSize(
            handle,
            opA,
            opB,
            &alpha_,
            descA.descriptor(),
            descB.unsafe_mutable_descriptor(),
            &beta_,
            descC.descriptor(),
            compute_type,
            algorithm,
            &buffer_size // output
            ));

        // 获取CUDA缓存分配器的引用
        auto& allocator = *c10::cuda::CUDACachingAllocator::get();
        // 使用buffer_size分配工作缓冲区
        auto work_data = allocator.allocate(buffer_size);

        // 执行CUSPARSE稀疏矩阵乘法
        TORCH_CUDASPARSE_CHECK(cusparseSpMM(
            handle,
            opA,
            opB,
            &alpha_,
            descA.descriptor(),
            descB.unsafe_mutable_descriptor(),
            &beta_,
            descC.descriptor(),
            compute_type,
            algorithm,
            work_data.get()));
      });

  // 如果result不与*result_相同，则将*result_的数据复制到result中
  if (!result.is_same(*result_)) {
    result.copy_(*result_);
  }
#endif // !(AT_USE_CUSPARSE_GENERIC_API() || AT_USE_HIPSPARSE_GENERIC_API())
}

// anonymous namespace结束
} 

// 定义函数addmm_out_sparse_csr，接收输入input、mat1、mat2、beta、alpha和输出结果C
void addmm_out_sparse_csr(
    const Tensor& input,
    const Tensor& mat1,
    const Tensor& mat2,
    const Scalar& beta,
    const Scalar& alpha,
    const Tensor& result) {
  // 检查至少有一个稀疏输入的断言
  TORCH_INTERNAL_ASSERT(
      !((mat1.layout() == kStrided) && (mat2.layout() == kStrided) &&
        (result.layout() == kStrided)),
      "Expected at least one sparse input");

  // 布局检查依次为 mat1, mat2, result
  // 条件顺序为 strided, csr, csc, bsr, bsc。
  // 有效组合直接返回
  // 无效组合会跳过，后续会通过 TORCH 检查生成详细的错误信息

  // mm 函数在需要时将输入复制到 result 中（例如 mm Triton 内核不要求 result 初始化为 input）
  if (mat1.layout() == kSparseBsr) {
    if (mat2.layout() == kStrided) {
      if (result.layout() == kStrided)
        // 调用块稀疏矩阵乘法函数
        return block_sparse_mm(input, mat1, mat2, beta, alpha, result);
    }
  }

  if (mat1.layout() == kStrided) {
    if (mat2.layout() == kSparseBsc) {
      if (result.layout() == kStrided) {
        // 对 result 进行转置
        auto result_t = result.transpose(-2, -1);
        // 根据 result 是否与 input 相同选择进行转置的操作
        auto input_t = (result.is_same(input) ? result_t : input.transpose(-2, -1));
        // 调用块稀疏矩阵乘法函数
        return block_sparse_mm(
            input_t,
            mat2.transpose(-2, -1),
            mat1.transpose(-2, -1),
            beta,
            alpha,
            result_t);
      }
    }
  }

  // 将 input 复制到 result：
  if (beta.toComplexDouble() != 0. && !result.is_same(input)) {
    result.copy_(input);
  }

  // mm 函数假设 result 包含 input：
  if (mat1.layout() == kStrided) {
    if (mat2.layout() == kSparseCsr) {
      if (result.layout() == kStrided) {
        // TODO: 如果支持，通过 cuSPARSE 添加原生 CSC 支持。
        // 调用稀疏矩阵乘法函数，传入转置后的稀疏 CSR 格式的 mat2
        return spmm(
            mat2.transpose(0, 1).to_sparse_csr(),
            mat1.transpose(0, 1),
            beta,
            alpha,
            result.transpose(0, 1));
      }
    }
    if (mat2.layout() == kSparseCsc) {
      if (result.layout() == kStrided) {
        // 调用稀疏矩阵乘法函数，传入转置后的 mat2
        return spmm(
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
        // 调用稀疏矩阵乘法函数
        return spmm(mat1, mat2, beta, alpha, result);
      }
    }
    if (mat2.layout() == kSparseCsr) {
      if (result.layout() == kSparseCsr) {
        // 调用稀疏矩阵乘法函数，执行 CSR 格式的稀疏矩阵乘法
        return spgemm(mat1, mat2, beta, alpha, result);
      }
    }
    if (mat2.layout() == kSparseCsc) {
      if (result.layout() == kSparseCsr) {
        // TODO: 如果支持，通过 cuSPARSE 添加原生 CSC 支持。
        // CSR @ CSC 内核因格式对齐而非常快
        // 调用稀疏矩阵乘法函数，传入 mat2 转为稀疏 CSR 格式后的参数
        return spgemm(mat1, mat2.to_sparse_csr(), beta, alpha, result);
      }
    }
  }
  if (mat1.layout() == kSparseCsc) {
    // 检查 mat2 是否以 Strided 布局存储
    if (mat2.layout() == kStrided) {
      // 如果 result 也以 Strided 布局存储
      if (result.layout() == kStrided) {
        // TODO: 如果支持的话，通过 cuSPARSE 添加对 CSC 的本地支持
        return spmm(mat1.to_sparse_csr(), mat2, beta, alpha, result);
      }
    }
    // 检查 mat2 是否以 SparseCsr 布局存储
    if (mat2.layout() == kSparseCsr) {
      // 如果 result 也以 SparseCsr 布局存储
      if (result.layout() == kSparseCsr)
        // TODO: 如果支持的话，通过 cuSPARSE 添加对 CSC 的本地支持
        return spgemm(mat1.to_sparse_csr(), mat2, beta, alpha, result);
    }
    // 检查 mat2 是否以 SparseCsc 布局存储
    if (mat2.layout() == kSparseCsc) {
      // 如果 result 以 SparseCsr 布局存储
      if (result.layout() == kSparseCsr) {
        // TODO: 如果支持的话，通过 cuSPARSE 添加对 CSC 的本地支持
        return spgemm(
            mat1.to_sparse_csr(), mat2.to_sparse_csr(), beta, alpha, result);
      }
      // 如果 result 以 SparseCsc 布局存储
      if (result.layout() == kSparseCsc) {
        // 进行 CSC 稀疏矩阵乘法，同时转置输入矩阵以适应 SparseCsc 布局
        return spgemm(
            mat2.transpose(-2, -1),
            mat1.transpose(-2, -1),
            beta,
            alpha,
            result.transpose(-2, -1));
      }
    }
  }
  // 如果以上条件都不满足，抛出错误，指出 CUDA 上的 addmm 计算未实现
  TORCH_CHECK(
      false,
      "addmm: computation on CUDA is not implemented for ",
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
    const at::sparse_csr::SparseCsrTensor& mat,
    const Tensor& vec,
    const Scalar& beta,
    const Scalar& alpha,
    const Tensor& result) {
  if (mat.layout() == kSparseBsr) {  // 检查稀疏矩阵的布局是否为 kSparseBsr
    return block_sparse_mv(mat, vec, beta, alpha, result);  // 如果是 kSparseBsr，则调用 block_sparse_mv 函数处理
  }
#if !(AT_USE_CUSPARSE_GENERIC_API() || AT_USE_HIPSPARSE_GENERIC_API())
  // 如果不使用通用的 CUDA 或 HIP 的稀疏库，则抛出错误信息
  TORCH_CHECK(
      false,
      "Calling addmv on a sparse GPU tensor requires compiling ",
      "PyTorch with CUDA 10.2+ (CUDA 11+ on Windows). ",
      "Please use PyTorch built with newer CUDA version.");
#else
  cusparseOperation_t opA = CUSPARSE_OPERATION_NON_TRANSPOSE;  // 设置操作类型为非转置

  // 准备 result 和 vec 的稀疏格式描述符，以备使用 cuSPARSE 函数
  c10::MaybeOwned<Tensor> result_ = prepare_dense_vector_for_cusparse(result);
  c10::MaybeOwned<Tensor> vec_ = prepare_dense_vector_for_cusparse(vec);

  // TODO: update this to support COO sparse layout
  auto descA = at::cuda::sparse::CuSparseSpMatCsrDescriptor(mat);  // 创建 CSR 格式的稀疏矩阵描述符
  auto descX = at::cuda::sparse::CuSparseDnVecDescriptor(*vec_);  // 创建稠密向量 x 的描述符
  auto descY = at::cuda::sparse::CuSparseDnVecDescriptor(*result_);  // 创建稠密向量 result 的描述符

  // 根据 cuSPARSE 的版本选择适当的算法类型
#if CUSPARSE_VERSION >= 11400
  cusparseSpMVAlg_t alg = CUSPARSE_SPMV_ALG_DEFAULT;  // CUDA 版本大于等于 11.4.0 使用默认的 SpMV 算法
#else
  cusparseSpMVAlg_t alg = CUSPARSE_MV_ALG_DEFAULT;  // CUDA 版本低于 11.4.0 使用默认的 MV 算法
#endif

  // SpMV 不支持统一精度计算，对于 float16/bfloat16 输入，compute_type 必须为 CUDA_R_32F，alpha 和 beta 的类型必须为 float
  auto dispatch_scalar_type = result.scalar_type();
  if (dispatch_scalar_type == at::ScalarType::Half ||
      dispatch_scalar_type == at::ScalarType::BFloat16) {
    // 设置默认的标量类型为浮点数
    dispatch_scalar_type = at::ScalarType::Float;
  }

  // 使用宏处理各种浮点数和复数类型的情况，实现稀疏矩阵-向量乘法的 CUDA 实现
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      dispatch_scalar_type,
      "addmv_out_sparse_csr_cuda_impl",
      [&] {
        // 将 beta 转换为当前标量类型的值
        auto beta_ = beta.to<scalar_t>();
        // 将 alpha 转换为当前标量类型的值
        auto alpha_ = alpha.to<scalar_t>();
        // 获取当前标量类型对应的 CUDA 计算数据类型
        cudaDataType compute_type = at::cuda::getCudaDataType<scalar_t>();
        // 获取当前 CUDA 上下文的稀疏处理句柄
        auto handle = at::cuda::getCurrentCUDASparseHandle();

        // 查询所需的缓冲区大小
        size_t buffer_size;
        TORCH_CUDASPARSE_CHECK(cusparseSpMV_bufferSize(
            handle,
            opA,
            &alpha_,
            descA.descriptor(),
            descX.descriptor(),
            &beta_,
            descY.descriptor(),
            compute_type,
            alg,
            &buffer_size // 输出：缓冲区大小
            ));

        // 获取当前 CUDA 内存分配器
        auto& allocator = *c10::cuda::CUDACachingAllocator::get();
        // 分配所需大小的工作数据缓冲区
        auto work_data = allocator.allocate(buffer_size);

        // 执行稀疏矩阵-向量乘法
        TORCH_CUDASPARSE_CHECK(cusparseSpMV(
            handle,
            opA,
            &alpha_,
            descA.descriptor(),
            descX.descriptor(),
            &beta_,
            descY.descriptor(),
            compute_type,
            alg,
            work_data.get()));
      });
  // 如果结果与 result_ 不是同一个张量，则将 result_ 的内容复制到 result 中
  if (!result.is_same(*result_)) {
    result.copy_(*result_);
  }
#endif // !(AT_USE_CUSPARSE_GENERIC_API() || AT_USE_HIPSPARSE_GENERIC_API())
}

/*
  Computes C = alpha * A + beta * B

  Args:
  * `A` - [in] sparse Tensor of size m × n.
  * `B` - [in] sparse Tensor of size m × n.
  * `C` - [out] sparse Tensor of size m × n.
*/
void add_out_sparse_csr(
    const at::sparse_csr::SparseCsrTensor& A,
    const at::sparse_csr::SparseCsrTensor& B,
    const Scalar& alpha,
    const Scalar& beta,
    const at::sparse_csr::SparseCsrTensor& C) {
  IntArrayRef A_sizes = A.sizes();
  auto ndim = A.dim();
  int m = at::native::cuda_int_cast(A_sizes[ndim - 2], "m");
  int n = at::native::cuda_int_cast(A_sizes[ndim - 1], "n");

  // Assertion to ensure all input tensors have the same size
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(A.sizes().equals(B.sizes()) && A.sizes().equals(C.sizes()));

  // Only 32-bit indices are supported; promote indices types if necessary
  const auto output_indices_dtype = promoteTypes(A.crow_indices().scalar_type(), B.crow_indices().scalar_type());

  // Create new sparse tensors A_32 and B_32 with 32-bit indices
  auto A_32 = at::native::_sparse_csr_tensor_unsafe(
      A.crow_indices().to(kInt),
      A.col_indices().to(kInt),
      A.values(),
      A.sizes(),
      A.scalar_type(),
      A.layout(),
      A.device());
  auto B_32 = at::native::_sparse_csr_tensor_unsafe(
      B.crow_indices().to(kInt),
      B.col_indices().to(kInt),
      B.values(),
      B.sizes(),
      B.scalar_type(),
      B.layout(),
      B.device());

  // Modify C tensor in-place to swap indices tensors with 32-bit variants
  auto C_crow_indices_backup = C.crow_indices();
  auto C_col_indices_backup = C.col_indices();
  indices_to_32_bit_inplace(C); // no-op with 32-bit indices

  // Cast nnzA and nnzB to int for CUDA operations
  int nnzA = at::native::cuda_int_cast(A_32._nnz(), "nnzA");
  int nnzB = at::native::cuda_int_cast(B_32._nnz(), "nnzB");

  // CuSparseMatDescriptor for sparse matrix operations on CUDA
  auto desc = at::cuda::sparse::CuSparseMatDescriptor();

  // Extract pointers to crow_indices and col_indices for A, B, and C tensors
  auto A_crow_indices = A_32.crow_indices();
  auto B_crow_indices = B_32.crow_indices();
  auto C_crow_indices = C.crow_indices();
  auto A_crow_indices_ptr = A_crow_indices.data_ptr<int>();
  auto B_crow_indices_ptr = B_crow_indices.data_ptr<int>();
  auto C_crow_indices_ptr = C_crow_indices.data_ptr<int>();

  auto A_col_indices = A_32.col_indices();
  auto B_col_indices = B_32.col_indices();
  auto C_col_indices = C.col_indices();
  auto A_col_indices_ptr = A_col_indices.data_ptr<int>();
  auto B_col_indices_ptr = B_col_indices.data_ptr<int>();
  auto C_col_indices_ptr = C_col_indices.data_ptr<int>();

  // Lambda function to fix nnz based on platform-specific conditions
  auto fix_nnz = [
#if AT_ROCM_ENABLED()
                     &C_crow_indices,
                     &m
#endif
  ](int nnz) -> int {
    // Adjust nnz value based on ROCm platform specifics
#if AT_ROCM_ENABLED()
    return std::max({nnz, C_crow_indices.narrow(-1, m, 1).item<int>()});
#else
    return nnz;
  };
}
/*
  解决一个线性方程组，其中系数在稀疏三角矩阵 A 中表示：op(A) X = B。

  Args:
  * `A` - 大小为 m × m 的稀疏张量。
  * `B` - 大小为 m × nrhs 的密集张量。
  * `X` - 大小为 m × nrhs 的密集张量。
  * `upper` - 控制计算中使用矩阵 A 的上三角部分还是下三角部分。
  * `transpose` - 如果为 true，则 op(A) = A^T。
  * `unitriangular` - 如果为 true，则假设矩阵 A 的对角元素为一。
*/
void triangular_solve_out_sparse_csr(
    const at::sparse_csr::SparseCsrTensor& A,
    const Tensor& B,
    const Tensor& X,
    bool upper,
    bool transpose,
    bool unitriangular) {
  if (B.numel() == 0 || X.numel() == 0 || A._nnz() == 0) {
    // 如果 A 的非零元素数为 0，则 A 是奇异的，无法求解。
    X.fill_(NAN);  // 填充结果张量 X 为 NaN
    return;
  }
  if (A.layout() == kSparseBsr) {
    // 如果 A 的布局为 kSparseBsr
    if (B.size(-1) == 1) {
      return block_sparse_triangular_solve_vec(A, B, X, upper, transpose, unitriangular);
      // 调用向量形式的块稀疏三角求解器
    } else {
      return block_sparse_triangular_solve_mat(A, B, X, upper, transpose, unitriangular);
      // 调用矩阵形式的块稀疏三角求解器
    }
  }
#if !AT_USE_CUSPARSE_GENERIC_SPSV()
  // 如果未启用 CUSPARSE 通用稀疏求解功能
  TORCH_CHECK(
      false,
      "Calling triangular solve on a sparse GPU tensor requires compiling ",
      "PyTorch with at least CUDA 11.3. ",
      "Please use PyTorch built with newer CUDA version.");
#else
  c10::MaybeOwned<Tensor> X_ = prepare_dense_matrix_for_cusparse(X);
  // 可能需要使用混合内存格式，但在 CUDA 11.3.1 中存在一个 bug：
  // 使用矩阵 B 的步长来写入结果到矩阵 X。
  // 作为解决方法，我们需要将矩阵转换为具有相同步长的格式。

  c10::MaybeOwned<Tensor> B_ = prepare_dense_matrix_for_cusparse(B, X_->strides());

  // TODO: update this to support COO sparse layout
  // 更新以支持 COO 稀疏布局

  auto descA = at::cuda::sparse::CuSparseSpMatCsrDescriptor(A);
  descA.set_mat_fill_mode(upper);
  descA.set_mat_diag_type(unitriangular);
  cusparseOperation_t opA = transpose ? CUSPARSE_OPERATION_TRANSPOSE
                                      : CUSPARSE_OPERATION_NON_TRANSPOSE;

  if (B.size(-1) == 1) {
    // 如果 B 的最后一个维度大小为 1
    // 使用宏 AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES 来处理不同的浮点数和复数类型
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
        X.scalar_type(), "triangular_solve_out_sparse_csr_cuda_impl", [&] {
          // 设置标量 alpha 为 1
          scalar_t alpha = 1;
          // 获取计算类型的 CUDA 数据类型
          cudaDataType compute_type = at::cuda::getCudaDataType<scalar_t>();
          // 获取当前 CUDA 稀疏库的句柄
          auto handle = at::cuda::getCurrentCUDASparseHandle();
          // 定义缓冲区大小
          size_t buffer_size;

          // 创建 CuSparseSpSVDescriptor 对象 desc_spsv
          auto desc_spsv = at::cuda::sparse::CuSparseSpSVDescriptor();
          // 根据输入的 B_ 创建 CuSparseDnVecDescriptor 对象 descB
          auto descB = at::cuda::sparse::CuSparseDnVecDescriptor(*B_);
          // 根据输入的 X_ 创建 CuSparseDnVecDescriptor 对象 descX
          auto descX = at::cuda::sparse::CuSparseDnVecDescriptor(*X_);
          // 调用 cusparseSpSV_bufferSize 获取所需的缓冲区大小
          TORCH_CUDASPARSE_CHECK(cusparseSpSV_bufferSize(
              handle,
              opA,
              &alpha,
              descA.descriptor(),
              descB.descriptor(),
              descX.descriptor(),
              compute_type,
              CUSPARSE_SPSV_ALG_DEFAULT,
              desc_spsv.descriptor(),
              &buffer_size // output
              ));

          // 获取 CUDA 内存分配器的引用
          auto& allocator = *c10::cuda::CUDACachingAllocator::get();
          // 使用分配器分配指定大小的工作数据内存
          auto work_data = allocator.allocate(buffer_size);

          // 调用 cusparseSpSV_analysis 进行稀疏矩阵向量求解的分析
          TORCH_CUDASPARSE_CHECK(cusparseSpSV_analysis(
              handle,
              opA,
              &alpha,
              descA.descriptor(),
              descB.descriptor(),
              descX.descriptor(),
              compute_type,
              CUSPARSE_SPSV_ALG_DEFAULT,
              desc_spsv.descriptor(),
              work_data.get()));

          // 调用 cusparseSpSV_solve 完成稀疏矩阵向量的求解
          TORCH_CUDASPARSE_CHECK(cusparseSpSV_solve(
              handle,
              opA,
              &alpha,
              descA.descriptor(),
              descB.descriptor(),
              descX.descriptor(),
              compute_type,
              CUSPARSE_SPSV_ALG_DEFAULT,
              desc_spsv.descriptor()));
        });
  } else {
#if !AT_USE_CUSPARSE_GENERIC_SPSM()
    // 如果未启用特定的 cusparse generic SPSM，则执行以下代码块
    TORCH_CHECK(
        false,
        "Calling triangular solve on a sparse GPU tensor requires compiling ",
        "PyTorch with at least CUDA 11.3.1. ",
        "Please use PyTorch built with newer CUDA version.");
#else
    // 如果启用了 cusparse generic SPSM，则执行以下代码块
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
        X.scalar_type(), "triangular_solve_out_sparse_csr_cuda_impl", [&] {
          scalar_t alpha = 1;
          cudaDataType compute_type = at::cuda::getCudaDataType<scalar_t>();
          auto handle = at::cuda::getCurrentCUDASparseHandle();
          size_t buffer_size;

          cusparseOperation_t opB = CUSPARSE_OPERATION_NON_TRANSPOSE;
          auto desc_spsm = at::cuda::sparse::CuSparseSpSMDescriptor();
          auto descB = at::cuda::sparse::CuSparseDnMatDescriptor(*B_);
          auto descX = at::cuda::sparse::CuSparseDnMatDescriptor(*X_);
          // 查询所需的缓冲区大小
          TORCH_CUDASPARSE_CHECK(cusparseSpSM_bufferSize(
              handle,
              opA,  // opA 未在提供的代码片段中定义
              opB,
              &alpha,
              descA.descriptor(),
              descB.descriptor(),
              descX.descriptor(),
              compute_type,
              CUSPARSE_SPSM_ALG_DEFAULT,
              desc_spsm.descriptor(),
              &buffer_size // output
              ));

          auto& allocator = *c10::cuda::CUDACachingAllocator::get();
          auto work_data = allocator.allocate(buffer_size);

          // 分析稀疏矩阵的结构
          TORCH_CUDASPARSE_CHECK(cusparseSpSM_analysis(
              handle,
              opA,  // opA 未在提供的代码片段中定义
              opB,
              &alpha,
              descA.descriptor(),
              descB.descriptor(),
              descX.descriptor(),
              compute_type,
              CUSPARSE_SPSM_ALG_DEFAULT,
              desc_spsm.descriptor(),
              work_data.get()));

          // 解决稀疏矩阵方程
          TORCH_CUDASPARSE_CHECK(cusparseSpSM_solve(
              handle,
              opA,  // opA 未在提供的代码片段中定义
              opB,
              &alpha,
              descA.descriptor(),
              descB.descriptor(),
              descX.descriptor(),
              compute_type,
              CUSPARSE_SPSM_ALG_DEFAULT,
              desc_spsm.descriptor()));
        });
#endif // !AT_USE_CUSPARSE_GENERIC_SPSM()

  // 如果 X 与 X_ 指向的对象不同，则复制 X_ 的内容到 X
  if (!X.is_same(*X_)) {
    X.copy_(*X_);
  }
#endif // !AT_USE_CUSPARSE_GENERIC_SPSV()
}
#else
  // 断言：A 的布局必须是 Strided
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(A.layout() == Layout::Strided);
  // 断言：B 的布局必须是 Strided
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(B.layout() == Layout::Strided);
  // 断言：C 必须是 CSR 稀疏格式
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(C.is_sparse_csr());

  // 断言：A 和 B 的批次数必须相等
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(batchCount(A) == batchCount(B));
  // 断言：A 和 C 的批次数必须相等
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(batchCount(A) == batchCount(C));

  // 定义操作类型为非转置
  cusparseOperation_t opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
  cusparseOperation_t opB = CUSPARSE_OPERATION_NON_TRANSPOSE;

  // 为 A 和 B 准备密集矩阵，返回一个 MaybeOwned<Tensor>
  c10::MaybeOwned<Tensor> A_ = prepare_dense_matrix_for_cusparse(A);
  c10::MaybeOwned<Tensor> B_ = prepare_dense_matrix_for_cusparse(B);

  // 在指定类型的闭包中分发处理浮点和复数类型
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      C.scalar_type(),
      "sampled_addmm_out_sparse_csr",
      [&] {
        // CUDA 11.6 不支持批量输入，会引发错误：
        // ** On entry to cusparseSDDMM_bufferSize(): batched SDDMM is not supported
        // 因此我们需要使用 for 循环来处理
        for (const auto i : c10::irange(batchCount(A))) {
          // 获取 A 的 CuSparse 常量密集矩阵描述符，使用批次偏移 i
          auto descA = at::cuda::sparse::CuSparseConstDnMatDescriptor(*A_, /*batch_offset=*/i);
          // 获取 B 的 CuSparse 常量密集矩阵描述符，使用批次偏移 i
          auto descB = at::cuda::sparse::CuSparseConstDnMatDescriptor(*B_, /*batch_offset=*/i);
          // 获取 C 的 CuSparse CSR 稀疏矩阵描述符，使用批次偏移 i
          auto descC = at::cuda::sparse::CuSparseSpMatCsrDescriptor(C, /*batch_offset=*/i);

          // 将 beta 转换为当前标量类型的值
          auto beta_ = beta.to<scalar_t>();
          // 将 alpha 转换为当前标量类型的值
          auto alpha_ = alpha.to<scalar_t>();
          // 获取计算类型
          auto compute_type = at::cuda::getCudaDataType<scalar_t>();
          // 获取当前 CUDA Sparse 句柄
          auto handle = at::cuda::getCurrentCUDASparseHandle();
          // 计算所需的缓冲区大小
          size_t buffer_size = 0;
          // 查询 SDDMM 执行所需的缓冲区大小
          TORCH_CUDASPARSE_CHECK(cusparseSDDMM_bufferSize(
              handle,
              opA,
              opB,
              &alpha_,
              descA.unsafe_mutable_descriptor(),
              descB.unsafe_mutable_descriptor(),
              &beta_,
              descC.descriptor(),
              compute_type,
              CUSPARSE_SDDMM_ALG_DEFAULT,
              &buffer_size // 输出参数：缓冲区大小
              ));

          // 分配缓冲区内存
          auto& allocator = *c10::cuda::CUDACachingAllocator::get();
          auto buffer = allocator.allocate(buffer_size);

          // 执行 SDDMM 前处理
          TORCH_CUDASPARSE_CHECK(cusparseSDDMM_preprocess(
              handle,
              opA,
              opB,
              &alpha_,
              descA.unsafe_mutable_descriptor(),
              descB.unsafe_mutable_descriptor(),
              &beta_,
              descC.descriptor(),
              compute_type,
              CUSPARSE_SDDMM_ALG_DEFAULT,
              buffer.get()));

          // 执行 SDDMM 计算
          TORCH_CUDASPARSE_CHECK(cusparseSDDMM(
              handle,
              opA,
              opB,
              &alpha_,
              descA.unsafe_mutable_descriptor(),
              descB.unsafe_mutable_descriptor(),
              &beta_,
              descC.descriptor(),
              compute_type,
              CUSPARSE_SDDMM_ALG_DEFAULT,
              buffer.get()));
        }
      });
#endif
}

} // namespace at::native::sparse::impl::cuda
```