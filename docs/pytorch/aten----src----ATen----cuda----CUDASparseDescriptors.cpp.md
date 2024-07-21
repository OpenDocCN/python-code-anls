# `.\pytorch\aten\src\ATen\cuda\CUDASparseDescriptors.cpp`

```py
// 包含 CUDA 稀疏操作相关的头文件
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDADataType.h>
#include <ATen/cuda/CUDASparse.h>
#include <ATen/cuda/CUDASparseDescriptors.h>
#include <ATen/native/LinearAlgebraUtils.h>
#include <ATen/native/cuda/MiscUtils.h>

// 定义了 at::cuda::sparse 命名空间，用于 CUDA 稀疏操作
namespace at::cuda::sparse {

// 销毁稠密矩阵描述符的函数实现
cusparseStatus_t destroyConstDnMat(const cusparseDnMatDescr* dnMatDescr) {
  return cusparseDestroyDnMat(const_cast<cusparseDnMatDescr*>(dnMatDescr));
}

// 如果使用通用的 cuSparse API 或者 HIPSPARSE API，进入匿名命名空间
#if AT_USE_CUSPARSE_GENERIC_API() || AT_USE_HIPSPARSE_GENERIC_API()

namespace {

// 检查给定的 CUDA 数据类型是否被当前 GPU 支持
void check_supported_cuda_type(cudaDataType cuda_type) {
  if (cuda_type == CUDA_R_16F) {
    // 获取当前 GPU 的属性
    cudaDeviceProp* prop = at::cuda::getCurrentDeviceProperties();
    // 检查是否支持 Float16 类型的稀疏操作，如果不支持，抛出错误信息
    TORCH_CHECK(
        prop->major >= 5 && ((10 * prop->major + prop->minor) >= 53),
        "Sparse operations with CUDA tensors of Float16 type are not supported on GPUs with compute capability < 5.3 (current: ",
        prop->major,
        ".",
        prop->minor,
        ")");
  }
  // 如果不是 ROCm 平台，检查是否支持 BFloat16 类型的稀疏操作
  if (cuda_type == CUDA_R_16BF) {
    cudaDeviceProp* prop = at::cuda::getCurrentDeviceProperties();
    TORCH_CHECK(
        prop->major >= 8,
        "Sparse operations with CUDA tensors of BFloat16 type are not supported on GPUs with compute capability < 8.0 (current: ",
        prop->major,
        ".",
        prop->minor,
        ")");
  }
}

} // 匿名命名空间结束

#endif // AT_USE_CUSPARSE_GENERIC_API() || AT_USE_HIPSPARSE_GENERIC_API()

// 根据 Torch 的 ScalarType 返回对应的 cuSparse 索引类型
cusparseIndexType_t getCuSparseIndexType(const c10::ScalarType& scalar_type) {
  if (scalar_type == c10::ScalarType::Int) {
    return CUSPARSE_INDEX_32I;
  } else if (scalar_type == c10::ScalarType::Long) {
    return CUSPARSE_INDEX_64I;
  } else {
    // 如果类型无法转换，抛出内部断言错误
    TORCH_INTERNAL_ASSERT(
        false, "Cannot convert type ", scalar_type, " to cusparseIndexType.");
  }
}

// 如果使用通用的 cuSparse API 或者 HIPSPARSE API，创建原始的稠密矩阵描述符
cusparseDnMatDescr_t createRawDnMatDescriptor(const Tensor& input, int64_t batch_offset, bool is_const=false) {
  // 内部断言，确保输入张量的布局是 kStrided
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(input.layout() == kStrided);
  // 获取输入张量的步长和大小
  IntArrayRef input_strides = input.strides();
  IntArrayRef input_sizes = input.sizes();
  auto ndim = input.dim();
  // 内部断言，确保张量维度大于等于 2
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(ndim >= 2);
  // 获取矩阵的行数和列数
  auto rows = input_sizes[ndim - 2];
  auto cols = input_sizes[ndim - 1];

  // 检查是否是列优先或行优先的内存布局
  bool is_column_major =
      at::native::is_blas_compatible_column_major_order(input);
  bool is_row_major = at::native::is_blas_compatible_row_major_order(input);
  // 内部断言，期望输入张量要么是行优先要么是列优先的连续内存布局
  TORCH_INTERNAL_ASSERT(
      is_column_major || is_row_major,
      "Expected either row or column major contiguous input.");

  // 计算主维度（leading dimension）
  auto leading_dimension =
      is_row_major ? input_strides[ndim - 2] : input_strides[ndim - 1];

  // 如果不是 ROCm 平台，按照不同的内存布局选择不同的顺序
#if !defined(USE_ROCM)
  auto order = is_row_major ? CUSPARSE_ORDER_ROW : CUSPARSE_ORDER_COL;
#else
  // 在 ROCm 平台上，期望是列优先的内存布局
  TORCH_INTERNAL_ASSERT(is_column_major, "Expected column major input.");
  auto order = CUSPARSE_ORDER_COL;
#endif
#ifdef AT_USE_CUSPARSE_GENERIC_API() || AT_USE_HIPSPARSE_GENERIC_API()

  // 如果输入张量的维度大于2且batch_offset大于等于0，则计算batch_stride，否则置为0
  auto batch_stride = ndim > 2 && batch_offset >= 0 ? input_strides[ndim - 3] : 0;
  
  // 确定数据指针，如果是常量则使用const_data_ptr()，否则使用data_ptr()
  void* data_ptr = is_const ? const_cast<void*>(input.const_data_ptr()) : input.data_ptr();
  
  // 计算values_ptr，即数据起始指针，考虑到批处理偏移和批处理步长
  void* values_ptr = static_cast<char*>(data_ptr) +
      batch_offset * batch_stride * input.itemsize();

  // 将输入张量的标量类型转换为CUDA数据类型
  cudaDataType value_type = ScalarTypeToCudaDataType(input.scalar_type());
  
  // 检查CUDA数据类型是否受支持
  check_supported_cuda_type(value_type);

  // 注意：在常量情况下，理想情况下我们会使用cusparseConstDnMatDescr_t和cusparseCreateConstDnMat，
  // 但这些是在CUDA 12中引入的，我们仍然需要支持CUDA 11
  // 创建密集矩阵描述符
  cusparseDnMatDescr_t raw_descriptor = nullptr;
  TORCH_CUDASPARSE_CHECK(cusparseCreateDnMat(
      &raw_descriptor,
      rows,
      cols,
      leading_dimension,
      values_ptr,
      value_type,
      order));

  // 如果张量维度大于等于3且batch_offset为-1，则设置批处理参数
  if (ndim >= 3 && batch_offset == -1) {
    // 计算批处理的数量
    int batch_count =
        at::native::cuda_int_cast(at::native::batchCount(input), "batch_count");
    // 设置密集矩阵的批处理参数
    TORCH_CUDASPARSE_CHECK(cusparseDnMatSetStridedBatch(
        raw_descriptor, batch_count, input_strides[ndim - 3]));
  }
  // 返回原始密集矩阵描述符
  return raw_descriptor;
}

// 构造函数，使用给定的输入张量和批处理偏移创建CuSparseDnMatDescriptor对象
CuSparseDnMatDescriptor::CuSparseDnMatDescriptor(const Tensor& input, int64_t batch_offset) {
  descriptor_.reset(createRawDnMatDescriptor(input, batch_offset));
}

// 构造函数，使用给定的输入张量和批处理偏移创建CuSparseConstDnMatDescriptor对象
CuSparseConstDnMatDescriptor::CuSparseConstDnMatDescriptor(const Tensor& input, int64_t batch_offset) {
  // 调用带有is_const参数的createRawDnMatDescriptor函数
  descriptor_.reset(createRawDnMatDescriptor(input, batch_offset, /*is_const*/true));
}

#endif // AT_USE_CUSPARSE_GENERIC_API() || AT_USE_HIPSPARSE_GENERIC_API()

// 构造函数，使用给定的输入张量创建CuSparseDnVecDescriptor对象
CuSparseDnVecDescriptor::CuSparseDnVecDescriptor(const Tensor& input) {
  // 断言：cuSPARSE不支持批处理向量
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      input.dim() == 1 || (input.dim() == 2 && input.size(-1) == 1));

  // 断言：cuSPARSE不支持非连续向量
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(input.is_contiguous());

  // 将输入张量的标量类型转换为CUDA数据类型
  cudaDataType value_type = ScalarTypeToCudaDataType(input.scalar_type());
  
  // 检查CUDA数据类型是否受支持
  check_supported_cuda_type(value_type);

  // 创建密集向量描述符
  cusparseDnVecDescr_t raw_descriptor = nullptr;
  TORCH_CUDASPARSE_CHECK(cusparseCreateDnVec(
      &raw_descriptor, input.numel(), input.data_ptr(), value_type));
  
  // 将原始描述符包装到智能指针中
  descriptor_.reset(raw_descriptor);
}
# 定义 CuSparseSpMatCsrDescriptor 类的构造函数，接受稀疏张量和批处理偏移作为参数
CuSparseSpMatCsrDescriptor::CuSparseSpMatCsrDescriptor(const Tensor& input, int64_t batch_offset) {
  // 断言输入张量是 CSR 格式的稀疏张量，仅在调试模式下生效
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(input.is_sparse_csr());
  // 断言输入张量的维度至少为 2，仅在调试模式下生效
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(input.dim() >= 2);

  // 获取输入张量的尺寸
  IntArrayRef input_sizes = input.sizes();
  auto ndim = input.dim();
  auto rows = input_sizes[ndim - 2];  // 获取稀疏矩阵的行数
  auto cols = input_sizes[ndim - 1];  // 获取稀疏矩阵的列数

  // 获取稀疏张量的 CSR 行偏移、列索引和值
  auto crow_indices = input.crow_indices();  // CSR 行偏移
  auto col_indices = input.col_indices();    // CSR 列索引
  auto values = input.values();              // CSR 值
  auto nnz = values.size(-1);                // 获取非零元素的数量
  c10::MaybeOwned<Tensor> values_ = values.expect_contiguous();  // 确保值是连续的

  // 断言 CSR 行偏移和列索引是连续存储的，仅在调试模式下生效
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(crow_indices.is_contiguous());
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(col_indices.is_contiguous());

  // 获取 cuSPARSE 所需的索引类型和值类型
  cusparseIndexType_t index_type =
      getCuSparseIndexType(crow_indices.scalar_type());  // 获取行偏移的数据类型
  cudaDataType value_type = ScalarTypeToCudaDataType(input.scalar_type());  // 获取值的数据类型
  check_supported_cuda_type(value_type);  // 检查 CUDA 支持的数据类型

  // 计算批处理模式下 CSR 行偏移、列索引和值的批处理步长
  auto crow_indices_batch_stride = crow_indices.dim() >= 2 && batch_offset >= 0
      ? crow_indices.stride(-2)
      : 0;
  auto col_indices_batch_stride =
      col_indices.dim() >= 2 && batch_offset >= 0 ? col_indices.stride(-2) : 0;
  auto values_batch_stride =
      values.dim() >= 2 && batch_offset >= 0 ? values_->stride(-2) : 0;

  // 创建 cuSPARSE 稀疏矩阵描述符
  cusparseSpMatDescr_t raw_descriptor = nullptr;
  TORCH_CUDASPARSE_CHECK(cusparseCreateCsr(
      &raw_descriptor,  // 输出的描述符
      rows,             // 矩阵的行数
      cols,             // 矩阵的列数
      nnz,              // 矩阵的非零元素数目
      // 稀疏矩阵的行偏移，大小为 rows + 1
      static_cast<char*>(crow_indices.data_ptr()) +
          batch_offset * crow_indices_batch_stride * crow_indices.itemsize(),
      // 稀疏矩阵的列索引，大小为 nnz
      static_cast<char*>(col_indices.data_ptr()) +
          batch_offset * col_indices_batch_stride * col_indices.itemsize(),
      // 稀疏矩阵的值，大小为 nnz
      static_cast<char*>(values_->data_ptr()) +
          batch_offset * values_batch_stride * values_->itemsize(),
      index_type,       // 行偏移索引的数据类型
      index_type,       // 列索引的数据类型
      CUSPARSE_INDEX_BASE_ZERO,  // 行偏移和列索引的起始索引
      value_type        // 值的数据类型
      ));

  // 如果输入张量是三维的且批处理偏移为 -1
  if (ndim == 3 && batch_offset == -1) {
    int batch_count =
        at::native::cuda_int_cast(at::native::batchCount(input), "batch_count");
    if (crow_indices.dim() >= 2 || values.dim() >= 2 ||
        col_indices.dim() >= 2) {
      // cuSPARSE 忽略步长并仅使用第一个批次
      TORCH_INTERNAL_ASSERT(
          false,
          "Support for batched CSR indices and values is not implemented.");
      // 设置批处理模式下的 CSR 矩阵描述符，仅在调试模式下生效
      TORCH_CUDASPARSE_CHECK(cusparseCsrSetStridedBatch(
          raw_descriptor,
          batch_count,
          crow_indices.stride(-2),
          values_->stride(-2)));
    } else {
      // 如果不是第一个分支，执行以下操作：
      // cuSPARSE 允许在批次矩阵乘法中跨批次广播索引和值
      // 使用 cuSPARSE 库设置稀疏矩阵批次操作的描述符
      TORCH_CUDASPARSE_CHECK(
          cusparseCsrSetStridedBatch(raw_descriptor, batch_count, 0, 0));
    }
  }
  
  // 将原始描述符赋值给成员变量 descriptor_
  descriptor_.reset(raw_descriptor);
}

#endif // AT_USE_CUSPARSE_GENERIC_API() || AT_USE_HIPSPARSE_GENERIC_API()

} // namespace at::cuda::sparse
```