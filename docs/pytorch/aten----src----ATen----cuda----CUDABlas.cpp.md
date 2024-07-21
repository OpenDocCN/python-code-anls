# `.\pytorch\aten\src\ATen\cuda\CUDABlas.cpp`

```py
/*
  Provides the implementations of CUDA BLAS function templates.
 */

// 包含 ATen 库的头文件，用于张量操作和 CUDA BLAS
#include <ATen/ATen.h>
// 包含 CUDA 版本的 BLAS 函数定义
#include <ATen/cuda/CUDABlas.h>
// 包含 CUDA 异常处理相关的头文件
#include <ATen/cuda/Exceptions.h>
// 包含 CUDA 数据类型定义的头文件
#include <ATen/cuda/CUDADataType.h>
// 包含可调节参数的 CUDA 版本的头文件
#include <ATen/cuda/tunable/Tunable.h>
// 包含可调节参数的 CUDA 版本的 GEMM 实现的头文件
#include <ATen/cuda/tunable/TunableGemm.h>
// 包含 CUDA 内存分配器相关的头文件
#include <c10/cuda/CUDACachingAllocator.h>
// 包含 CUDA 基础函数的头文件
#include <c10/cuda/CUDAFunctions.h>
// 包含 C10 库的导出宏定义
#include <c10/macros/Export.h>
// 包含 C10 工具类中的迭代器定义
#include <c10/util/irange.h>

// 如果使用 ROCm 平台，则需要包含下面的头文件
#ifdef USE_ROCM
// 包含 ROCm 平台的 hipblaslt 扩展 API 的头文件
#include <hipblaslt/hipblaslt-ext.hpp>
// 因为 hipblas 没有直接接受 flags 的 API，所以需要调用 rocblas
#include <hipblas/hipblas.h>
#include <rocblas/rocblas.h>
// 定义一个宏，将 ROCBLAS 版本号转换为十进制数以进行比较
#define PYTORCH_ROCBLAS_VERSION_DECIMAL (ROCBLAS_VERSION_MAJOR * 100 + ROCBLAS_VERSION_MINOR)
// 如果 ROCBLAS 版本大于等于 2.42，则使用 GEMM flags 的备用实现
#define USE_GEMM_FLAGS_FP16_ALT_IMPL (PYTORCH_ROCBLAS_VERSION_DECIMAL >= 242)

// 以下是一些必要的函数和宏定义，用于处理 hipblas 和 rocblas 之间的转换和错误处理

// 将 hipblas 的操作类型转换为 rocblas 的操作类型
static rocblas_operation hipOperationToRocOperation(hipblasOperation_t op)
{
    switch(op)
    {
    case HIPBLAS_OP_N:
        return rocblas_operation_none;
    case HIPBLAS_OP_T:
        return rocblas_operation_transpose;
    case HIPBLAS_OP_C:
        return rocblas_operation_conjugate_transpose;
    }
    // 如果无法匹配任何操作类型，则抛出错误
    AT_ERROR("HIPBLAS_STATUS_INVALID_ENUM");
}

// 将 rocBLAS 的状态转换为 HIPBLAS 的状态
static hipblasStatus_t rocBLASStatusToHIPStatus(rocblas_status error)
{
    switch(error)
    {
    case rocblas_status_size_unchanged:
    case rocblas_status_size_increased:
    case rocblas_status_success:
        return HIPBLAS_STATUS_SUCCESS;
    case rocblas_status_invalid_handle:
        return HIPBLAS_STATUS_NOT_INITIALIZED;
    case rocblas_status_not_implemented:
        return HIPBLAS_STATUS_NOT_SUPPORTED;
    case rocblas_status_invalid_pointer:
    case rocblas_status_invalid_size:
    case rocblas_status_invalid_value:
        return HIPBLAS_STATUS_INVALID_VALUE;
    case rocblas_status_memory_error:
        return HIPBLAS_STATUS_ALLOC_FAILED;
    case rocblas_status_internal_error:
        return HIPBLAS_STATUS_INTERNAL_ERROR;
    }
    // 如果无法匹配任何状态，则抛出错误
    AT_ERROR("HIPBLAS_STATUS_INVALID_ENUM");
}

// 由于 hipblas 没有 hipblasSetMathMode 函数，定义一个宏以兼容性处理
#define hipblasSetMathMode(handle, flags) HIPBLAS_STATUS_SUCCESS

// 如果未使用 hipblas v2，则需要定义一些类型的宏，以兼容性处理
#ifndef HIPBLAS_V2
#define HIP_R_16F  HIPBLAS_R_16F
#define HIP_R_32F  HIPBLAS_R_32F
#define HIP_R_64F  HIPBLAS_R_64F
#define HIP_C_16F  HIPBLAS_C_16F
#define HIP_C_32F  HIPBLAS_C_32F
#define HIP_C_64F  HIPBLAS_C_64F
#define HIP_R_8I   HIPBLAS_R_8I
#define HIP_R_8U   HIPBLAS_R_8U
#define HIP_R_32I  HIPBLAS_R_32I
#define HIP_R_32U  HIPBLAS_R_32U
#define HIP_C_8I   HIPBLAS_C_8I
#define HIP_C_8U   HIPBLAS_C_8U
#define HIP_C_32I  HIPBLAS_C_32I
#define HIP_C_32U  HIPBLAS_C_32U
#define HIP_R_16BF HIPBLAS_R_16B
#define HIP_C_16BF HIPBLAS_C_16B
#endif

#endif
#define CUDABLAS_POSINT_CHECK(FD, X)         \
  TORCH_CHECK(                               \
      (X > 0 && X <= INT_MAX),               \  // 检查 X 是否为正数且不超过 INT_MAX
      "at::cuda::blas::" #FD " argument " #X \
      " must be positive and less than ",    \  // 错误信息部分：要求 X 必须为正数且小于指定值
      INT_MAX,                               \  // INT_MAX 的值
      " but got ",                           \  // 错误信息部分：显示实际传入的 X 的值
      X)

#define CUDABLAS_NONNEGINT_CHECK(FD, X)       \
  TORCH_CHECK(                                \
      (X >= 0 && X <= INT_MAX),               \  // 检查 X 是否为非负数且不超过 INT_MAX
      "at::cuda::blas::" #FD " argument " #X  \  // 错误信息部分：要求 X 必须为非负数且小于指定值
      " must be non-negative and less than ", \
      INT_MAX,                                \  // INT_MAX 的值
      " but got ",                            \  // 错误信息部分：显示实际传入的 X 的值
      X)

namespace {

static cublasOperation_t _cublasOpFromChar(char op) {
  switch (op) {
    case 'n':
    case 'N':
      return CUBLAS_OP_N;                    \  // 返回未转置的操作
    case 't':
    case 'T':
      return CUBLAS_OP_T;                    \  // 返回转置操作
    case 'c':
    case 'C':
      return CUBLAS_OP_C;                    \  // 返回共轭转置操作
  }
  AT_ERROR(                                  \
      "_cublasOpFromChar input should be 't', 'n' or 'c' but got `", op, "`");  // 错误处理：输入不是预期的字符
}

static void _cublasAdjustLdLevel2(int64_t m, int64_t n, int64_t* lda) {
  // Note: leading dimensions generally are checked that they are > 0
  // and at least as big the result requires (even if the value won't
  // be used).

  // Q: Why does Level3 check trans but this doesn't?
  // A: In level 2, the sizes (m, n) specify the size of A
  // (independent of trans value). In level 3. the sizes (m, n, k)
  // specify the sizes of op(A), op(B) where op depend on trans
  // values.
  if (n <= 1)
    *lda = std::max<int64_t>(m, 1);           // 调整 Level 2 的 leading dimension
}

static void _cublasAdjustLdLevel3(
    char transa,
    char transb,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t* lda,
    int64_t* ldb,
    int64_t* ldc) {
  bool transa_ = ((transa != 'n') && (transa != 'N'));
  bool transb_ = ((transb != 'n') && (transb != 'N'));

  // Note: leading dimensions generally are checked that they are > 0
  // and at least as big the result requires (even if the value won't
  // be used).
  if (n <= 1)
    *ldc = std::max<int64_t>(m, 1);           // 调整 Level 3 的 leading dimension

  if (transa_) {
    if (m <= 1)
      *lda = std::max<int64_t>(k, 1);         // 根据 transa 的值调整 leading dimension
  } else {
    if (k <= 1)
      *lda = std::max<int64_t>(m, 1);         // 根据 transa 的值调整 leading dimension
  }

  if (transb_) {
    if (k <= 1)
      *ldb = std::max<int64_t>(n, 1);         // 根据 transb 的值调整 leading dimension
  } else {
    if (n <= 1)
      *ldb = std::max<int64_t>(k, 1);         // 根据 transb 的值调整 leading dimension
  }
}

#ifndef USE_ROCM
uint32_t _getAlignment(uintptr_t address) {
  // alignment are in bytes
  uint32_t alignment = 256;
  for (; ; alignment /= 2) {
    if (!(address % alignment)) {
      return alignment;                      // 返回地址对齐的字节数
    }
  }
}
#endif

static size_t _parseChosenWorkspaceSize() {
  const char * val = getenv("CUBLASLT_WORKSPACE_SIZE");
#ifdef USE_ROCM
  if (!val) {
    // accept either env var
    val = getenv("HIPBLASLT_WORKSPACE_SIZE");
  }
#endif
  size_t workspace_size = 1024; /* default size in KiB according to #73328 */
  if (val) {
    try {
      workspace_size = std::stoi(val);        // 尝试将环境变量转换为整数，作为工作空间的大小
    } catch(std::invalid_argument const& e) {
      # 捕获 std::invalid_argument 异常，通常表示参数无效
      TORCH_WARN("invalid CUBLASLT_WORKSPACE_SIZE,", 
                 " using default workspace size of ", workspace_size, " KiB.");
      # 输出警告信息，指示 CUBLASLT_WORKSPACE_SIZE 无效，使用默认的工作空间大小（以 KiB 为单位）
    } catch(std::out_of_range const& e) {
      # 捕获 std::out_of_range 异常，通常表示值超出有效范围
      TORCH_WARN("CUBLASLT_WORKSPACE_SIZE out of range,", 
                 " using default workspace size of ", workspace_size, " KiB.");
      # 输出警告信息，指示 CUBLASLT_WORKSPACE_SIZE 超出范围，使用默认的工作空间大小（以 KiB 为单位）
    }
  }
  # 返回计算后的工作空间大小，以字节为单位
  return workspace_size * 1024;
} // anonymous namespace


namespace at::cuda::blas {


/* LEVEL 3 BLAS FUNCTIONS */


#define GEMM_CHECK_ARGVALUES(Dtype)           \
  do {                                        \
    CUDABLAS_NONNEGINT_CHECK(gemm<Dtype>, m); \
    CUDABLAS_NONNEGINT_CHECK(gemm<Dtype>, n); \
    CUDABLAS_NONNEGINT_CHECK(gemm<Dtype>, k); \
    CUDABLAS_POSINT_CHECK(gemm<Dtype>, lda);  \
    CUDABLAS_POSINT_CHECK(gemm<Dtype>, ldb);  \
    CUDABLAS_POSINT_CHECK(gemm<Dtype>, ldc);  \
  } while (0)


#define BGEMM_CHECK_ARGVALUES(Dtype)           \
  do {                                        \
    CUDABLAS_NONNEGINT_CHECK(bgemm<Dtype>, m); \
    CUDABLAS_NONNEGINT_CHECK(bgemm<Dtype>, n); \
    CUDABLAS_NONNEGINT_CHECK(bgemm<Dtype>, k); \
    CUDABLAS_POSINT_CHECK(bgemm<Dtype>, lda);  \
    CUDABLAS_POSINT_CHECK(bgemm<Dtype>, ldb);  \
    CUDABLAS_POSINT_CHECK(bgemm<Dtype>, ldc);  \
    CUDABLAS_NONNEGINT_CHECK(bgemm<Dtype>, num_batches);  \
  } while (0)


namespace {


// Following the pattern of CuSparseDescriptor
// Defined here for now because this is the only place cublas_lt interface is
// used but can be moved to a header once cublas_lt interface is used in
// multiple places.
template <typename T, cublasStatus_t (*destructor)(T*)>
struct CuBlasLtDeleter {
  void operator()(T* x) {
    if (x != nullptr) {
      TORCH_CUDABLAS_CHECK(destructor(x));
    }
  }
};


template <typename T, cublasStatus_t (*destructor)(T*)>
class CuBlasLtDescriptor {
 public:
  T* descriptor() const {
    return descriptor_.get();
  }
  T* descriptor() {
    return descriptor_.get();
  }

 protected:
  std::unique_ptr<T, CuBlasLtDeleter<T, destructor>> descriptor_;
};


class CuBlasLtMatmulDescriptor : public CuBlasLtDescriptor<
                                     cublasLtMatmulDescOpaque_t,
                                     &cublasLtMatmulDescDestroy> {
 public:
  CuBlasLtMatmulDescriptor(
      cublasComputeType_t compute_type,
      cudaDataType_t scale_type) {
    cublasLtMatmulDesc_t raw_descriptor = nullptr;
    TORCH_CUDABLAS_CHECK(
        cublasLtMatmulDescCreate(&raw_descriptor, compute_type, scale_type));
    descriptor_.reset(raw_descriptor);
  }
  template <typename T>
  inline void setAttribute(cublasLtMatmulDescAttributes_t attr, const T value) {
    TORCH_CUDABLAS_CHECK(::cublasLtMatmulDescSetAttribute(descriptor(), attr, &value, sizeof(T)));
  }
};


class CuBlasLtMatrixLayout : public CuBlasLtDescriptor<
                                 cublasLtMatrixLayoutOpaque_t,
                                 &cublasLtMatrixLayoutDestroy> {
 public:
  CuBlasLtMatrixLayout(
      cudaDataType_t type,
      uint64_t rows,
      uint64_t cols,
      int64_t ld,
      bool t = false) {
    cublasLtMatrixLayout_t raw_descriptor = nullptr;
    TORCH_CUDABLAS_CHECK(
        cublasLtMatrixLayoutCreate(&raw_descriptor, type, t ? cols : rows, t ? rows : cols, ld));
    # 调用 TORCH_CUDABLAS_CHECK 宏，创建 CUBLAS LT 矩阵布局对象并初始化
    descriptor_.reset(raw_descriptor);
  }
  template <typename T>
  inline void setAttribute(cublasLtMatrixLayoutAttribute_t attr, const T value) {
    # 设置矩阵布局对象的属性
    TORCH_CUDABLAS_CHECK(::cublasLtMatrixLayoutSetAttribute(descriptor(), attr, &value, sizeof(T)));
  }
};

// CuBlasLtMatmulPreference 类定义，继承自 CuBlasLtDescriptor
// 使用 CuBlasLtMatmulPreferenceOpaque_t 类型和 cublasLtMatmulPreferenceDestroy 函数进行描述
class CuBlasLtMatmulPreference : public CuBlasLtDescriptor<
                                     cublasLtMatmulPreferenceOpaque_t,
                                     &cublasLtMatmulPreferenceDestroy> {
 public:
  // 构造函数，初始化 cublasLtMatmulPreference_t 对象
  CuBlasLtMatmulPreference() {
    cublasLtMatmulPreference_t raw_descriptor = nullptr;
    // 创建 cublasLtMatmulPreference 对象
    TORCH_CUDABLAS_CHECK(cublasLtMatmulPreferenceCreate(&raw_descriptor));
    descriptor_.reset(raw_descriptor);
  }

  // 设置属性模板函数
  template <typename T>
  inline void setAttribute(cublasLtMatmulPreferenceAttributes_t attr, const T value) {
    // 设置 cublasLtMatmulPreference 对象的属性
    TORCH_CUDABLAS_CHECK(::cublasLtMatmulPreferenceSetAttribute(descriptor(), attr, &value, sizeof(T)));
  }
};
} // namespace

// Dtype 类型模板函数 bgemm_internal_cublaslt
template <typename Dtype>
inline void bgemm_internal_cublaslt(CUDABLAS_BGEMM_ARGTYPES(Dtype)) {
  // 默认使用 CUDA_R_32F 类型
  cudaDataType_t abcType = CUDA_R_32F;
  cublasComputeType_t computeType = CUBLAS_COMPUTE_32F;
  cudaDataType_t scaleType = CUDA_R_32F;

  // 根据 Dtype 类型设置不同的数据类型和计算类型
  if constexpr (std::is_same_v<Dtype, double>) {
    abcType = CUDA_R_64F;
    computeType = CUBLAS_COMPUTE_64F;
    scaleType = CUDA_R_64F;
  } else if constexpr (std::is_same_v<Dtype, float>) {
    // 如果 Dtype 是 float 类型，并且允许 TF32 加速，则使用 CUBLAS_COMPUTE_32F_FAST_TF32
#ifndef USE_ROCM
    if (at::globalContext().allowTF32CuBLAS()) {
      computeType = CUBLAS_COMPUTE_32F_FAST_TF32;
    }
#endif
  } else if constexpr (std::is_same_v<Dtype, c10::complex<double>>) {
    abcType = CUDA_C_64F;
    computeType = CUBLAS_COMPUTE_64F;
    scaleType = CUDA_C_64F;
  } else if constexpr (std::is_same_v<Dtype, c10::complex<float>>) {
    abcType = CUDA_C_32F;
    scaleType = CUDA_C_32F;
  } else if constexpr (std::is_same_v<Dtype, at::Half>) {
    abcType = CUDA_R_16F;
  } else if constexpr (std::is_same_v<Dtype, at::BFloat16>) {
    abcType = CUDA_R_16BF;
  } else {
    // 如果 Dtype 类型未被实现，抛出错误
    AT_ERROR("at::cuda::blas::bgemm_internal_cublaslt: not implemented for ", typeid(Dtype).name());
  }

  // 提示 CuBLAS 配置可能不确定性
  globalContext().alertCuBLASConfigNotDeterministic();

  // 获取当前 CUDA CuBLAS Lt 句柄
  cublasLtHandle_t ltHandle = at::cuda::getCurrentCUDABlasLtHandle();

  // 根据 transa 和 transb 转换为 cublasOperation_t 类型
  cublasOperation_t opa = _cublasOpFromChar(transa);
  cublasOperation_t opb = _cublasOpFromChar(transb);

  // 调整 m, n, k 的 leading dimensions
  _cublasAdjustLdLevel3(transa, transb, m, n, k, &lda, &ldb, &ldc);

  // 创建 CuBlasLtMatmulDescriptor 对象并设置属性
  CuBlasLtMatmulDescriptor computeDesc(computeType, scaleType);
  computeDesc.setAttribute(CUBLASLT_MATMUL_DESC_TRANSA, opa);
  computeDesc.setAttribute(CUBLASLT_MATMUL_DESC_TRANSB, opb);

  // 创建 CuBlasLtMatrixLayout 对象 Adesc, Bdesc, Cdesc，并设置相关属性
  CuBlasLtMatrixLayout Adesc(abcType, m, k, lda, opa == CUBLAS_OP_T);
  CuBlasLtMatrixLayout Bdesc(abcType, k, n, ldb, opb == CUBLAS_OP_T);
  CuBlasLtMatrixLayout Cdesc(abcType, m, n, ldc);

  // 如果有多个批次，设置批次相关属性
  if (num_batches > 1) {
    int num_batches_as_int = static_cast<int>(num_batches);
    Adesc.setAttribute(CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, num_batches_as_int);
    Bdesc.setAttribute(CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, num_batches_as_int);
    Cdesc.setAttribute(CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, num_batches_as_int);
    Adesc.setAttribute(CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, stridea);
    Bdesc.setAttribute(CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, strideb);
    Cdesc.setAttribute(CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, stridec);
  }


# 设置 CuBLASLt 矩阵描述符的偏移量布局属性
Cdesc.setAttribute(CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, stridec);



  CuBlasLtMatmulPreference preference;
  // See https://github.com/pytorch/pytorch/issues/73328 for reasoning behind
  // setting this to 1M.
  size_t workspaceSize = _getWorkspaceSize();
  preference.setAttribute(CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, workspaceSize);


# 创建 CuBLASLt 矩阵乘法偏好设置对象
CuBlasLtMatmulPreference preference;
# 设置最大工作空间大小为 _getWorkspaceSize() 返回的值，详细理由见
# https://github.com/pytorch/pytorch/issues/73328
size_t workspaceSize = _getWorkspaceSize();
preference.setAttribute(CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, workspaceSize);
#ifndef USE_ROCM
  // 如果未定义 USE_ROCM，则获取指针 a、b、c 的对齐要求并设置到变量中
  uint32_t a_alignment = _getAlignment(reinterpret_cast<uintptr_t>(a));
  uint32_t b_alignment = _getAlignment(reinterpret_cast<uintptr_t>(b));
  uint32_t c_alignment = _getAlignment(reinterpret_cast<uintptr_t>(c));
  // 设置 CUBLAS LT 矩阵乘法的最小对齐要求
  preference.setAttribute(CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, a_alignment);
  preference.setAttribute(CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES, b_alignment);
  preference.setAttribute(CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, c_alignment);
#endif

  // 获取 CUDA 的内存分配器，并为工作空间分配内存
  auto& allocator = *::c10::cuda::CUDACachingAllocator::get();
  auto workspace = allocator.allocate(workspaceSize);
  // 检查分配工作空间是否成功
  TORCH_CHECK(workspace.get() != nullptr, "OOM trying to allocate workspace for cublaslt");

  // 进行 CUBLAS LT 矩阵乘法的启发式选择
  cublasLtMatmulHeuristicResult_t heuristicResult = {};
  int returnedResult = 0;
  // 调用 CUBLAS LT 获取启发式算法
  TORCH_CUDABLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(
      ltHandle,
      computeDesc.descriptor(),
      Adesc.descriptor(),
      Bdesc.descriptor(),
      Cdesc.descriptor(),
      Cdesc.descriptor(),
      preference.descriptor(),
      1,
      &heuristicResult,
      &returnedResult));
  // 检查返回结果，若为 0 则表示不支持该配置
  if (returnedResult == 0) {
    TORCH_CUDABLAS_CHECK(CUBLAS_STATUS_NOT_SUPPORTED);
  }

  // 调用 CUBLAS LT 进行矩阵乘法运算
  cublasStatus_t cublasStatus = cublasLtMatmul(
      ltHandle,
      computeDesc.descriptor(),
      &alpha,
      a,
      Adesc.descriptor(),
      b,
      Bdesc.descriptor(),
      &beta,
      c,
      Cdesc.descriptor(),
      c,
      Cdesc.descriptor(),
      &heuristicResult.algo,
      workspace.mutable_get(),
      workspaceSize,
      at::cuda::getCurrentCUDAStream());
  // 检查 CUBLAS 操作是否成功，并输出详细的错误信息
  TORCH_CHECK(
      cublasStatus == CUBLAS_STATUS_SUCCESS,
      "CUDA error: ",
      at::cuda::blas::_cublasGetErrorEnum(cublasStatus),
      " when calling cublasLtMatmul with transpose_mat1 ",
      (opa == CUBLAS_OP_T),
      " transpose_mat2 ",
      (opb == CUBLAS_OP_T),
      " m ",
      m,
      " n ",
      n,
      " k ",
      k,
      " lda ",
      lda,
      " ldb ",
      ldb,
      " ldc ",
      ldc,
      " abcType ",
      abcType,
      " computeType ",
      computeType,
      " scaleType ",
      scaleType);
}


template <typename Dtype>
inline void bgemm_internal_cublas(CUDABLAS_BGEMM_ARGTYPES(Dtype)) {
  // 对于未实现的类型，抛出错误信息
  AT_ERROR("at::cuda::blas::bgemm_internal_cublas: not implemented for ", typeid(Dtype).name());
}

template <>
void bgemm_internal_cublas<double>(CUDABLAS_BGEMM_ARGTYPES(double)) {
  // 参见文档中的 [Writing Nondeterministic Operations]，警告 CuBLAS 配置可能是非确定性的
  globalContext().alertCuBLASConfigNotDeterministic();
  // 获取当前的 CuBLAS 句柄
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  // 根据输入参数设置 CuBLAS 操作的转置情况
  cublasOperation_t opa = _cublasOpFromChar(transa);
  cublasOperation_t opb = _cublasOpFromChar(transb);
  // 调整输入矩阵的 leading dimension
  _cublasAdjustLdLevel3(transa, transb, m, n, k, &lda, &ldb, &ldc);
  // 检查输入参数的有效性
  BGEMM_CHECK_ARGVALUES(double);
  // 调用 CuBLAS DGEMM Strided Batched 函数进行矩阵乘法运算
  TORCH_CUDABLAS_CHECK(cublasDgemmStridedBatched(
      handle, opa, opb, m, n, k, &alpha, a, lda, stridea, b, ldb, strideb, &beta, c, ldc, stridec, num_batches));
}
void bgemm_internal_cublas<float>(CUDABLAS_BGEMM_ARGTYPES(float)) {
  // 根据注意事项 [写非确定性操作]，警告全局上下文的CuBLAS配置是非确定性的
  globalContext().alertCuBLASConfigNotDeterministic();
  // 获取当前CUDA CuBLAS句柄
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  // 根据转置参数transa获取对应的CuBLAS操作类型
  cublasOperation_t opa = _cublasOpFromChar(transa);
  // 根据转置参数transb获取对应的CuBLAS操作类型
  cublasOperation_t opb = _cublasOpFromChar(transb);
  // 调整m, n, k的leading dimensions（步长）为Level 3的参数，可能会修改lda, ldb, ldc
  _cublasAdjustLdLevel3(transa, transb, m, n, k, &lda, &ldb, &ldc);
  // 检查参数值是否符合BGEMM的要求
  BGEMM_CHECK_ARGVALUES(float);
  // 调用CuBLAS进行单精度浮点数的批量strided矩阵乘法
  TORCH_CUDABLAS_CHECK(cublasSgemmStridedBatched(
      handle, opa, opb, m, n, k, &alpha, a, lda, stridea, b, ldb, strideb, &beta, c, ldc, stridec, num_batches));
}

template <>
void bgemm_internal_cublas<c10::complex<double>>(CUDABLAS_BGEMM_ARGTYPES(c10::complex<double>)) {
  // See Note [Writing Nondeterministic Operations]
  globalContext().alertCuBLASConfigNotDeterministic();
  // 获取当前CUDA CuBLAS句柄
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  // 根据转置参数transa获取对应的CuBLAS操作类型
  cublasOperation_t opa = _cublasOpFromChar(transa);
  // 根据转置参数transb获取对应的CuBLAS操作类型
  cublasOperation_t opb = _cublasOpFromChar(transb);
  // 调整m, n, k的leading dimensions（步长）为Level 3的参数，可能会修改lda, ldb, ldc
  _cublasAdjustLdLevel3(transa, transb, m, n, k, &lda, &ldb, &ldc);
  // 检查参数值是否符合BGEMM的要求
  BGEMM_CHECK_ARGVALUES(c10::complex<double>);
  // 调用CuBLAS进行双精度复数的批量strided矩阵乘法
  TORCH_CUDABLAS_CHECK(cublasZgemmStridedBatched(
      handle, opa, opb, m, n, k, reinterpret_cast<const cuDoubleComplex*>(&alpha), reinterpret_cast<const cuDoubleComplex*>(a),
      lda, stridea, reinterpret_cast<const cuDoubleComplex*>(b), ldb, strideb, reinterpret_cast<const cuDoubleComplex*>(&beta),
      reinterpret_cast<cuDoubleComplex*>(c), ldc, stridec, num_batches));
}

template <>
void bgemm_internal_cublas<c10::complex<float>>(CUDABLAS_BGEMM_ARGTYPES(c10::complex<float>)) {
  // See Note [Writing Nondeterministic Operations]
  globalContext().alertCuBLASConfigNotDeterministic();
  // 获取当前CUDA CuBLAS句柄
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  // 根据转置参数transa获取对应的CuBLAS操作类型
  cublasOperation_t opa = _cublasOpFromChar(transa);
  // 根据转置参数transb获取对应的CuBLAS操作类型
  cublasOperation_t opb = _cublasOpFromChar(transb);
  // 调整m, n, k的leading dimensions（步长）为Level 3的参数，可能会修改lda, ldb, ldc
  _cublasAdjustLdLevel3(transa, transb, m, n, k, &lda, &ldb, &ldc);
  // 检查参数值是否符合BGEMM的要求
  BGEMM_CHECK_ARGVALUES(c10::complex<float>);
  // 调用CuBLAS进行单精度复数的批量strided矩阵乘法
  TORCH_CUDABLAS_CHECK(cublasCgemmStridedBatched(
      handle, opa, opb, m, n, k, reinterpret_cast<const cuComplex*>(&alpha), reinterpret_cast<const cuComplex*>(a),
      lda, stridea, reinterpret_cast<const cuComplex*>(b), ldb, strideb, reinterpret_cast<const cuComplex*>(&beta),
      reinterpret_cast<cuComplex*>(c), ldc, stridec, num_batches));
}

template <>
void bgemm_internal_cublas<at::Half>(CUDABLAS_BGEMM_ARGTYPES(at::Half)) {
  // See Note [Writing Nondeterministic Operations]
  globalContext().alertCuBLASConfigNotDeterministic();
  // 获取当前CUDA CuBLAS句柄
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  // 根据转置参数transa获取对应的CuBLAS操作类型
  cublasOperation_t opa = _cublasOpFromChar(transa);
  // 根据转置参数transb获取对应的CuBLAS操作类型
  cublasOperation_t opb = _cublasOpFromChar(transb);
  // 调整m, n, k的leading dimensions（步长）为Level 3的参数，可能会修改lda, ldb, ldc
  _cublasAdjustLdLevel3(transa, transb, m, n, k, &lda, &ldb, &ldc);
  // 检查参数值是否符合BGEMM的要求
  BGEMM_CHECK_ARGVALUES(at::Half);
  // 将alpha和beta值转换为单精度浮点数
  float falpha = alpha;
  float fbeta = beta;
#ifdef USE_ROCM
  int flag = 0;
#if USE_GEMM_FLAGS_FP16_ALT_IMPL
  // 如果启用了 FP16 备选实现的标志，则根据当前是否为后向传播过程设置相应的标志
  flag = at::ROCmBackwardPassGuard::is_backward_pass() ? rocblas_gemm_flags_fp16_alt_impl : 0;
#endif

  // 调用 rocBLAS 进行批量 strided 的混合精度（FP16）通用矩阵乘法运算
  TORCH_CUDABLAS_CHECK(rocBLASStatusToHIPStatus(rocblas_gemm_strided_batched_ex((rocblas_handle)handle,
                                   hipOperationToRocOperation(opa),
                                   hipOperationToRocOperation(opb), (int)m, (int)n, (int)k,
                                   (void*)&falpha, a, rocblas_datatype_f16_r, (int)lda, stridea,
                                   b, rocblas_datatype_f16_r, (int)ldb, strideb,
                                   (void*)&fbeta, c, rocblas_datatype_f16_r, (int)ldc, stridec,
                                   c, rocblas_datatype_f16_r, (int)ldc, stridec,
                                   (int) num_batches, rocblas_datatype_f32_r, rocblas_gemm_algo_standard,
                                   0, flag)));

#else
  // 获取当前 CUDA 设备的属性
  cudaDeviceProp* prop = at::cuda::getCurrentDeviceProperties();

  // 如果当前设备的主版本号大于等于 5，使用 cublasGemmStridedBatchedEx 进行 FP16 混合精度的矩阵乘法
  if (prop->major >= 5){
    TORCH_CUDABLAS_CHECK(cublasGemmStridedBatchedEx(
      handle, opa, opb, m, n, k,
      (void*)(&falpha), a, CUDA_R_16F, lda, stridea,
      b, CUDA_R_16F, ldb, strideb, (void*)(&fbeta),
      c, CUDA_R_16F, ldc, stridec,
      num_batches, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  } else {
    // 对于主版本号小于 5 的设备，使用循环调用 at::cuda::blas::gemm<at::Half> 进行 FP16 混合精度的矩阵乘法
    for (const auto i : c10::irange(num_batches)) {
      at::cuda::blas::gemm<at::Half>(
        transa, transb,
        m, n, k,
        alpha, (a + i * stridea), lda,
        (b + i * strideb), ldb, beta,
        (c + i * stridec), ldc);
    }
  }
#endif // USE_ROCM
}

template <>
void bgemm_internal_cublas<at::BFloat16>(CUDABLAS_BGEMM_ARGTYPES(at::BFloat16)) {
  // 发出警告，表示正在执行的 CuBLAS 操作不是确定性的，参见 Note [Writing Nondeterministic Operations]
  globalContext().alertCuBLASConfigNotDeterministic();
  BGEMM_CHECK_ARGVALUES(at::BFloat16);
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  cublasOperation_t opa = _cublasOpFromChar(transa);
  cublasOperation_t opb = _cublasOpFromChar(transb);
  const float falpha = alpha;
  const float fbeta = beta;
  _cublasAdjustLdLevel3(transa, transb, m, n, k, &lda, &ldb, &ldc);

#if defined(USE_ROCM)
  // 在 ROCm 环境下，使用 32 位浮点数进行计算
  auto compute_type = CUBLAS_COMPUTE_32F;
#else
  // 在 CUDA 环境下，使用 32 位浮点数进行计算
  auto compute_type = CUDA_R_32F;
#endif

  // 调用 cublasGemmStridedBatchedEx 进行批量 strided 的 BFloat16 混合精度矩阵乘法
  TORCH_CUDABLAS_CHECK(cublasGemmStridedBatchedEx(handle,
                                  opa, opb, (int)m, (int)n, (int)k,
                                  (void*)&falpha, a, CUDA_R_16BF, (int)lda, stridea,
                                  b, CUDA_R_16BF, (int)ldb, strideb,
                                  (void*)&fbeta, c, CUDA_R_16BF, (int)ldc, stridec,
                                  (int)num_batches,
                                  compute_type,
                                  CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

template <>
void bgemm_internal<double>(CUDABLAS_BGEMM_ARGTYPES(double))
{
  // 如果使用 CuBLASlt 作为首选后端，发出警告，因为 hipblaslt 目前不支持双精度 gemm 操作
  if (at::globalContext().blasPreferredBackend() == BlasBackend::Cublaslt) {
#ifdef USE_ROCM
    // hipblaslt 目前不支持双精度 gemm 操作
    # 调用 bgemm_internal_cublas 模板函数，使用 double 类型作为模板参数，并传递 CUDABLAS_BGEMM_ARGS(double) 宏展开后的参数列表作为参数
    bgemm_internal_cublas<double>(CUDABLAS_BGEMM_ARGS(double));
#else
    // 如果未定义 USE_ROCM，则使用 bgemm_internal_cublaslt<double> 函数处理双精度浮点数的矩阵乘法
    bgemm_internal_cublaslt<double>(CUDABLAS_BGEMM_ARGS(double));
#endif
  }
  else {
    // 否则，使用 bgemm_internal_cublas<double> 函数处理双精度浮点数的矩阵乘法
    bgemm_internal_cublas<double>(CUDABLAS_BGEMM_ARGS(double));
  }
}

template <>
void bgemm_internal<float>(CUDABLAS_BGEMM_ARGTYPES(float))
{
  // 如果当前使用的 BLAS 后端为 Cublaslt
  if (at::globalContext().blasPreferredBackend() == BlasBackend::Cublaslt) {
    // 调用 bgemm_internal_cublaslt<float> 处理单精度浮点数的矩阵乘法
    bgemm_internal_cublaslt<float>(CUDABLAS_BGEMM_ARGS(float));
  }
  else {
    // 否则，调用 bgemm_internal_cublas<float> 处理单精度浮点数的矩阵乘法
    bgemm_internal_cublas<float>(CUDABLAS_BGEMM_ARGS(float));
  }
}

template <>
void bgemm_internal<c10::complex<double>>(CUDABLAS_BGEMM_ARGTYPES(c10::complex<double>))
{
  // 如果当前使用的 BLAS 后端为 Cublaslt
  if (at::globalContext().blasPreferredBackend() == BlasBackend::Cublaslt) {
#ifdef USE_ROCM
    // 如果定义了 USE_ROCM，则打印注释说明 hipblaslt 目前不支持 complex<double> 的矩阵乘法
    // 否则，调用 bgemm_internal_cublas<c10::complex<double>> 处理 complex<double> 的矩阵乘法
    bgemm_internal_cublas<c10::complex<double>>(CUDABLAS_BGEMM_ARGS(c10::complex<double>));
#else
    // 否则，调用 bgemm_internal_cublaslt<c10::complex<double>> 处理 complex<double> 的矩阵乘法
    bgemm_internal_cublaslt<c10::complex<double>>(CUDABLAS_BGEMM_ARGS(c10::complex<double>));
#endif
  }
  else {
    // 否则，调用 bgemm_internal_cublas<c10::complex<double>> 处理 complex<double> 的矩阵乘法
    bgemm_internal_cublas<c10::complex<double>>(CUDABLAS_BGEMM_ARGS(c10::complex<double>));
  }
}

template <>
void bgemm_internal<c10::complex<float>>(CUDABLAS_BGEMM_ARGTYPES(c10::complex<float>))
{
  // 如果当前使用的 BLAS 后端为 Cublaslt
  if (at::globalContext().blasPreferredBackend() == BlasBackend::Cublaslt) {
#ifdef USE_ROCM
    // 如果定义了 USE_ROCM，则打印注释说明 hipblaslt 目前不支持 complex<float> 的矩阵乘法
    // 否则，调用 bgemm_internal_cublas<c10::complex<float>> 处理 complex<float> 的矩阵乘法
    bgemm_internal_cublas<c10::complex<float>>(CUDABLAS_BGEMM_ARGS(c10::complex<float>));
#else
    // 否则，调用 bgemm_internal_cublaslt<c10::complex<float>> 处理 complex<float> 的矩阵乘法
    bgemm_internal_cublaslt<c10::complex<float>>(CUDABLAS_BGEMM_ARGS(c10::complex<float>));
#endif
  }
  else {
    // 否则，调用 bgemm_internal_cublas<c10::complex<float>> 处理 complex<float> 的矩阵乘法
    bgemm_internal_cublas<c10::complex<float>>(CUDABLAS_BGEMM_ARGS(c10::complex<float>));
  }
}

template <>
void bgemm_internal<at::Half>(CUDABLAS_BGEMM_ARGTYPES(at::Half))
{
  // 如果当前使用的 BLAS 后端为 Cublaslt
  if (at::globalContext().blasPreferredBackend() == BlasBackend::Cublaslt) {
    // 调用 bgemm_internal_cublaslt<at::Half> 处理 Half 精度浮点数的矩阵乘法
    bgemm_internal_cublaslt<at::Half>(CUDABLAS_BGEMM_ARGS(at::Half));
  }
  else {
    // 否则，调用 bgemm_internal_cublas<at::Half> 处理 Half 精度浮点数的矩阵乘法
    bgemm_internal_cublas<at::Half>(CUDABLAS_BGEMM_ARGS(at::Half));
  }
}

template <>
void bgemm_internal<at::BFloat16>(CUDABLAS_BGEMM_ARGTYPES(at::BFloat16))
{
  // 如果当前使用的 BLAS 后端为 Cublaslt
  if (at::globalContext().blasPreferredBackend() == BlasBackend::Cublaslt) {
    // 调用 bgemm_internal_cublaslt<at::BFloat16> 处理 BFloat16 类型的矩阵乘法
    bgemm_internal_cublaslt<at::BFloat16>(CUDABLAS_BGEMM_ARGS(at::BFloat16));
  }
  else {
    // 否则，调用 bgemm_internal_cublas<at::BFloat16> 处理 BFloat16 类型的矩阵乘法
    bgemm_internal_cublas<at::BFloat16>(CUDABLAS_BGEMM_ARGS(at::BFloat16));
  }
}

template <typename DType>
inline void bgemm_tunable(CUDABLAS_BGEMM_ARGTYPES(DType)) {
  // 创建参数对象 params 并初始化各个参数
  tunable::GemmStridedBatchedParams<DType> params;
  params.transa = transa;
  params.transb = transb;
  params.m = m;
  params.n = n;
  params.k = k;
  params.alpha = alpha;
  params.a = a;
  params.lda = lda;
  params.stride_a = stridea;
  params.b = b;
  params.ldb = ldb;
  params.stride_b = strideb;
  params.beta = beta;
  params.c = c;
  params.ldc = ldc;
  params.stride_c = stridec;
  params.batch = num_batches;

  // 检查是否需要转置操作
  bool transa_ = ((transa != 'n') && (transa != 'N'));
  bool transb_ = ((transb != 'n') && (transb != 'N'));

  // 如果需要同时对 A 和 B 进行转置
  if (transa_ && transb_) {
    // 创建适用于可调节的 GemmStridedBatchedTunableOp 操作符 bgemm
    static tunable::GemmStridedBatchedTunableOp<DType, tunable::BlasOp::T, tunable::BlasOp::T> bgemm{};
    # 调用适用于不同矩阵转置状态的通用矩阵乘法函数

  }
  else if (transa_ && !transb_) {
    # 定义静态变量 bgemm，使用适用于转置A但不转置B的参数配置
    static tunable::GemmStridedBatchedTunableOp<DType, tunable::BlasOp::T, tunable::BlasOp::N> bgemm{};
    # 调用 bgemm 处理参数
    bgemm(&params);
  }
  else if (!transa_ && transb_) {
    # 定义静态变量 bgemm，使用适用于不转置A但转置B的参数配置
    static tunable::GemmStridedBatchedTunableOp<DType, tunable::BlasOp::N, tunable::BlasOp::T> bgemm{};
    # 调用 bgemm 处理参数
    bgemm(&params);
  }
  else if (!transa_ && !transb_) {
    # 定义静态变量 bgemm，使用适用于不转置A且不转置B的参数配置
    static tunable::GemmStridedBatchedTunableOp<DType, tunable::BlasOp::N, tunable::BlasOp::N> bgemm{};
    # 调用 bgemm 处理参数
    bgemm(&params);
  }
  else {
    # 如果不满足任何前面的条件，抛出不可达错误
    TORCH_CHECK(false, "unreachable");
  }
// 定义模板特化，用于双精度数据类型的批量矩阵乘法操作
template <>
void bgemm<double>(CUDABLAS_BGEMM_ARGTYPES(double)) {
  // 获取调优上下文对象
  auto tuning_ctx = at::cuda::tunable::getTuningContext();
  // 检查是否启用可调优操作
  if (tuning_ctx->IsTunableOpEnabled()) {
    // 若启用，调用可调优的双精度矩阵乘法实现
    bgemm_tunable<double>(CUDABLAS_BGEMM_ARGS(double));
  }
  else {
    // 若未启用，调用内部实现的双精度矩阵乘法
    bgemm_internal<double>(CUDABLAS_BGEMM_ARGS(double));
  }
}

// 定义模板特化，用于单精度数据类型的批量矩阵乘法操作
template <>
void bgemm<float>(CUDABLAS_BGEMM_ARGTYPES(float)) {
  // 获取调优上下文对象
  auto tuning_ctx = at::cuda::tunable::getTuningContext();
  // 检查是否启用可调优操作
  if (tuning_ctx->IsTunableOpEnabled()) {
    // 若启用，调用可调优的单精度矩阵乘法实现
    bgemm_tunable<float>(CUDABLAS_BGEMM_ARGS(float));
  }
  else {
    // 若未启用，调用内部实现的单精度矩阵乘法
    bgemm_internal<float>(CUDABLAS_BGEMM_ARGS(float));
  }
}

// 定义模板特化，用于双精度复数数据类型的批量矩阵乘法操作
template <>
void bgemm<c10::complex<double>>(CUDABLAS_BGEMM_ARGTYPES(c10::complex<double>)) {
  // 获取调优上下文对象
  auto tuning_ctx = at::cuda::tunable::getTuningContext();
  // 检查是否启用可调优操作
  if (tuning_ctx->IsTunableOpEnabled()) {
    // 若启用，调用可调优的双精度复数矩阵乘法实现
    bgemm_tunable<c10::complex<double>>(CUDABLAS_BGEMM_ARGS(c10::complex<double>));
  }
  else {
    // 若未启用，调用内部实现的双精度复数矩阵乘法
    bgemm_internal<c10::complex<double>>(CUDABLAS_BGEMM_ARGS(c10::complex<double>));
  }
}

// 定义模板特化，用于单精度复数数据类型的批量矩阵乘法操作
template <>
void bgemm<c10::complex<float>>(CUDABLAS_BGEMM_ARGTYPES(c10::complex<float>)) {
  // 获取调优上下文对象
  auto tuning_ctx = at::cuda::tunable::getTuningContext();
  // 检查是否启用可调优操作
  if (tuning_ctx->IsTunableOpEnabled()) {
    // 若启用，调用可调优的单精度复数矩阵乘法实现
    bgemm_tunable<c10::complex<float>>(CUDABLAS_BGEMM_ARGS(c10::complex<float>));
  }
  else {
    // 若未启用，调用内部实现的单精度复数矩阵乘法
    bgemm_internal<c10::complex<float>>(CUDABLAS_BGEMM_ARGS(c10::complex<float>));
  }
}

// 定义模板特化，用于Half数据类型的批量矩阵乘法操作
template <>
void bgemm<at::Half>(CUDABLAS_BGEMM_ARGTYPES(at::Half)) {
  // 获取调优上下文对象
  auto tuning_ctx = at::cuda::tunable::getTuningContext();
  // 检查是否启用可调优操作
  if (tuning_ctx->IsTunableOpEnabled()) {
    // 若启用，调用可调优的Half精度矩阵乘法实现
    bgemm_tunable<at::Half>(CUDABLAS_BGEMM_ARGS(at::Half));
  }
  else {
    // 若未启用，调用内部实现的Half精度矩阵乘法
    bgemm_internal<at::Half>(CUDABLAS_BGEMM_ARGS(at::Half));
  }
}

// 定义模板特化，用于BFloat16数据类型的批量矩阵乘法操作
template <>
void bgemm<at::BFloat16>(CUDABLAS_BGEMM_ARGTYPES(at::BFloat16)) {
  // 获取调优上下文对象
  auto tuning_ctx = at::cuda::tunable::getTuningContext();
  // 检查是否启用可调优操作
  if (tuning_ctx->IsTunableOpEnabled()) {
    // 若启用，调用可调优的BFloat16精度矩阵乘法实现
    bgemm_tunable<at::BFloat16>(CUDABLAS_BGEMM_ARGS(at::BFloat16));
  }
  else {
    // 若未启用，调用内部实现的BFloat16精度矩阵乘法
    bgemm_internal<at::BFloat16>(CUDABLAS_BGEMM_ARGS(at::BFloat16));
  }
}

// 定义模板函数，用于通用的GEMM操作，但只是简单报错，未实现具体的内部函数
template <typename Dtype>
inline void gemm_internal_cublas(CUDABLAS_GEMM_ARGTYPES(Dtype)) {
  AT_ERROR("at::cuda::blas::gemm_internal_cublas: not implemented for ", typeid(Dtype).name());
}
void gemm_internal_cublas<double>(CUDABLAS_GEMM_ARGTYPES(double)) {
  // 在进行不确定性操作时发出警告，参见注释 [编写不确定性操作]
  globalContext().alertCuBLASConfigNotDeterministic();
  // 获取当前 CUDA CuBLAS 句柄
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  // 根据字符转换得到 CuBLAS 操作类型
  cublasOperation_t opa = _cublasOpFromChar(transa);
  cublasOperation_t opb = _cublasOpFromChar(transb);
  // 调整矩阵的 leading dimensions（LD3）以匹配 CuBLAS 需求
  _cublasAdjustLdLevel3(transa, transb, m, n, k, &lda, &ldb, &ldc);
  // 检查参数值，确保它们在有效范围内
  GEMM_CHECK_ARGVALUES(double);
  // 调用 CuBLAS 的双精度矩阵乘法函数
  TORCH_CUDABLAS_CHECK(cublasDgemm(
      handle, opa, opb, m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc));
}

template <>
void gemm_internal_cublas<float>(CUDABLAS_GEMM_ARGTYPES(float)) {
  // 在进行不确定性操作时发出警告，参见注释 [编写不确定性操作]
  globalContext().alertCuBLASConfigNotDeterministic();
  // 获取当前 CUDA CuBLAS 句柄
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  // 根据字符转换得到 CuBLAS 操作类型
  cublasOperation_t opa = _cublasOpFromChar(transa);
  cublasOperation_t opb = _cublasOpFromChar(transb);
  // 调整矩阵的 leading dimensions（LD3）以匹配 CuBLAS 需求
  _cublasAdjustLdLevel3(transa, transb, m, n, k, &lda, &ldb, &ldc);
  // 检查参数值，确保它们在有效范围内
  GEMM_CHECK_ARGVALUES(float);
  // 调用 CuBLAS 的单精度矩阵乘法函数
  TORCH_CUDABLAS_CHECK(cublasSgemm(
      handle, opa, opb, m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc));
}

template <>
void gemm_internal_cublas<c10::complex<double>>(CUDABLAS_GEMM_ARGTYPES(c10::complex<double>)) {
  // 在进行不确定性操作时发出警告，参见注释 [编写不确定性操作]
  globalContext().alertCuBLASConfigNotDeterministic();
  // 获取当前 CUDA CuBLAS 句柄
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  // 根据字符转换得到 CuBLAS 操作类型
  cublasOperation_t opa = _cublasOpFromChar(transa);
  cublasOperation_t opb = _cublasOpFromChar(transb);
  // 调整矩阵的 leading dimensions（LD3）以匹配 CuBLAS 需求
  _cublasAdjustLdLevel3(transa, transb, m, n, k, &lda, &ldb, &ldc);
  // 检查参数值，确保它们在有效范围内
  GEMM_CHECK_ARGVALUES(c10::complex<double>);
  // 调用 CuBLAS 的双精度复数矩阵乘法函数
  TORCH_CUDABLAS_CHECK(cublasZgemm(
      handle, opa, opb, m, n, k, reinterpret_cast<const cuDoubleComplex*>(&alpha), reinterpret_cast<const cuDoubleComplex*>(a),
      lda, reinterpret_cast<const cuDoubleComplex*>(b), ldb, reinterpret_cast<const cuDoubleComplex*>(&beta),
      reinterpret_cast<cuDoubleComplex*>(c), ldc));
}

template <>
void gemm_internal_cublas<c10::complex<float>>(CUDABLAS_GEMM_ARGTYPES(c10::complex<float>)) {
  // 在进行不确定性操作时发出警告，参见注释 [编写不确定性操作]
  globalContext().alertCuBLASConfigNotDeterministic();
  // 获取当前 CUDA CuBLAS 句柄
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  // 根据字符转换得到 CuBLAS 操作类型
  cublasOperation_t opa = _cublasOpFromChar(transa);
  cublasOperation_t opb = _cublasOpFromChar(transb);
  // 调整矩阵的 leading dimensions（LD3）以匹配 CuBLAS 需求
  _cublasAdjustLdLevel3(transa, transb, m, n, k, &lda, &ldb, &ldc);
  // 检查参数值，确保它们在有效范围内
  GEMM_CHECK_ARGVALUES(c10::complex<float>);
  // 调用 CuBLAS 的单精度复数矩阵乘法函数
  TORCH_CUDABLAS_CHECK(cublasCgemm(
      handle, opa, opb, m, n, k, reinterpret_cast<const cuComplex*>(&alpha), reinterpret_cast<const cuComplex*>(a),
      lda, reinterpret_cast<const cuComplex*>(b), ldb, reinterpret_cast<const cuComplex*>(&beta),
      reinterpret_cast<cuComplex*>(c), ldc));
}
void gemm_internal_cublas<at::Half>(CUDABLAS_GEMM_ARGTYPES(at::Half)) {
  // See Note [Writing Nondeterministic Operations]
  // 警告：执行不确定性操作
  globalContext().alertCuBLASConfigNotDeterministic();
  
  // 获取当前 CUDA CuBLAS 句柄
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  
  // 根据字符参数转换为 CuBLAS 的操作类型
  cublasOperation_t opa = _cublasOpFromChar(transa);
  cublasOperation_t opb = _cublasOpFromChar(transb);
  
  // 将 alpha 和 beta 参数转换为 float 类型
  float falpha = alpha;
  float fbeta = beta;
  
  // 调整第三级的 leading dimension，以适应传输矩阵的尺寸
  _cublasAdjustLdLevel3(transa, transb, m, n, k, &lda, &ldb, &ldc);
  
  // 检查参数值是否符合要求
  GEMM_CHECK_ARGVALUES(at::Half);
  
#ifdef USE_ROCM
  // 如果使用 ROCm 平台
  int flag = 0;
  
  // 如果启用了替代的 FP16 实现，则根据后向传递标志设置 flag
#if USE_GEMM_FLAGS_FP16_ALT_IMPL
  flag = at::ROCmBackwardPassGuard::is_backward_pass() ? rocblas_gemm_flags_fp16_alt_impl : 0;
#endif
  
  // 执行 ROCBLAS 的 gemm_ex 函数调用，并将返回状态转换为 HIP 状态
  TORCH_CUDABLAS_CHECK(rocBLASStatusToHIPStatus(rocblas_gemm_ex(
      (rocblas_handle)handle,
      hipOperationToRocOperation(opa),
      hipOperationToRocOperation(opb),
      m,
      n,
      k,
      &falpha,
      a,
      rocblas_datatype_f16_r,
      lda,
      b,
      rocblas_datatype_f16_r,
      ldb,
      &fbeta,
      c,
      rocblas_datatype_f16_r,
      ldc,
      c,
      rocblas_datatype_f16_r,
      ldc,
      rocblas_datatype_f32_r,
      rocblas_gemm_algo_standard,
      0,
      flag)));
#else
  // 如果不使用 ROCm 平台，获取当前 CUDA 设备的属性
  cudaDeviceProp* prop = at::cuda::getCurrentDeviceProperties();
  
  // 如果 CUDA 设备的主版本号大于等于 5
  if (prop->major >= 5) {
#ifndef USE_ROCM
    // 设置 CuBLAS 的数学模式为默认值
    cublasMath_t cublas_flags = CUBLAS_DEFAULT_MATH;
    
    // 如果不允许 FP16 降级精度的约简操作，则更新数学模式
    if (!at::globalContext().allowFP16ReductionCuBLAS()) {
      cublas_flags = static_cast<cublasMath_t>(cublas_flags | CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION);
    }
#endif
    
    // 设置 CuBLAS 的数学模式，避免意外的溢出问题
    // 禁止可能导致意外溢出问题的 fp16 降级
    TORCH_CUDABLAS_CHECK(cublasSetMathMode(handle, cublas_flags));
    
    // 调用 CuBLAS 的 gemmEx 函数执行矩阵乘法运算，使用默认的张量操作
    TORCH_CUDABLAS_CHECK(cublasGemmEx(
        handle,
        opa,
        opb,
        m,
        n,
        k,
        &falpha,
        a,
        CUDA_R_16F,
        lda,
        b,
        CUDA_R_16F,
        ldb,
        &fbeta,
        c,
        CUDA_R_16F,
        ldc,
        CUDA_R_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    
    // 恢复 CuBLAS 的数学模式为默认值
    TORCH_CUDABLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
  } else {
    // 如果 CUDA 设备的主版本号小于 5，调用 CuBLAS 的 sgemmEx 函数执行单精度浮点数矩阵乘法运算
    TORCH_CUDABLAS_CHECK(cublasSgemmEx(
        handle,
        opa,
        opb,
        m,
        n,
        k,
        &falpha,
        a,
        CUDA_R_16F,
        lda,
        b,
        CUDA_R_16F,
        ldb,
        &fbeta,
        c,
        CUDA_R_16F,
        ldc));
  }
#endif
}

template <>
void gemm_internal_cublas<at::BFloat16>(CUDABLAS_GEMM_ARGTYPES(at::BFloat16)) {
  // 警告：执行不确定性操作
  globalContext().alertCuBLASConfigNotDeterministic();
  
  // 获取当前 CUDA CuBLAS 句柄
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  
  // 根据字符参数转换为 CuBLAS 的操作类型
  cublasOperation_t opa = _cublasOpFromChar(transa);
  cublasOperation_t opb = _cublasOpFromChar(transb);
  
  // 将 alpha 和 beta 参数转换为 float 类型
  float falpha = alpha;
  float fbeta = beta;
  
  // 调整第三级的 leading dimension，以适应传输矩阵的尺寸
  _cublasAdjustLdLevel3(transa, transb, m, n, k, &lda, &ldb, &ldc);
  
  // 检查参数值是否符合要求
  GEMM_CHECK_ARGVALUES(at::BFloat16);
  
#ifndef USE_ROCM
  // 如果不使用 ROCm 平台，获取 CuBLAS 的数学模式
  cublasMath_t cublas_flags = CUBLAS_DEFAULT_MATH;
  
  // 如果不允许 BF16 降级精度的约简操作，则更新数学模式
  if (!at::globalContext().allowBF16ReductionCuBLAS()) {


继续编辑此处以添加剩余部分的注释。
    // 使用 static_cast 将 cublas_flags 转换为 cublasMath_t 类型，然后按位或运算符 | 将 cublas_flags 和 CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION 的值进行按位或操作
    cublas_flags = static_cast<cublasMath_t>(cublas_flags | CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION);
  }
#endif
// 如果定义了 USE_ROCM，则使用 CUBLAS_COMPUTE_32F 作为计算类型，否则使用 CUDA_R_32F
#if defined(USE_ROCM)
  auto compute_type = CUBLAS_COMPUTE_32F;
#else
  auto compute_type = CUDA_R_32F;
#endif
// 设置 cuBLAS 的数学模式为 cublas_flags 指定的模式
TORCH_CUDABLAS_CHECK(cublasSetMathMode(handle, cublas_flags));
// 调用 cuBLAS 的 gemmEx 函数执行矩阵乘法运算
TORCH_CUDABLAS_CHECK(cublasGemmEx(
    handle,
    opa,
    opb,
    m,
    n,
    k,
    &falpha,
    a,
    CUDA_R_16BF,
    lda,
    b,
    CUDA_R_16BF,
    ldb,
    &fbeta,
    c,
    CUDA_R_16BF,
    ldc,
    compute_type,
    CUBLAS_GEMM_DEFAULT_TENSOR_OP));
// 将 cuBLAS 的数学模式恢复为默认值
TORCH_CUDABLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
}

template <>
void gemm_internal<double>(CUDABLAS_GEMM_ARGTYPES(double))
{
  // 根据当前环境优选的 cuBLAS 后端选择执行双精度矩阵乘法的具体实现
  if (at::globalContext().blasPreferredBackend() == BlasBackend::Cublaslt) {
#ifdef USE_ROCM
    // 当使用 ROCm 时，使用 cublaslt 不支持双精度 gemm 的警告注释
    gemm_internal_cublas<double>(CUDABLAS_GEMM_ARGS(double));
#else
    // 否则使用 cublaslt 实现双精度矩阵乘法
    gemm_internal_cublaslt<double>(CUDABLAS_GEMM_ARGS(double));
#endif
  }
  else {
    // 使用 cuBLAS 实现双精度矩阵乘法
    gemm_internal_cublas<double>(CUDABLAS_GEMM_ARGS(double));
  }
}

template <>
void gemm_internal<float>(CUDABLAS_GEMM_ARGTYPES(float))
{
  // 根据当前环境优选的 cuBLAS 后端选择执行单精度矩阵乘法的具体实现
  if (at::globalContext().blasPreferredBackend() == BlasBackend::Cublaslt) {
    // 使用 cublaslt 实现单精度矩阵乘法
    gemm_internal_cublaslt<float>(CUDABLAS_GEMM_ARGS(float));
  }
  else {
    // 使用 cuBLAS 实现单精度矩阵乘法
    gemm_internal_cublas<float>(CUDABLAS_GEMM_ARGS(float));
  }
}

template <>
void gemm_internal<c10::complex<double>>(CUDABLAS_GEMM_ARGTYPES(c10::complex<double>))
{
  // 根据当前环境优选的 cuBLAS 后端选择执行复数双精度矩阵乘法的具体实现
  if (at::globalContext().blasPreferredBackend() == BlasBackend::Cublaslt) {
#ifdef USE_ROCM
    // 当使用 ROCm 时，使用 cublaslt 不支持复数双精度 gemm 的警告注释
    gemm_internal_cublas<c10::complex<double>>(CUDABLAS_GEMM_ARGS(c10::complex<double>));
#else
    // 否则使用 cublaslt 实现复数双精度矩阵乘法
    gemm_internal_cublaslt<c10::complex<double>>(CUDABLAS_GEMM_ARGS(c10::complex<double>));
#endif
  }
  else {
    // 使用 cuBLAS 实现复数双精度矩阵乘法
    gemm_internal_cublas<c10::complex<double>>(CUDABLAS_GEMM_ARGS(c10::complex<double>));
  }
}

template <>
void gemm_internal<c10::complex<float>>(CUDABLAS_GEMM_ARGTYPES(c10::complex<float>))
{
  // 根据当前环境优选的 cuBLAS 后端选择执行复数单精度矩阵乘法的具体实现
  if (at::globalContext().blasPreferredBackend() == BlasBackend::Cublaslt) {
#ifdef USE_ROCM
    // 当使用 ROCm 时，使用 cublaslt 不支持复数单精度 gemm 的警告注释
    gemm_internal_cublas<c10::complex<float>>(CUDABLAS_GEMM_ARGS(c10::complex<float>));
#else
    // 否则使用 cublaslt 实现复数单精度矩阵乘法
    gemm_internal_cublaslt<c10::complex<float>>(CUDABLAS_GEMM_ARGS(c10::complex<float>));
#endif
  }
  else {
    // 使用 cuBLAS 实现复数单精度矩阵乘法
    gemm_internal_cublas<c10::complex<float>>(CUDABLAS_GEMM_ARGS(c10::complex<float>));
  }
}

template <>
void gemm_internal<at::Half>(CUDABLAS_GEMM_ARGTYPES(at::Half))
{
  // 根据当前环境优选的 cuBLAS 后端选择执行 Half 精度矩阵乘法的具体实现
  if (at::globalContext().blasPreferredBackend() == BlasBackend::Cublaslt) {
    // 使用 cublaslt 实现 Half 精度矩阵乘法
    gemm_internal_cublaslt<at::Half>(CUDABLAS_GEMM_ARGS(at::Half));
  }
  else {
    // 使用 cuBLAS 实现 Half 精度矩阵乘法
    gemm_internal_cublas<at::Half>(CUDABLAS_GEMM_ARGS(at::Half));
  }
}

template <>
void gemm_internal<at::BFloat16>(CUDABLAS_GEMM_ARGTYPES(at::BFloat16))
{
  // 根据当前环境优选的 cuBLAS 后端选择执行 BFloat16 精度矩阵乘法的具体实现
  if (at::globalContext().blasPreferredBackend() == BlasBackend::Cublaslt) {
    // 使用 cublaslt 实现 BFloat16 精度矩阵乘法
    gemm_internal_cublaslt<at::BFloat16>(CUDABLAS_GEMM_ARGS(at::BFloat16));
  }
  else {
    // 使用 cuBLAS 实现 BFloat16 精度矩阵乘法
    // 使用CUBlas库执行通用矩阵乘法（GEMM）操作，模板参数为at::BFloat16类型
    gemm_internal_cublas<at::BFloat16>(CUDABLAS_GEMM_ARGS(at::BFloat16));
    // 结束函数定义
    }
template <typename DType>
inline void gemm_tunable(CUDABLAS_GEMM_ARGTYPES(DType)) {
  tunable::GemmParams<DType> params;
  params.transa = transa;
  params.transb = transb;
  params.m = m;
  params.n = n;
  params.k = k;
  params.alpha = alpha;
  params.a = a;
  params.lda = lda;
  params.b = b;
  params.ldb = ldb;
  params.beta = beta;
  params.c = c;
  params.ldc = ldc;

  bool transa_ = ((transa != 'n') && (transa != 'N'));  // 检查是否需要对 A 矩阵进行转置
  bool transb_ = ((transb != 'n') && (transb != 'N'));  // 检查是否需要对 B 矩阵进行转置

  if (transa_ && transb_) {  // 如果 A 和 B 都需要转置
    static tunable::GemmTunableOp<DType, tunable::BlasOp::T, tunable::BlasOp::T> gemm{};
    gemm(&params);  // 执行可调优的矩阵乘法操作
  }
  else if (transa_ && !transb_) {  // 如果只有 A 需要转置
    static tunable::GemmTunableOp<DType, tunable::BlasOp::T, tunable::BlasOp::N> gemm{};
    gemm(&params);  // 执行可调优的矩阵乘法操作
  }
  else if (!transa_ && transb_) {  // 如果只有 B 需要转置
    static tunable::GemmTunableOp<DType, tunable::BlasOp::N, tunable::BlasOp::T> gemm{};
    gemm(&params);  // 执行可调优的矩阵乘法操作
  }
  else if (!transa_ && !transb_) {  // 如果 A 和 B 都不需要转置
    static tunable::GemmTunableOp<DType, tunable::BlasOp::N, tunable::BlasOp::N> gemm{};
    gemm(&params);  // 执行可调优的矩阵乘法操作
  }
  else {
    TORCH_CHECK(false, "unreachable");  // 不可达代码分支，用于异常情况检测
  }
}

template <>
void gemm<double>(CUDABLAS_GEMM_ARGTYPES(double)) {
  auto tuning_ctx = at::cuda::tunable::getTuningContext();
  if (tuning_ctx->IsTunableOpEnabled()) {
    gemm_tunable<double>(CUDABLAS_GEMM_ARGS(double));  // 如果可调优操作被启用，调用双精度浮点数类型的可调优矩阵乘法
  }
  else {
    gemm_internal<double>(CUDABLAS_GEMM_ARGS(double));  // 否则，调用内部定义的双精度浮点数类型的矩阵乘法
  }
}

template <>
void gemm<float>(CUDABLAS_GEMM_ARGTYPES(float)) {
  auto tuning_ctx = at::cuda::tunable::getTuningContext();
  if (tuning_ctx->IsTunableOpEnabled()) {
    gemm_tunable<float>(CUDABLAS_GEMM_ARGS(float));  // 如果可调优操作被启用，调用单精度浮点数类型的可调优矩阵乘法
  }
  else {
    gemm_internal<float>(CUDABLAS_GEMM_ARGS(float));  // 否则，调用内部定义的单精度浮点数类型的矩阵乘法
  }
}

template <>
void gemm<c10::complex<double>>(CUDABLAS_GEMM_ARGTYPES(c10::complex<double>)) {
  auto tuning_ctx = at::cuda::tunable::getTuningContext();
  if (tuning_ctx->IsTunableOpEnabled()) {
    gemm_tunable<c10::complex<double>>(CUDABLAS_GEMM_ARGS(c10::complex<double>));  // 如果可调优操作被启用，调用复双精度浮点数类型的可调优矩阵乘法
  }
  else {
    gemm_internal<c10::complex<double>>(CUDABLAS_GEMM_ARGS(c10::complex<double>));  // 否则，调用内部定义的复双精度浮点数类型的矩阵乘法
  }
}

template <>
void gemm<c10::complex<float>>(CUDABLAS_GEMM_ARGTYPES(c10::complex<float>)) {
  auto tuning_ctx = at::cuda::tunable::getTuningContext();
  if (tuning_ctx->IsTunableOpEnabled()) {
    gemm_tunable<c10::complex<float>>(CUDABLAS_GEMM_ARGS(c10::complex<float>));  // 如果可调优操作被启用，调用复单精度浮点数类型的可调优矩阵乘法
  }
  else {
    gemm_internal<c10::complex<float>>(CUDABLAS_GEMM_ARGS(c10::complex<float>));  // 否则，调用内部定义的复单精度浮点数类型的矩阵乘法
  }
}

template <>
void gemm<at::Half>(CUDABLAS_GEMM_ARGTYPES(at::Half)) {
  auto tuning_ctx = at::cuda::tunable::getTuningContext();
  if (tuning_ctx->IsTunableOpEnabled()) {
    gemm_tunable<at::Half>(CUDABLAS_GEMM_ARGS(at::Half));  // 如果可调优操作被启用，调用半精度浮点数类型的可调优矩阵乘法
  }
  else {
    gemm_internal<at::Half>(CUDABLAS_GEMM_ARGS(at::Half));  // 否则，调用内部定义的半精度浮点数类型的矩阵乘法
  }
}

template <>
void gemm<at::BFloat16>(CUDABLAS_GEMM_ARGTYPES(at::BFloat16)) {
  auto tuning_ctx = at::cuda::tunable::getTuningContext();
  if (tuning_ctx->IsTunableOpEnabled()) {
    gemm_tunable<at::BFloat16>(CUDABLAS_GEMM_ARGS(at::BFloat16));

调用名为 `gemm_tunable` 的模板函数，使用模板参数 `at::BFloat16`，并传递宏定义 `CUDABLAS_GEMM_ARGS` 对 `at::BFloat16` 进行参数化处理。


  }
  else {
    gemm_internal<at::BFloat16>(CUDABLAS_GEMM_ARGS(at::BFloat16));
  }

在条件分支中，如果之前的条件不满足，则调用名为 `gemm_internal` 的模板函数，同样使用模板参数 `at::BFloat16`，并传递宏定义 `CUDABLAS_GEMM_ARGS` 对 `at::BFloat16` 进行参数化处理。
// 结束 gemm_and_bias 函数的定义
template <typename Dtype>
void gemm_and_bias(
    bool transpose_mat1,                          // 是否转置矩阵1
    bool transpose_mat2,                          // 是否转置矩阵2
    int64_t m,                                    // 矩阵C的行数
    int64_t n,                                    // 矩阵C的列数
    int64_t k,                                    // 矩阵A的列数或矩阵B的行数
    at::opmath_type<Dtype> alpha_val,             // 矩阵乘法的 alpha 值
    const Dtype* mat1_ptr,                        // 矩阵A的指针
    int64_t mat1_ld,                              // 矩阵A的leading dimension
    const Dtype* mat2_ptr,                        // 矩阵B的指针
    int64_t mat2_ld,                              // 矩阵B的leading dimension
    const Dtype* bias,                            // 偏置向量的指针
    Dtype* result_ptr,                            // 矩阵C的结果指针
    int64_t result_ld,                            // 矩阵C的leading dimension
    GEMMAndBiasActivationEpilogue activation) {   // 激活函数类型
  using opmath_t = at::opmath_type<Dtype>;
  opmath_t beta_val = 0;                          // 在epilogue中添加偏置项

  cudaDataType_t abcType = CUDA_R_32F;            // 默认使用32位浮点数类型
  cublasComputeType_t computeType = CUBLAS_COMPUTE_32F;  // 默认使用32位浮点数计算
  cudaDataType_t scaleType = CUDA_R_32F;          // 默认使用32位浮点数作为scaleType
  if constexpr (std::is_same_v<Dtype, double>) {
    abcType = CUDA_R_64F;                         // 如果Dtype为double，则使用64位浮点数类型
    computeType = CUBLAS_COMPUTE_64F;             // 使用64位浮点数计算
    scaleType = CUDA_R_64F;                       // scaleType也为64位浮点数
  } else if constexpr (std::is_same_v<Dtype, float>) {
#ifndef USE_ROCM
    if (at::globalContext().allowTF32CuBLAS()) {
      computeType = CUBLAS_COMPUTE_32F_FAST_TF32; // 如果允许使用TF32CuBLAS，则使用快速TF32模式
    }
#endif
    abcType = CUDA_R_32F;                         // 否则，默认为32位浮点数类型
  } else if constexpr (std::is_same_v<Dtype, at::Half>) {
    abcType = CUDA_R_16F;                         // 如果Dtype为Half，则使用16位浮点数类型
  } else if constexpr (std::is_same_v<Dtype, at::BFloat16>) {
    abcType = CUDA_R_16BF;                        // 如果Dtype为BFloat16，则使用16位Brain Float类型
  }

  CuBlasLtMatmulDescriptor computeDesc(computeType, scaleType);  // 创建CuBLAS LT矩阵乘法描述符
  cublasOperation_t transa = transpose_mat1 ? CUBLAS_OP_T : CUBLAS_OP_N;  // 确定矩阵A的转置属性
  computeDesc.setAttribute(CUBLASLT_MATMUL_DESC_TRANSA, transa);   // 设置矩阵A的转置属性
  cublasOperation_t transb = transpose_mat2 ? CUBLAS_OP_T : CUBLAS_OP_N;  // 确定矩阵B的转置属性
  computeDesc.setAttribute(CUBLASLT_MATMUL_DESC_TRANSB, transb);   // 设置矩阵B的转置属性
  cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;   // 默认使用偏置项epilogue
  if (activation == GEMMAndBiasActivationEpilogue::RELU) {
    epilogue = CUBLASLT_EPILOGUE_RELU_BIAS;        // 如果激活函数是RELU，则设置RELU偏置epilogue
  } else if (activation == GEMMAndBiasActivationEpilogue::GELU) {
#if CUDA_VERSION >= 11040 || defined(USE_ROCM)
    epilogue = CUBLASLT_EPILOGUE_GELU_BIAS;        // 如果激活函数是GELU，并且CUDA版本大于等于11040或者定义了USE_ROCM，则设置GELU偏置epilogue
#endif
  }

  if (bias != nullptr) {
    computeDesc.setAttribute(CUBLASLT_MATMUL_DESC_EPILOGUE, epilogue);  // 设置epilogue类型
    computeDesc.setAttribute(CUBLASLT_MATMUL_DESC_BIAS_POINTER, bias);  // 设置偏置指针
  }

  CuBlasLtMatrixLayout Adesc(abcType, m, k, mat1_ld, transpose_mat1);  // 创建矩阵A的布局描述
  CuBlasLtMatrixLayout Bdesc(abcType, k, n, mat2_ld, transpose_mat2);  // 创建矩阵B的布局描述
  CuBlasLtMatrixLayout Cdesc(abcType, m, n, result_ld);  // 创建矩阵C的布局描述

  CuBlasLtMatmulPreference preference;  // 创建CuBLAS LT矩阵乘法的偏好设置对象
  // 参考 https://github.com/pytorch/pytorch/issues/73328，设置最大工作空间为1M字节
  size_t workspaceSize = _getWorkspaceSize();
  preference.setAttribute(CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, workspaceSize);
#ifndef USE_ROCM
  // 获取 mat1_ptr 的内存对齐要求
  uint32_t a_alignment = _getAlignment(reinterpret_cast<uintptr_t>(mat1_ptr));
  // 获取 mat2_ptr 的内存对齐要求
  uint32_t b_alignment = _getAlignment(reinterpret_cast<uintptr_t>(mat2_ptr));
  // 获取 result_ptr 的内存对齐要求
  uint32_t c_alignment = _getAlignment(reinterpret_cast<uintptr_t>(result_ptr));
  // 获取 bias 的内存对齐要求
  uint32_t d_alignment = _getAlignment(reinterpret_cast<uintptr_t>(bias));
  // 设置 CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES 属性为 mat1_ptr 的对齐要求
  preference.setAttribute(CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, a_alignment);
  // 设置 CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES 属性为 mat2_ptr 的对齐要求
  preference.setAttribute(CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES, b_alignment);
  // 设置 CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES 属性为 result_ptr 的对齐要求
  preference.setAttribute(CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, c_alignment);
  // 设置 CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES 属性为 bias 的对齐要求
  preference.setAttribute(CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES, d_alignment);
#endif

  // 获取 CUDA 的内存分配器
  auto& allocator = *::c10::cuda::CUDACachingAllocator::get();
  // 分配 workspaceSize 大小的内存空间
  auto workspace = allocator.allocate(workspaceSize);
  // 检查内存是否成功分配
  TORCH_CHECK(workspace.get() != nullptr, "OOM trying to allocate workspace for cublaslt");

  // 初始化 cublasLtMatmulHeuristicResult_t 结构体
  cublasLtMatmulHeuristicResult_t heuristicResult = {};
  int returnedResult = 0;
  // 获取当前的 cublasLtHandle
  cublasLtHandle_t ltHandle = at::cuda::getCurrentCUDABlasLtHandle();
  // 获取 matmul 算法的启发式结果
  TORCH_CUDABLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(
      ltHandle,
      computeDesc.descriptor(),
      Adesc.descriptor(),
      Bdesc.descriptor(),
      Cdesc.descriptor(),
      Cdesc.descriptor(),
      preference.descriptor(),
      1,
      &heuristicResult,
      &returnedResult));
  // 检查返回结果是否为成功
  if (returnedResult == 0) {
    // 如果返回结果为0，则表示不支持当前配置
    TORCH_CUDABLAS_CHECK(CUBLAS_STATUS_NOT_SUPPORTED);
  }

  // 执行 cublasLtMatmul 函数进行矩阵乘法计算
  cublasStatus_t cublasStatus = cublasLtMatmul(
      ltHandle,
      computeDesc.descriptor(),
      &alpha_val,
      mat1_ptr,
      Adesc.descriptor(),
      mat2_ptr,
      Bdesc.descriptor(),
      &beta_val,
      result_ptr,
      Cdesc.descriptor(),
      result_ptr,
      Cdesc.descriptor(),
      &heuristicResult.algo,
      workspace.mutable_get(),
      workspaceSize,
      at::cuda::getCurrentCUDAStream());
  // 检查 cublas 执行状态是否成功
  TORCH_CHECK(
      cublasStatus == CUBLAS_STATUS_SUCCESS,
      "CUDA error: ",
      at::cuda::blas::_cublasGetErrorEnum(cublasStatus),
      " when calling cublasLtMatmul with transpose_mat1 ",
      transpose_mat1,
      " transpose_mat2 ",
      transpose_mat2,
      " m ",
      m,
      " n ",
      n,
      " k ",
      k,
      " mat1_ld ",
      mat1_ld,
      " mat2_ld ",
      mat2_ld,
      " result_ld ",
      result_ld,
      " abcType ",
      abcType,
      " computeType ",
      computeType,
      " scaleType ",
      scaleType);
}

// 模板实例化：双精度版本的 gemm_and_bias 函数声明
template void gemm_and_bias(
    bool transpose_mat1,
    bool transpose_mat2,
    int64_t m,
    int64_t n,
    int64_t k,
    at::opmath_type<double> alpha_val,
    const double* mat1_ptr,
    int64_t mat1_ld,
    const double* mat2_ptr,
    int64_t mat2_ld,
    const double* bias,
    double* result_ptr,
    int64_t result_ld,
    GEMMAndBiasActivationEpilogue activation);

// 模板实例化：单精度版本的 gemm_and_bias 函数声明
template void gemm_and_bias(
    bool transpose_mat1,
    bool transpose_mat2,
    int64_t m,
    int64_t n,
    int64_t k,
    at::opmath_type<float> alpha_val,
    const float* mat1_ptr,
    int64_t mat1_ld,
    const float* mat2_ptr,
    int64_t mat2_ld,
    const float* bias,
    float* result_ptr,
    int64_t result_ld,
    GEMMAndBiasActivationEpilogue activation);


这段代码实现了一个使用 CUDA 的矩阵乘法和偏置加法的函数模板 `gemm_and_bias`。在非 ROCm 环境下，它会获取输入指针的内存对齐要求，并设置给 CUDA 的优化偏好。然后，它通过 CUDA 的 `cublasLtMatmul` 函数执行矩阵乘法操作，并检查执行状态。
    const float* mat2_ptr,      // 指向第二个矩阵（matrix2）的指针，其元素类型为 float 类型
    int64_t mat2_ld,            // 第二个矩阵（matrix2）的 leading dimension（主导维度）
                                // （即在内存中的行数或列数，取决于矩阵的存储方式）
    const float* bias,          // 指向偏置（bias）数组的指针，其元素类型为 float 类型
    float* result_ptr,          // 指向结果矩阵的指针，其元素类型为 float 类型
    int64_t result_ld,          // 结果矩阵的 leading dimension（主导维度）
                                // （即在内存中的行数或列数，取决于矩阵的存储方式）
    GEMMAndBiasActivationEpilogue activation);  // 结构体或函数指针，用于表示 GEMM（General Matrix Multiply）
                                               // 运算和偏置激活（activation）后的结尾处理
// 声明模板函数gemm_and_bias，用于矩阵乘法和偏置操作，支持半精度数据类型
template void gemm_and_bias(
    bool transpose_mat1,  // 是否对第一个矩阵进行转置操作
    bool transpose_mat2,  // 是否对第二个矩阵进行转置操作
    int64_t m,            // 矩阵C的行数
    int64_t n,            // 矩阵C的列数
    int64_t k,            // 矩阵A的列数和矩阵B的行数
    at::opmath_type<at::Half> alpha_val,  // 乘法的缩放因子alpha的值，半精度
    const at::Half* mat1_ptr,  // 第一个矩阵的数据指针
    int64_t mat1_ld,           // 第一个矩阵的leading dimension（列数）
    const at::Half* mat2_ptr,  // 第二个矩阵的数据指针
    int64_t mat2_ld,           // 第二个矩阵的leading dimension（列数）
    const at::Half* bias,      // 偏置向量的数据指针
    at::Half* result_ptr,      // 结果矩阵的数据指针
    int64_t result_ld,         // 结果矩阵的leading dimension（列数）
    GEMMAndBiasActivationEpilogue activation  // GEMM计算后的激活处理
);

// 声明模板函数gemm_and_bias，用于矩阵乘法和偏置操作，支持BFloat16数据类型
template void gemm_and_bias(
    bool transpose_mat1,         // 是否对第一个矩阵进行转置操作
    bool transpose_mat2,         // 是否对第二个矩阵进行转置操作
    int64_t m,                   // 矩阵C的行数
    int64_t n,                   // 矩阵C的列数
    int64_t k,                   // 矩阵A的列数和矩阵B的行数
    at::opmath_type<at::BFloat16> alpha_val,  // 乘法的缩放因子alpha的值，BFloat16数据类型
    const at::BFloat16* mat1_ptr,  // 第一个矩阵的数据指针
    int64_t mat1_ld,                // 第一个矩阵的leading dimension（列数）
    const at::BFloat16* mat2_ptr,  // 第二个矩阵的数据指针
    int64_t mat2_ld,                // 第二个矩阵的leading dimension（列数）
    const at::BFloat16* bias,      // 偏置向量的数据指针
    at::BFloat16* result_ptr,      // 结果矩阵的数据指针
    int64_t result_ld,             // 结果矩阵的leading dimension（列数）
    GEMMAndBiasActivationEpilogue activation  // GEMM计算后的激活处理
);

// 定义函数scaled_gemm，执行缩放后的矩阵乘法操作
void scaled_gemm(
    char transa,              // 第一个矩阵的转置类型：'n'或't'
    char transb,              // 第二个矩阵的转置类型：'n'或't'
    int64_t m,                // 矩阵C的行数
    int64_t n,                // 矩阵C的列数
    int64_t k,                // 矩阵A的列数和矩阵B的行数
    const void* mat1_ptr,     // 第一个矩阵的数据指针
    const void* mat1_scale_ptr,  // 第一个矩阵的缩放因子指针
    int64_t mat1_ld,          // 第一个矩阵的leading dimension（列数）
    ScalarType mat1_dtype,    // 第一个矩阵的数据类型
    const void* mat2_ptr,     // 第二个矩阵的数据指针
    const void* mat2_scale_ptr,  // 第二个矩阵的缩放因子指针
    int64_t mat2_ld,          // 第二个矩阵的leading dimension（列数）
    ScalarType mat2_dtype,    // 第二个矩阵的数据类型
    const void* bias_ptr,     // 偏置向量的数据指针
    ScalarType bias_dtype,    // 偏置向量的数据类型
    void* result_ptr,         // 结果矩阵的数据指针
    const void *result_scale_ptr,  // 结果矩阵的缩放因子指针
    int64_t result_ld,        // 结果矩阵的leading dimension（列数）
    ScalarType result_dtype,  // 结果矩阵的数据类型
    void* amax_ptr,           // 用于记录最大值的指针
    bool use_fast_accum       // 是否启用快速累积模式
) {
#if CUDA_VERSION >= 11080 || defined(USE_ROCM)
  const auto computeType = CUBLAS_COMPUTE_32F;  // 指定计算类型为单精度浮点数
  const auto scaleType = CUDA_R_32F;  // 指定缩放类型为单精度浮点数
  const int8_t fastAccuMode = use_fast_accum ? 1 : 0;  // 根据use_fast_accum决定是否启用快速累积模式
  const float alpha_val = 1.0;  // 矩阵乘法的alpha因子，默认为1.0
  const float beta_val = 0.0;   // 矩阵乘法的beta因子，默认为0.0
  CuBlasLtMatmulDescriptor computeDesc(computeType, scaleType);  // 创建CuBLAS Lt矩阵乘法描述符
  computeDesc.setAttribute(CUBLASLT_MATMUL_DESC_TRANSA, _cublasOpFromChar(transa));  // 设置矩阵A的转置属性
  computeDesc.setAttribute(CUBLASLT_MATMUL_DESC_TRANSB, _cublasOpFromChar(transb));  // 设置矩阵B的转置属性
  computeDesc.setAttribute(CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, mat1_scale_ptr);  // 设置矩阵A的缩放因子
  computeDesc.setAttribute(CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, mat2_scale_ptr);  // 设置矩阵B的缩放因子
  computeDesc.setAttribute(CUBLASLT_MATMUL_DESC_D_SCALE_POINTER, result_scale_ptr);  // 设置结果矩阵的缩放因子
#if !defined(USE_ROCM) || (defined(USE_ROCM) && ROCM_VERSION >= 60200)
  // 在ROCm版本6.2以上支持Amax特性
  if (isFloat8Type(result_dtype)) {
    computeDesc.setAttribute(CUBLASLT_MATMUL_DESC_AMAX_D_POINTER, amax_ptr);  // 设置记录最大值的指针
  }
#endif
#ifndef USE_ROCM
  computeDesc.setAttribute(CUBLASLT_MATMUL_DESC_FAST_ACCUM, fastAccuMode);  // 设置是否启用快速累积模式
#endif
  CuBlasLtMatrixLayout Adesc(ScalarTypeToCudaDataType(mat1_dtype), m, k, mat1_ld, transa == 't');  // 创建CuBLAS Lt矩阵A的布局描述符
  CuBlasLtMatrixLayout Bdesc(ScalarTypeToCudaDataType(mat2_dtype), k, n, mat2_ld, transb == 't');  // 创建CuBLAS Lt矩阵B的布局描述符
#ifdef USE_ROCM
  // ROCm环境下Cdesc未使用，但需要设置一个合理的值以满足hipBLAS Lt的要求
  CuBlasLtMatrixLayout Cdesc(ScalarTypeToCudaDataType(result_dtype), m, n, result_ld);
#else
  CuBlasLtMatrixLayout Cdesc(ScalarTypeToCudaDataType(bias_dtype), m, n, result_ld);  // 创建CuBLAS Lt矩阵
    // 设置计算描述对象的偏置指针属性，指向偏置数据的内存地址
    computeDesc.setAttribute(CUBLASLT_MATMUL_DESC_BIAS_POINTER, bias_ptr);
    // 设置计算描述对象的后处理属性为带偏置
    computeDesc.setAttribute(CUBLASLT_MATMUL_DESC_EPILOGUE, CUBLASLT_EPILOGUE_BIAS);
    // 设置计算描述对象的偏置数据类型属性，将偏置数据类型转换为对应的CUDA数据类型
    computeDesc.setAttribute(CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, ScalarTypeToCudaDataType(bias_dtype));
  }
  // 获取需要的工作空间大小
  size_t workspaceSize = _getWorkspaceSize();
  // 获取CUDA的内存分配器
  auto& allocator = *::c10::cuda::CUDACachingAllocator::get();
  // 分配工作空间
  auto workspace = allocator.allocate(workspaceSize);
  // 检查是否成功分配了工作空间，如果失败则输出内存溢出的错误信息
  TORCH_CHECK(workspace.get() != nullptr, "OOM trying to allocate workspace for cublaslt");

  // 设置CuBlasLtMatmulPreference对象，包括设置最大工作空间字节数
  CuBlasLtMatmulPreference preference;
  preference.setAttribute(CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, workspaceSize);
  // 初始化用于存储启发式搜索结果的结构体
  cublasLtMatmulHeuristicResult_t heuristicResult = {};
  int returnedResult = 0;
  // 获取当前CUDA的CuBLAS LT句柄
  cublasLtHandle_t ltHandle = at::cuda::getCurrentCUDABlasLtHandle();
  // 执行CuBLAS LT的矩阵乘法算法的启发式选择过程
  TORCH_CUDABLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(
      ltHandle,
      computeDesc.descriptor(),
      Adesc.descriptor(),
      Bdesc.descriptor(),
      Cdesc.descriptor(),
      Ddesc.descriptor(),
      preference.descriptor(),
      1,
      &heuristicResult,
      &returnedResult));
  // 检查启发式选择是否成功
  if (returnedResult == 0) {
#ifndef USE_ROCM
    // 如果未定义 USE_ROCM 宏，则抛出不支持的错误状态
    TORCH_CUDABLAS_CHECK(CUBLAS_STATUS_NOT_SUPPORTED);
#else
    // 定义一个存储所有算法结果的向量
    std::vector<hipblasLtMatmulHeuristicResult_t> all_algos;
    // 调用 hipBLAS LT 扩展库的函数，获取所有算法
    TORCH_CUDABLAS_CHECK(hipblaslt_ext::getAllAlgos(
        ltHandle,
        hipblaslt_ext::GemmType::HIPBLASLT_GEMM,
        _cublasOpFromChar(transa),   // 将字符转换为 cuBLAS 操作符
        _cublasOpFromChar(transb),   // 将字符转换为 cuBLAS 操作符
        ScalarTypeToCudaDataType(mat1_dtype),  // 将标量类型转换为 CUDA 数据类型
        ScalarTypeToCudaDataType(mat2_dtype),  // 同上
        ScalarTypeToCudaDataType(result_dtype), // 同上
        ScalarTypeToCudaDataType(result_dtype), // 同上
        CUBLAS_COMPUTE_32F,  // 设置计算类型为 32 位浮点数
        all_algos));  // 存储获取到的所有算法结果

    // 如果没有可用的算法结果，则抛出不支持的错误状态
    if (all_algos.size() == 0) {
      TORCH_CUDABLAS_CHECK(CUBLAS_STATUS_NOT_SUPPORTED);
    }

    // 在所有算法中选择第一个有效的解决方案
    bool found = false;
    for (size_t i = 0; i < all_algos.size(); i++) {
        size_t ret_workspace_size = 0;
        // 检查特定算法是否受支持，并获取需要的工作空间大小
        auto is_valid_status = hipblaslt_ext::matmulIsAlgoSupported(
                ltHandle,
                computeDesc.descriptor(),
                &alpha_val,
                Adesc.descriptor(),
                Bdesc.descriptor(),
                &beta_val,
                Cdesc.descriptor(),
                Ddesc.descriptor(),
                all_algos[i].algo,
                ret_workspace_size);
        // 如果算法支持且所需的工作空间大小小于等于可用的工作空间大小，则选择此算法
        if (is_valid_status == HIPBLAS_STATUS_SUCCESS) {
            if (ret_workspace_size <= workspaceSize) {
                heuristicResult = all_algos[i];
                found = true;
                break;
            }
        }
    }
    // 如果没有找到有效的 hipBLAS LT 解决方案，则抛出错误
    TORCH_CHECK(found, "could not find valid hipblaslt solution");
#endif

// 调用 cuBLAS LT 的矩阵乘法函数进行计算
cublasStatus_t cublasStatus = cublasLtMatmul(
    ltHandle,
    computeDesc.descriptor(),
    &alpha_val,
    mat1_ptr,
    Adesc.descriptor(),
    mat2_ptr,
    Bdesc.descriptor(),
    &beta_val,
#ifdef USE_ROCM
    result_ptr, // 在 USE_ROCM 定义下，由于 beta_val 为 0，因此未使用，但 hipBLAS LT 不能处理 nullptr
#else
    nullptr,
#endif
    Cdesc.descriptor(),
    result_ptr,
    Ddesc.descriptor(),
    &heuristicResult.algo,
    workspace.mutable_get(),
    workspaceSize,
    at::cuda::getCurrentCUDAStream());

// 检查 cuBLAS 操作的执行状态，如果失败，则抛出错误
TORCH_CHECK(
    cublasStatus == CUBLAS_STATUS_SUCCESS,
    "CUDA error: ",
    at::cuda::blas::_cublasGetErrorEnum(cublasStatus),
    " when calling cublasLtMatmul with transpose_mat1 ",
    transa,
    " transpose_mat2 ",
    transb,
    " m ",
    m,
    " n ",
    n,
    " k ",
    k,
    " mat1_ld ",
    mat1_ld,
    " mat2_ld ",
    mat2_ld,
    " result_ld ",
    result_ld,
    " computeType ",
    computeType,
    " scaleType ",
    scaleType);

// 如果未定义 USE_ROCM 或 CUDA 版本低于 11.8，则抛出错误
return;
#endif // CUDA_VERSION >= 11080 || defined(USE_ROCM)
TORCH_CHECK(false, "scaled_gemm is only supported for CUDA 11.8 and above");
}
  // 声明一个名为 n 的 int64_t 类型的参数，表示矩阵 mat1 的行数
  int64_t n,
  // 声明一个名为 k 的 int64_t 类型的参数，表示矩阵 mat1 的列数（也是 mat2 的行数）
  int64_t k,
  // 声明一个指向 int8_t 类型的常量指针，指向矩阵 mat1 的数据
  const int8_t* mat1_ptr,
  // 声明一个 int64_t 类型的参数，表示矩阵 mat1 的行跨度
  int64_t mat1_ld,
  // 声明一个指向 int8_t 类型的常量指针，指向矩阵 mat2 的数据
  const int8_t* mat2_ptr,
  // 声明一个 int64_t 类型的参数，表示矩阵 mat2 的行跨度
  int64_t mat2_ld,
  // 声明一个指向 int32_t 类型的指针，用于存储乘法结果的数据
  int32_t* result_ptr,
  // 声明一个 int64_t 类型的参数，表示乘法结果矩阵的行跨度
  int64_t result_ld) {

  // 设置计算类型为 CUBLAS_COMPUTE_32I
  cublasComputeType_t computeType = CUBLAS_COMPUTE_32I;
  // 设置 scaleType 为 CUDA_R_32I
  cudaDataType_t scaleType = CUDA_R_32I;

  // 设置 abType 为 CUDA_R_8I，表示矩阵元素的数据类型
  cudaDataType_t abType = CUDA_R_8I;
  // 设置 cType 为 CUDA_R_32I，表示乘法结果矩阵的数据类型
  cudaDataType_t cType = CUDA_R_32I;

  // 创建 CuBlasLtMatmulDescriptor 对象，指定计算类型和数据类型
  CuBlasLtMatmulDescriptor computeDesc(computeType, scaleType);
  // 根据 transpose_mat1 的值设置 transa，控制是否转置矩阵 mat1
  cublasOperation_t transa = transpose_mat1 ? CUBLAS_OP_T : CUBLAS_OP_N;
  // 将转置信息设置到 computeDesc 中
  computeDesc.setAttribute(CUBLASLT_MATMUL_DESC_TRANSA, transa);
  // 根据 transpose_mat2 的值设置 transb，控制是否转置矩阵 mat2
  cublasOperation_t transb = transpose_mat2 ? CUBLAS_OP_T : CUBLAS_OP_N;
  // 将转置信息设置到 computeDesc 中
  computeDesc.setAttribute(CUBLASLT_MATMUL_DESC_TRANSB, transb);

  // 创建 CuBlasLtMatrixLayout 对象 Adesc，描述矩阵 mat1 的布局信息
  CuBlasLtMatrixLayout Adesc(abType, m, k, mat1_ld, transpose_mat1);
  // 创建 CuBlasLtMatrixLayout 对象 Bdesc，描述矩阵 mat2 的布局信息
  CuBlasLtMatrixLayout Bdesc(abType, k, n, mat2_ld, transpose_mat2);
  // 创建 CuBlasLtMatrixLayout 对象 Cdesc，描述乘法结果矩阵的布局信息
  CuBlasLtMatrixLayout Cdesc(cType, m, n, result_ld);

  // 设置 alpha_val 为 1，与 scaleType 的数据类型对应
  at::opmath_type<int32_t> alpha_val = 1;
  // 设置 beta_val 为 0，与 cType 的数据类型对应
  int32_t beta_val = 0;
  // 获取当前的 CuBLAS LT 句柄
  cublasLtHandle_t ltHandle = at::cuda::getCurrentCUDABlasLtHandle();
#ifdef USE_ROCM
  // 定义 CuBlasLtMatmulPreference 对象，用于指定计算偏好
  CuBlasLtMatmulPreference preference;
  // 获取推荐的工作空间大小
  size_t workspaceSize = _getWorkspaceSize();
  // 设置偏好属性，指定最大工作空间字节数
  preference.setAttribute(CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, workspaceSize);
  // 获取 CUDA 内存分配器的实例
  auto& allocator = *::c10::cuda::CUDACachingAllocator::get();
  // 分配工作空间
  auto workspace = allocator.allocate(workspaceSize);
  // 定义用于存储启发式结果的结构体
  cublasLtMatmulHeuristicResult_t heuristicResult = {};
  // 定义用于存储返回结果的整数
  int returnedResult = 0;
  // 获取启发式算法结果
  TORCH_CUDABLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(
      ltHandle,
      computeDesc.descriptor(),
      Adesc.descriptor(),
      Bdesc.descriptor(),
      Cdesc.descriptor(),
      Cdesc.descriptor(),
      preference.descriptor(),
      1,
      &heuristicResult,
      &returnedResult));
  // 检查是否成功获取启发式算法结果
  if (returnedResult == 0) {
    // 如果获取失败，抛出不支持的异常
    TORCH_CUDABLAS_CHECK(CUBLAS_STATUS_NOT_SUPPORTED);
  }
#endif

  // 调用 CuBlasLtMatmul 执行矩阵乘法
  cublasStatus_t cublasStatus = cublasLtMatmul(
      ltHandle,
      computeDesc.descriptor(),
      &alpha_val,
      mat1_ptr,
      Adesc.descriptor(),
      mat2_ptr,
      Bdesc.descriptor(),
      &beta_val,
      result_ptr,
      Cdesc.descriptor(),
      result_ptr,
      Cdesc.descriptor(),
#ifdef USE_ROCM
      &heuristicResult.algo, // ROCm 平台下使用启发式算法结果
#else
      nullptr, // 其他平台不使用启发式算法
#endif
#ifdef USE_ROCM
      workspace.mutable_get(), // ROCm 平台下使用工作空间
#else
      nullptr, // 其他平台不使用工作空间
#endif
#ifdef USE_ROCM
      workspaceSize, // ROCm 平台下指定工作空间大小
#else
      0, // 其他平台工作空间大小设为 0
#endif
      at::cuda::getCurrentCUDAStream());
  // 检查 CuBlasLtMatmul 调用的执行状态
  TORCH_CHECK(
      cublasStatus == CUBLAS_STATUS_SUCCESS,
      "CUDA error: ",
      at::cuda::blas::_cublasGetErrorEnum(cublasStatus),
      " when calling cublasLtMatmul with transpose_mat1 ",
      transpose_mat1,
      " transpose_mat2 ",
      transpose_mat2,
      " m ",
      m,
      " n ",
      n,
      " k ",
      k,
      " mat1_ld ",
      mat1_ld,
      " mat2_ld ",
      mat2_ld,
      " result_ld ",
      result_ld,
      " abType ",
      abType,
      " cType ",
      cType,
      " computeType ",
      computeType,
      " scaleType ",
      scaleType);
}

template <>
void trsm<float>(CUDABLAS_TRSM_ARGTYPES(float)) {
  // 调用 cuBLAS 中的 cublasStrsm 执行单精度浮点数的 TRSM 运算
  TORCH_CUDABLAS_CHECK(cublasStrsm(
      handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb));
}

template <>
void trsm<double>(CUDABLAS_TRSM_ARGTYPES(double)) {
  // 调用 cuBLAS 中的 cublasDtrsm 执行双精度浮点数的 TRSM 运算
  TORCH_CUDABLAS_CHECK(cublasDtrsm(
      handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb));
}

template <>
void trsm<c10::complex<float>>(CUDABLAS_TRSM_ARGTYPES(c10::complex<float>)) {
  // 调用 cuBLAS 中的 cublasCtrsm 执行单精度复数的 TRSM 运算
  TORCH_CUDABLAS_CHECK(cublasCtrsm(
      handle,
      side,
      uplo,
      trans,
      diag,
      m,
      n,
      reinterpret_cast<const cuComplex*>(alpha),
      reinterpret_cast<const cuComplex*>(A),
      lda,
      reinterpret_cast<cuComplex*>(B),
      ldb));
}
void trsm<c10::complex<double>>(CUDABLAS_TRSM_ARGTYPES(c10::complex<double>)) {
  // 调用 cuBLAS 库中的双精度复数矩阵三角解算函数 trsm，用于解决线性方程组
  TORCH_CUDABLAS_CHECK(cublasZtrsm(
      handle,
      side,
      uplo,
      trans,
      diag,
      m,
      n,
      reinterpret_cast<const cuDoubleComplex*>(alpha),
      reinterpret_cast<const cuDoubleComplex*>(A),
      lda,
      reinterpret_cast<cuDoubleComplex*>(B),
      ldb));
}

template <>
void trsmBatched<float>(CUDABLAS_TRSM_BATCHED_ARGTYPES(float)) {
  // 调用 cuBLAS 库中的批量单精度矩阵三角解算函数 trsmBatched
  TORCH_CUDABLAS_CHECK(cublasStrsmBatched(
      handle,
      side,
      uplo,
      trans,
      diag,
      m,
      n,
      alpha,
      A,
      lda,
      B,
      ldb,
      batchCount));
}

template <>
void trsmBatched<double>(CUDABLAS_TRSM_BATCHED_ARGTYPES(double)) {
  // 调用 cuBLAS 库中的批量双精度矩阵三角解算函数 trsmBatched
  TORCH_CUDABLAS_CHECK(cublasDtrsmBatched(
      handle,
      side,
      uplo,
      trans,
      diag,
      m,
      n,
      alpha,
      A,
      lda,
      B,
      ldb,
      batchCount));
}

template <>
void trsmBatched<c10::complex<float>>(
    CUDABLAS_TRSM_BATCHED_ARGTYPES(c10::complex<float>)) {
  // 调用 cuBLAS 库中的批量单精度复数矩阵三角解算函数 trsmBatched
  TORCH_CUDABLAS_CHECK(cublasCtrsmBatched(
      handle,
      side,
      uplo,
      trans,
      diag,
      m,
      n,
      reinterpret_cast<const cuComplex*>(alpha),
      reinterpret_cast<cuComplex**>(A),
      lda,
      reinterpret_cast<cuComplex**>(B),
      ldb,
      batchCount));
}

template <>
void trsmBatched<c10::complex<double>>(
    CUDABLAS_TRSM_BATCHED_ARGTYPES(c10::complex<double>)) {
  // 调用 cuBLAS 库中的批量双精度复数矩阵三角解算函数 trsmBatched
  TORCH_CUDABLAS_CHECK(cublasZtrsmBatched(
      handle,
      side,
      uplo,
      trans,
      diag,
      m,
      n,
      reinterpret_cast<const cuDoubleComplex*>(alpha),
      reinterpret_cast<cuDoubleComplex**>(A),
      lda,
      reinterpret_cast<cuDoubleComplex**>(B),
      ldb,
      batchCount));
}

/* LEVEL 2 BLAS FUNCTIONS */

#define GEMV_CHECK_ARGVALUES(Dtype)           \
  do {                                        \
    CUDABLAS_NONNEGINT_CHECK(gemv<Dtype>, m); \
    CUDABLAS_NONNEGINT_CHECK(gemv<Dtype>, n); \
    CUDABLAS_POSINT_CHECK(gemv<Dtype>, lda);  \
    CUDABLAS_POSINT_CHECK(gemv<Dtype>, incx); \
    CUDABLAS_POSINT_CHECK(gemv<Dtype>, incy); \
  } while (0)

template <>
void gemv<c10::complex<double>>(CUDABLAS_GEMV_ARGTYPES(c10::complex<double>)) {
  // 通知全局上下文，在使用 cuBLAS 进行复杂操作时可能是非确定性的
  // See Note [Writing Nondeterministic Operations]
  globalContext().alertCuBLASConfigNotDeterministic();
  // 获取当前 CUDA cuBLAS 句柄
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  // 根据转置标志获取 cublasOperation_t 类型的操作符
  cublasOperation_t op = _cublasOpFromChar(trans);
  // 调整 Level 2 操作的矩阵行数和列数
  _cublasAdjustLdLevel2(m, n, &lda);
  // 检查 gemv 操作的参数值是否合法
  GEMV_CHECK_ARGVALUES(c10::complex<double>);
  // 调用 cuBLAS 库中的双精度复数矩阵-向量乘法函数 gemv
  TORCH_CUDABLAS_CHECK(
      cublasZgemv(handle, op, m, n, reinterpret_cast<const cuDoubleComplex*>(&alpha), reinterpret_cast<const cuDoubleComplex*>(a),
      lda, reinterpret_cast<const cuDoubleComplex*>(x), incx, reinterpret_cast<const cuDoubleComplex*>(&beta),
      reinterpret_cast<cuDoubleComplex*>(y), incy));
}

template <>
void gemv<c10::complex<float>>(CUDABLAS_GEMV_ARGTYPES(c10::complex<float>)) {
  // gemv is bw bound, and does not benefit from TF32. But the precision
  // loss still happens on TF32. So we disable it here.
  // gemv 是受带宽限制的，不受 TF32 的益处。但是 TF32 仍会导致精度损失，因此我们在这里禁用它。
  NoTF32Guard disable_tf32;
  // See Note [Writing Nondeterministic Operations]
  // 参见注释 [编写非确定性操作]
  globalContext().alertCuBLASConfigNotDeterministic();
  // 获取当前 CUDA cuBLAS 句柄
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  // 根据转置标志 trans 获取 cublas 操作类型
  cublasOperation_t op = _cublasOpFromChar(trans);
  // 调整矩阵操作的 leading dimension 参数
  _cublasAdjustLdLevel2(m, n, &lda);
  // 检查参数值是否有效
  GEMV_CHECK_ARGVALUES(c10::complex<float>);
  // 调用 cublas 函数执行复数单精度向量乘法
  TORCH_CUDABLAS_CHECK(
      cublasCgemv(handle, op, m, n, reinterpret_cast<const cuComplex*>(&alpha), reinterpret_cast<const cuComplex*>(a),
      lda, reinterpret_cast<const cuComplex*>(x), incx, reinterpret_cast<const cuComplex*>(&beta),
      reinterpret_cast<cuComplex*>(y), incy));
}

template <>
void gemv<double>(CUDABLAS_GEMV_ARGTYPES(double)) {
  // See Note [Writing Nondeterministic Operations]
  // 参见注释 [编写非确定性操作]
  globalContext().alertCuBLASConfigNotDeterministic();
  // 获取当前 CUDA cuBLAS 句柄
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  // 根据转置标志 trans 获取 cublas 操作类型
  cublasOperation_t op = _cublasOpFromChar(trans);
  // 调整矩阵操作的 leading dimension 参数
  _cublasAdjustLdLevel2(m, n, &lda);
  // 检查参数值是否有效
  GEMV_CHECK_ARGVALUES(double);
  // 调用 cublas 函数执行双精度向量乘法
  TORCH_CUDABLAS_CHECK(
      cublasDgemv(handle, op, m, n, &alpha, a, lda, x, incx, &beta, y, incy));
}

template <>
void gemv<float>(CUDABLAS_GEMV_ARGTYPES(float)) {
  // gemv is bw bound, and does not benefit from TF32. But the precision
  // loss still happens on TF32. So we disable it here.
  // gemv 是受带宽限制的，不受 TF32 的益处。但是 TF32 仍会导致精度损失，因此我们在这里禁用它。
  NoTF32Guard disable_tf32;
  // See Note [Writing Nondeterministic Operations]
  // 参见注释 [编写非确定性操作]
  globalContext().alertCuBLASConfigNotDeterministic();
  // 获取当前 CUDA cuBLAS 句柄
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  // 根据转置标志 trans 获取 cublas 操作类型
  cublasOperation_t op = _cublasOpFromChar(trans);
  // 调整矩阵操作的 leading dimension 参数
  _cublasAdjustLdLevel2(m, n, &lda);
  // 检查参数值是否有效
  GEMV_CHECK_ARGVALUES(float);
  // 调用 cublas 函数执行单精度向量乘法
  TORCH_CUDABLAS_CHECK(
      cublasSgemv(handle, op, m, n, &alpha, a, lda, x, incx, &beta, y, incy));
}

template <>
void gemv<at::Half>(CUDABLAS_GEMV_ARGTYPES(at::Half)) {
  // In general, cublas regards matrices as column-major.
  // The cublasS/Dgemv usages in cuda::blas::gemv<float>/<double> above
  // require that external blas::gemv callers obey the following convention:
  //
  // If "a" is row-major with shape (output, summed) in blas::gemv's caller,
  // caller interprets it as column-major with shape (summed, output), passes
  // summed and output respectively to our local vars m, n, and requests that cublas
  // internally transpose ("trans") the column-major interpretation of a.
  //
  // There's no such thing as "cublasHalfgemv", so here we hack gemv with a gemm.
  // However, we must allow the same calling convention, because the caller shouldn't
  // have to swap args based on whether it's calling blas::gemv<at::Half> or <float>.

  // 一般来说，cublas 把矩阵视为列主序。
  // 上面 cuda::blas::gemv<float>/<double> 中 cublasS/Dgemv 的使用
  // 要求外部 blas::gemv 调用者遵循以下约定：
  //
  // 如果 "a" 在 blas::gemv 调用者处以行主序形式，形状为 (output, summed)，
  // 调用者将其解释为列主序形式，形状为 (summed, output)，分别传递给我们的本地变量 m, n，
  // 并请求 cublas 在内部转置矩阵 "a" 的列主序解释。
  //
  // 没有 "cublasHalfgemv" 这样的东西，所以我们在这里使用 gemm 来模拟 gemv。
  // 然而，我们必须允许相同的调用约定，因为调用者不应该根据调用 blas::gemv<at::Half> 还是 <float> 而交换参数。

  bool trans_bool = (_cublasOpFromChar(trans) != CUBLAS_OP_N);
  if (trans_bool) {
    std::swap(m, n);
  }
  // 交换变量 m 和 n 的值，以确保 m 是输出的大小，n 是输入向量 x 的元素数
  // 不论 gemv<> 的调用者使用的是行优先还是列优先顺序

  // 如果 incy > 1，将向量 y 解释为列优先矩阵，只有一行（形状为 (1, output)）并且 leading dim 是 incy。
  // trans(a)*x 计算出的矩阵只有一列（形状为 (output, 1)），与 y 不匹配。
  // 因此，我们将 x 解释为类似于 y 的列优先矩阵，只有一行（形状为 (1, summed)）并且 leading dim 是 incx。
  // gemm 然后执行 x*transpose(trans(a))，产生一个只有一行的矩阵（形状为 (1, output)），与 y 匹配。
  char trans_flipped = (trans_bool ? 'n' : 't');
  // 调用 gemm 函数，使用半精度（at::Half），执行矩阵乘法计算
  gemm<at::Half>(
      'n', trans_flipped, 1, m, n, alpha, x, incx, a, lda, beta, y, incy);
/* LEVEL 1 BLAS FUNCTIONS */

// 模板特化：双精度浮点数的 dot 函数实现
template <>
void dot<double>(CUDABLAS_DOT_ARGTYPES(double)) {
  // 调用 cuBLAS 的双精度浮点数点积函数 cublasDdot
  TORCH_CUDABLAS_CHECK(cublasDdot(handle, n, x, incx, y, incy, result));
}

// 模板特化：单精度浮点数的 dot 函数实现
template <>
void dot<float>(CUDABLAS_DOT_ARGTYPES(float)) {
  // 调用 cuBLAS 的单精度浮点数点积函数 cublasSdot
  TORCH_CUDABLAS_CHECK(cublasSdot(handle, n, x, incx, y, incy, result));
}

// 模板特化：双精度复数的 dot 函数实现
template <>
void dot<c10::complex<double>>(CUDABLAS_DOT_ARGTYPES(c10::complex<double>)) {
  // 调用 cuBLAS 的双精度复数点积函数 cublasZdotu
  TORCH_CUDABLAS_CHECK(cublasZdotu(handle, n, reinterpret_cast<const cuDoubleComplex*>(x),
                                   incx, reinterpret_cast<const cuDoubleComplex*>(y), incy,
                                   reinterpret_cast<cuDoubleComplex*>(result)));
}

// 模板特化：单精度复数的 dot 函数实现
template <>
void dot<c10::complex<float>>(CUDABLAS_DOT_ARGTYPES(c10::complex<float>)) {
  // 调用 cuBLAS 的单精度复数点积函数 cublasCdotu
  TORCH_CUDABLAS_CHECK(cublasCdotu(handle, n, reinterpret_cast<const cuComplex*>(x),
                                   incx, reinterpret_cast<const cuComplex*>(y), incy,
                                   reinterpret_cast<cuComplex*>(result)));
}

// 模板特化：半精度浮点数的 dot 函数实现
template <>
void dot<at::Half>(CUDABLAS_DOT_ARGTYPES(at::Half)) {
  // 调用 cuBLAS 的半精度浮点数点积函数 cublasDotEx
  TORCH_CUDABLAS_CHECK(cublasDotEx(
      handle,
      n,
      x,
      CUDA_R_16F,
      incx,
      y,
      CUDA_R_16F,
      incy,
      result,
      CUDA_R_16F,
      CUDA_R_32F));
}

// 模板特化：BF16（Brain Float 16）的 dot 函数实现
template <>
void dot<at::BFloat16>(CUDABLAS_DOT_ARGTYPES(at::BFloat16)) {
  // 调用 cuBLAS 的 BF16 点积函数 cublasDotEx
  TORCH_CUDABLAS_CHECK(cublasDotEx(
      handle,
      n,
      x,
      CUDA_R_16BF,
      incx,
      y,
      CUDA_R_16BF,
      incy,
      result,
      CUDA_R_16BF,
      CUDA_R_32F));
}

// 模板特化：单精度复数的向共轭点积函数实现
template <>
void vdot<c10::complex<float>>(CUDABLAS_DOT_ARGTYPES(c10::complex<float>)) {
  // 调用 cuBLAS 的单精度复数向共轭点积函数 cublasCdotc
  TORCH_CUDABLAS_CHECK(cublasCdotc(handle, n, reinterpret_cast<const cuComplex*>(x),
                                   incx, reinterpret_cast<const cuComplex*>(y), incy,
                                   reinterpret_cast<cuComplex*>(result)));
}

// 模板特化：双精度复数的向共轭点积函数实现
template <>
void vdot<c10::complex<double>>(CUDABLAS_DOT_ARGTYPES(c10::complex<double>)) {
  // 调用 cuBLAS 的双精度复数向共轭点积函数 cublasZdotc
  TORCH_CUDABLAS_CHECK(cublasZdotc(handle, n, reinterpret_cast<const cuDoubleComplex*>(x),
                                   incx, reinterpret_cast<const cuDoubleComplex*>(y), incy,
                                   reinterpret_cast<cuDoubleComplex*>(result)));
}

// 模板特化：单精度浮点数批量解线性方程组函数实现
template <>
void getrsBatched<float>(CUDABLAS_GETRS_ARGTYPES(float)) {
  // 调用 cuBLAS 的单精度浮点数批量解线性方程组函数 cublasSgetrsBatched
  TORCH_CUDABLAS_CHECK(cublasSgetrsBatched(
      handle,
      trans,
      n,
      nrhs,
      dA_array,
      lda,
      ipiv_array,
      dB_array,
      ldb,
      info_array,
      batchsize));
}
template <>
// 特化模板函数，用于解决 double 类型的批量解线性方程组问题
void getrsBatched<double>(CUDABLAS_GETRS_ARGTYPES(double)) {
  // 调用 cuBLAS 库函数 cublasDgetrsBatched 解线性方程组
  TORCH_CUDABLAS_CHECK(cublasDgetrsBatched(
      handle,        // cuBLAS 句柄
      trans,         // 转置标志
      n,             // 矩阵的阶数
      nrhs,          // 右侧矩阵的列数
      dA_array,      // 存储 A 矩阵的设备指针数组
      lda,           // A 矩阵的每列偏移量
      ipiv_array,    // 存储主元的设备指针数组
      dB_array,      // 存储 B 矩阵的设备指针数组
      ldb,           // B 矩阵的每列偏移量
      info_array,    // 存储信息码的设备指针数组
      batchsize));   // 批处理大小
}

template <>
// 特化模板函数，用于解决 c10::complex<float> 类型的批量解线性方程组问题
void getrsBatched<c10::complex<float>>(CUDABLAS_GETRS_ARGTYPES(c10::complex<float>)) {
  // 调用 cuBLAS 库函数 cublasCgetrsBatched 解线性方程组
  TORCH_CUDABLAS_CHECK(cublasCgetrsBatched(
      handle,                                    // cuBLAS 句柄
      trans,                                     // 转置标志
      n,                                         // 矩阵的阶数
      nrhs,                                      // 右侧矩阵的列数
      reinterpret_cast<cuComplex**>(dA_array),   // 存储 A 矩阵的复数设备指针数组
      lda,                                       // A 矩阵的每列偏移量
      ipiv_array,                                // 存储主元的设备指针数组
      reinterpret_cast<cuComplex**>(dB_array),   // 存储 B 矩阵的复数设备指针数组
      ldb,                                       // B 矩阵的每列偏移量
      info_array,                                // 存储信息码的设备指针数组
      batchsize));                               // 批处理大小
}

template <>
// 特化模板函数，用于解决 c10::complex<double> 类型的批量解线性方程组问题
void getrsBatched<c10::complex<double>>(CUDABLAS_GETRS_ARGTYPES(c10::complex<double>)) {
  // 调用 cuBLAS 库函数 cublasZgetrsBatched 解线性方程组
  TORCH_CUDABLAS_CHECK(cublasZgetrsBatched(
      handle,                                        // cuBLAS 句柄
      trans,                                         // 转置标志
      n,                                             // 矩阵的阶数
      nrhs,                                          // 右侧矩阵的列数
      reinterpret_cast<cuDoubleComplex**>(dA_array),  // 存储 A 矩阵的复数设备指针数组
      lda,                                           // A 矩阵的每列偏移量
      ipiv_array,                                    // 存储主元的设备指针数组
      reinterpret_cast<cuDoubleComplex**>(dB_array),  // 存储 B 矩阵的复数设备指针数组
      ldb,                                           // B 矩阵的每列偏移量
      info_array,                                    // 存储信息码的设备指针数组
      batchsize));                                   // 批处理大小
}

template <>
// 特化模板函数，用于解决 float 类型的批量 QR 分解问题
void geqrfBatched<float>(CUDABLAS_GEQRF_BATCHED_ARGTYPES(float)) {
  // 调用 cuBLAS 库函数 cublasSgeqrfBatched 进行批量 QR 分解
  TORCH_CUDABLAS_CHECK(cublasSgeqrfBatched(
      handle,        // cuBLAS 句柄
      m,             // 矩阵的行数
      n,             // 矩阵的列数
      A_array,       // 存储 A 矩阵的设备指针数组
      lda,           // A 矩阵的每列偏移量
      tau_array,     // 存储 Householder 向量的设备指针数组
      info,          // 存储信息码的设备指针
      batchsize));   // 批处理大小
}

template <>
// 特化模板函数，用于解决 double 类型的批量 QR 分解问题
void geqrfBatched<double>(CUDABLAS_GEQRF_BATCHED_ARGTYPES(double)) {
  // 调用 cuBLAS 库函数 cublasDgeqrfBatched 进行批量 QR 分解
  TORCH_CUDABLAS_CHECK(cublasDgeqrfBatched(
      handle,        // cuBLAS 句柄
      m,             // 矩阵的行数
      n,             // 矩阵的列数
      A_array,       // 存储 A 矩阵的设备指针数组
      lda,           // A 矩阵的每列偏移量
      tau_array,     // 存储 Householder 向量的设备指针数组
      info,          // 存储信息码的设备指针
      batchsize));   // 批处理大小
}

template <>
// 特化模板函数，用于解决 c10::complex<float> 类型的批量 QR 分解问题
void geqrfBatched<c10::complex<float>>(
    CUDABLAS_GEQRF_BATCHED_ARGTYPES(c10::complex<float>)) {
  // 调用 cuBLAS 库函数 cublasCgeqrfBatched 进行批量 QR 分解
  TORCH_CUDABLAS_CHECK(cublasCgeqrfBatched(
      handle,                                        // cuBLAS 句柄
      m,                                             // 矩阵的行数
      n,                                             // 矩阵的列数
      reinterpret_cast<cuComplex**>(A_array),         // 存储 A 矩阵的复数设备指针数组
      lda,                                           // A 矩阵的每列偏移量
      reinterpret_cast<cuComplex**>(tau_array),      // 存储 Householder 向量的复数设备指针数组
      info,                                          // 存储信息码的设备指针
      batchsize));                                   // 批处理大小
}

template <>
// 特化模板函数，用于解决 c10::complex<double> 类型的批量 QR 分解问题
void geqrfBatched<c10::complex<double>>(
    CUDABLAS_GEQRF_BATCHED_ARGTYPES(c10::complex<double>)) {
  // 调用 cuBLAS 库函数 cublasZgeqrfBatched 进行批量 QR 分解
  TORCH_CUDABLAS_CHECK(cublasZgeqrfBatched(
      handle,                                            // cuBLAS 句柄
      m,                                                 // 矩阵的行数
      n,                                                 // 矩阵的列数
      reinterpret_cast<cuDoubleComplex**>(A_array),       // 存储 A 矩阵的复数设备指针数组
      lda,                                               // A 矩阵的每列偏移量
      reinterpret_cast<cuDoubleComplex**>(tau_array),    // 存储 Householder 向量的复数设备指针数组
      info,                                              // 存储信息码的设备指针
      batchsize));                                       // 批处理大小
}

template <>
// 特化模板函数，用于解决 double 类型的批量 LU 分解问题
void getrfBatched<double>(
    int n, double** dA_array, int ldda, int* ipiv_array, int* info_array, int batchsize) {
  // 调用 cuBLAS 库函数 cublasDgetrfBatched 进行批量 LU 分解
  auto handle = at::cuda::getCurrentCUDAB
    int batchsize) {


    // 定义一个函数，用于在 CUDA 中执行批处理的 Zgetrf 操作
    auto handle = at::cuda::getCurrentCUDABlasHandle();
    // 获取当前 CUDA 的 cuBLAS 句柄
    
    TORCH_CUDABLAS_CHECK(cublasZgetrfBatched(
        handle,
        n,
        reinterpret_cast<cuDoubleComplex**>(dA_array),
        ldda,
        ipiv_array,
        info_array,
        batchsize));
    // 调用 cuBLAS 库的批处理 Zgetrf 函数，对一批次的双精度复数矩阵进行 LU 分解，
    // 参数分别为：CUDA cuBLAS 句柄，矩阵数量 n，矩阵数组 dA_array 的指针，
    // 每个矩阵的 leading dimension ldda，存储置换信息的数组 ipiv_array，
    // 存储操作结果的数组 info_array，以及批次大小 batchsize
} // 结束 at::cuda::blas 命名空间的定义

template <>
void getrfBatched<c10::complex<float>>(
    int n,
    c10::complex<float>** dA_array,
    int ldda,
    int* ipiv_array,
    int* info_array,
    int batchsize) {
  // 获取当前 CUDA cuBLAS 句柄
  auto handle = at::cuda::getCurrentCUDABlasHandle();
  // 调用 cuBLAS 批量复杂浮点 LU 分解操作
  TORCH_CUDABLAS_CHECK(cublasCgetrfBatched(
      handle,
      n,
      reinterpret_cast<cuComplex**>(dA_array),
      ldda,
      ipiv_array,
      info_array,
      batchsize));
}

template <>
void gelsBatched<double>(CUDABLAS_GELS_BATCHED_ARGTYPES(double)) {
  // 调用 cuBLAS 批量双精度解线性最小二乘问题操作
  TORCH_CUDABLAS_CHECK(cublasDgelsBatched(
      handle, trans, m, n, nrhs, dA_array, ldda, dC_array, lddc, info, devInfoArray, batchSize));
}

template <>
void gelsBatched<float>(CUDABLAS_GELS_BATCHED_ARGTYPES(float)) {
  // 调用 cuBLAS 批量单精度解线性最小二乘问题操作
  TORCH_CUDABLAS_CHECK(cublasSgelsBatched(
      handle, trans, m, n, nrhs, dA_array, ldda, dC_array, lddc, info, devInfoArray, batchSize));
}

template <>
void gelsBatched<c10::complex<double>>(CUDABLAS_GELS_BATCHED_ARGTYPES(c10::complex<double>)) {
  // 调用 cuBLAS 批量复杂双精度解线性最小二乘问题操作
  TORCH_CUDABLAS_CHECK(cublasZgelsBatched(
      handle, trans,
      m, n, nrhs,
      reinterpret_cast<cuDoubleComplex**>(dA_array),
      ldda,
      reinterpret_cast<cuDoubleComplex**>(dC_array),
      lddc,
      info,
      devInfoArray,
      batchSize));
}

template <>
void gelsBatched<c10::complex<float>>(CUDABLAS_GELS_BATCHED_ARGTYPES(c10::complex<float>)) {
  // 调用 cuBLAS 批量复杂单精度解线性最小二乘问题操作
  TORCH_CUDABLAS_CHECK(cublasCgelsBatched(
      handle, trans,
      m, n, nrhs,
      reinterpret_cast<cuComplex**>(dA_array),
      ldda,
      reinterpret_cast<cuComplex**>(dC_array),
      lddc,
      info,
      devInfoArray,
      batchSize));
}

} // 结束 at::cuda::blas 命名空间的定义
```