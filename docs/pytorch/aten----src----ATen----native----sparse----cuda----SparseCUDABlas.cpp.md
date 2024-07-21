# `.\pytorch\aten\src\ATen\native\sparse\cuda\SparseCUDABlas.cpp`

```py
// 定义编译选项，仅包含方法操作符
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

// 引入CUDA环境头文件
#include <ATen/cuda/CUDAContext.h>
// 引入异常处理工具
#include <c10/util/Exception.h>
// 引入CUDA异常处理工具
#include <ATen/cuda/Exceptions.h>
// 引入CUDA稀疏矩阵乘法的CUDA实现头文件
#include <ATen/native/sparse/cuda/SparseCUDABlas.h>
// 引入CUDA内存分配器
#include <c10/cuda/CUDACachingAllocator.h>

// 引入cuSPARSE库头文件
#include <cusparse.h>

// cuSPARSE的稀疏矩阵乘法（cusparseSpMM）的限制和支持声明
// 在CUDA 11.0版本以上的通用API在所有平台上都可用
// 在CUDA 10.1+版本中，除了Windows系统外，在所有平台上都可用
// 在其它系统中使用这些API将导致编译时或运行时失败
// 后续版本中将扩展其支持
#if defined(CUDART_VERSION) && (CUSPARSE_VERSION >= 11000 || (!defined(_MSC_VER) && CUSPARSE_VERSION >= 10301))
#define IS_SPMM_AVAILABLE() 1
#else
#define IS_SPMM_AVAILABLE() 0
#endif

// 是否支持cuSPARSE的HIP实现（IS_SPMM_HIP_AVAILABLE）
#if defined(USE_ROCM)
#define IS_SPMM_HIP_AVAILABLE() 1
#else
#define IS_SPMM_HIP_AVAILABLE() 0
#endif

// 如果cuSPARSE的稀疏矩阵乘法API可用，或者cuSPARSE的HIP实现可用，则引入库类型头文件
#if IS_SPMM_AVAILABLE() || IS_SPMM_HIP_AVAILABLE()
#include <library_types.h>
#endif

// 如果cuSPARSE版本低于10100或未定义CUSPARSE_VERSION，则定义cuSPARSE错误字符串函数
#if !defined(CUSPARSE_VERSION) || (CUSPARSE_VERSION < 10100)
const char* cusparseGetErrorString(cusparseStatus_t status) {
  switch(status)
  {
    case CUSPARSE_STATUS_SUCCESS:
      return "success";

    case CUSPARSE_STATUS_NOT_INITIALIZED:
      return "library not initialized";

    case CUSPARSE_STATUS_ALLOC_FAILED:
      return "resource allocation failed";

    case CUSPARSE_STATUS_INVALID_VALUE:
      return "an invalid numeric value was used as an argument";

    case CUSPARSE_STATUS_ARCH_MISMATCH:
      return "an absent device architectural feature is required";

    case CUSPARSE_STATUS_MAPPING_ERROR:
      return "an access to GPU memory space failed";

    case CUSPARSE_STATUS_EXECUTION_FAILED:
      return "the GPU program failed to execute";

    case CUSPARSE_STATUS_INTERNAL_ERROR:
      return "an internal operation failed";

    case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
      return "the matrix type is not supported by this function";

    case CUSPARSE_STATUS_ZERO_PIVOT:
      return "an entry of the matrix is either structural zero or numerical zero (singular block)";

    default:
      return "unknown error";
  }
}
#endif

// 定义在命名空间 at::native::sparse::cuda 下的函数 Xcoo2csr
namespace at::native::sparse::cuda {

// 将COO格式的稀疏矩阵转换为CSR格式的函数
void Xcoo2csr(const int *coorowind, int64_t nnz, int64_t m, int *csrrowptr) {
  // 检查m和nnz的值是否符合cusparseXcoo2csr函数的要求
  TORCH_CHECK((m <= INT_MAX) && (nnz <= INT_MAX),
    "cusparseXcoo2csr only supports m, nnz with the bound [val] <= ",
    INT_MAX);

  // 将nnz和m转换为int类型
  int i_nnz = (int)nnz;
  int i_m = (int)m;

  // 获取当前CUDA稀疏库的句柄
  auto handle = at::cuda::getCurrentCUDASparseHandle();
  // 调用cuSPARSE函数将COO格式转换为CSR格式
  TORCH_CUDASPARSE_CHECK(cusparseXcoo2csr(handle, coorowind, i_nnz, i_m, csrrowptr, CUSPARSE_INDEX_BASE_ZERO));
}

// 将char类型的转置标志转换为cuSPARSE的操作类型
cusparseOperation_t convertTransToCusparseOperation(char trans) {
  if (trans == 't') return CUSPARSE_OPERATION_TRANSPOSE;
  else if (trans == 'n') return CUSPARSE_OPERATION_NON_TRANSPOSE;
  else if (trans == 'c') return CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE;
  else {
    // 如果输入的转置标志不在支持的范围内，则抛出错误
    AT_ERROR("trans must be one of: t, n, c");
  }
}

// 如果cuSPARSE的稀疏矩阵乘法API可用，或者cuSPARSE的HIP实现可用，则定义匿名命名空间
#if IS_SPMM_AVAILABLE() || IS_SPMM_HIP_AVAILABLE()

// 定义一个模板类或函数（未完整展示）
void _csrmm2(
  char transa, char transb,
  int64_t m, int64_t n, int64_t k, int64_t nnz,
  T *alpha, T *csrvala, int *csrrowptra, int *csrcolinda,
  T *b, int64_t ldb, T *beta, T *c, int64_t ldc,
  cudaDataType cusparse_value_type)
{
  // 如果输入中有空指针，直接返回，不执行后续操作
  if (csrvala == nullptr || b == nullptr || c == nullptr) return;

  // 将 transa 和 transb 转换为 cusparseOperation_t 类型
  cusparseOperation_t opa = convertTransToCusparseOperation(transa);
  cusparseOperation_t opb = convertTransToCusparseOperation(transb);

  // cusparseSpMM 实际上支持 int64_t 类型的参数。
  // 为了支持 int64_t，这里的索引指针 csrrowptra 和 csrcolinda 必须作为 int64_t 类型传递。
  TORCH_CHECK((m <= INT_MAX) && (n <= INT_MAX) && (k <= INT_MAX) && (nnz <= INT_MAX) && (ldb <= INT_MAX) && (ldc <= INT_MAX),
    "目前 cusparseSpMM 仅支持 m、n、k、nnz、ldb、ldc 的值小于等于 ", INT_MAX, "。",
    "如果需要更大的值，请在 GitHub 上提交一个问题。"
  );

  // 根据 transa 的值确定 ma 和 ka
  int64_t ma = m, ka = k;
  if (transa != 'n') std::swap(ma, ka);

  // 创建稀疏矩阵描述符 descA
  cusparseSpMatDescr_t descA;
  TORCH_CUDASPARSE_CHECK(cusparseCreateCsr(
    &descA,                     /* 输出 */
    ma, ka, nnz,                /* 行数、列数、非零元素个数 */
    csrrowptra,                 /* 稀疏矩阵的行偏移量，大小为行数+1 */
    csrcolinda,                 /* 稀疏矩阵的列索引，大小为 nnz */
    csrvala,                    /* 稀疏矩阵的值，大小为 nnz */
    CUSPARSE_INDEX_32I,         /* 行偏移量索引的数据类型 */
    CUSPARSE_INDEX_32I,         /* 列索引的数据类型 */
    CUSPARSE_INDEX_BASE_ZERO,   /* 行偏移量和列索引的起始索引 */
    cusparse_value_type         /* 值的数据类型 */
  ));

  // 根据 transb 的值确定 kb 和 nb
  int64_t kb = k, nb = n;
  if (transb != 'n') std::swap(kb, nb);

  // 创建稠密矩阵描述符 descB
  cusparseDnMatDescr_t descB;
  TORCH_CUDASPARSE_CHECK(cusparseCreateDnMat(
    &descB,               /* 输出 */
    kb, nb, ldb,          /* 行数、列数、领先维度 */
    b,                    /* 值 */
    cusparse_value_type,  /* 值的数据类型 */
    CUSPARSE_ORDER_COL    /* 存储布局，目前只支持列主序 */
  ));

  // 创建稠密矩阵描述符 descC
  cusparseDnMatDescr_t descC;
  TORCH_CUDASPARSE_CHECK(cusparseCreateDnMat(
    &descC,               /* 输出 */
    m, n, ldc,            /* 行数、列数、领先维度 */
    c,                    /* 值 */
    cusparse_value_type,  /* 值的数据类型 */
    CUSPARSE_ORDER_COL    /* 存储布局，目前只支持列主序 */
  ));


  auto handle = at::cuda::getCurrentCUDASparseHandle();
  cudaDeviceProp* prop = at::cuda::getCurrentDeviceProperties();
  // 在 CUDA 11.8+ 版本中，SM89 上的 ALG1 算法存在问题
#if !defined(USE_ROCM)
  auto default_alg = prop->major == 8 && prop->minor == 9 ? CUSPARSE_SPMM_CSR_ALG2 : CUSPARSE_SPMM_CSR_ALG1;
#else
  auto default_alg = CUSPARSE_SPMM_CSR_ALG1;
#endif

  // 获取 cusparseSpMM 所需的缓冲区大小
  size_t bufferSize;
  TORCH_CUDASPARSE_CHECK(cusparseSpMM_bufferSize(
    handle, opa, opb,
    alpha,                    # 乘法运算的系数
    descA, descB,             # 稀疏矩阵 A 和 B 的描述符
    beta,                     # 加法运算的系数
    descC,                    # 稀疏矩阵 C 的描述符
    cusparse_value_type,      /* 计算执行时使用的数据类型 */
    default_alg,              /* CSR 稀疏矩阵格式的默认计算算法 */
    &bufferSize               /* 输出：计算所需的缓冲区大小 */
  ));

  auto& allocator = *c10::cuda::CUDACachingAllocator::get();
  auto dataPtr = allocator.allocate(bufferSize);

  TORCH_CUDASPARSE_CHECK(cusparseSpMM(
    handle, opa, opb,
    alpha,                    # 乘法运算的系数
    descA, descB,             # 稀疏矩阵 A 和 B 的描述符
    beta,                     # 加法运算的系数
    descC,                    # 稀疏矩阵 C 的描述符
    cusparse_value_type,      /* 计算执行时使用的数据类型 */
    default_alg,              /* CSR 稀疏矩阵格式的默认计算算法 */
    dataPtr.get()             /* 外部缓冲区 */
  ));

  TORCH_CUDASPARSE_CHECK(cusparseDestroySpMat(descA));  # 销毁稀疏矩阵 A 的描述符
  TORCH_CUDASPARSE_CHECK(cusparseDestroyDnMat(descB));  # 销毁稠密矩阵 B 的描述符
  TORCH_CUDASPARSE_CHECK(cusparseDestroyDnMat(descC));  # 销毁稠密矩阵 C 的描述符

  // TODO: Proper fix is to create real descriptor classes
#else

// 如果未定义 CUDA 使用的情况下，定义一个调整矩阵维度的函数
void adjustLd(char transb, int64_t m, int64_t n, int64_t k, int64_t *ldb, int64_t *ldc)
{
  // 判断是否需要转置操作
  int transb_ = ((transb == 't') || (transb == 'T'));

  // 如果输出矩阵是列向量，调整输出矩阵的列维度为行维度
  if(n == 1)
    *ldc = m;

  // 如果需要转置输入矩阵，并且输入矩阵是行向量，调整输入矩阵的列维度为列维度
  if(transb_)
  {
    if(k == 1)
      *ldb = n;
  }
  // 如果不需要转置输入矩阵，并且输出矩阵是列向量，调整输入矩阵的列维度为行维度
  else
  {
    if(n == 1)
      *ldb = k;
  }
}

// 定义一个特化版本的矩阵乘法函数，处理 float 类型的数据
void Scsrmm2(char transa, char transb, int64_t m, int64_t n, int64_t k, int64_t nnz, 
             const float *alpha, const float *csrvala, int *csrrowptra, int *csrcolinda, 
             const float *b, int64_t ldb, const float *beta, float *c, int64_t ldc)
{
  // 调整传输标记和矩阵维度，确保它们符合要求
  adjustLd(transb, m, n, k, &ldb, &ldc);
  
  // 将传输标记转换为cuSPARSE库所需的操作类型
  cusparseOperation_t opa = convertTransToCusparseOperation(transa);
  cusparseOperation_t opb = convertTransToCusparseOperation(transb);

  // 检查矩阵维度和其他参数是否不超过INT_MAX，否则报错
  TORCH_CHECK((m <= INT_MAX) && (n <= INT_MAX) && (k <= INT_MAX) && (nnz <= INT_MAX)  && (ldb <= INT_MAX) && (ldc <= INT_MAX),
    "cusparseScsrmm2 only supports m, n, k, nnz, ldb, ldc with the bound [val] <= ", INT_MAX);
  
  // 将int64_t类型的维度转换为int类型
  int i_m = (int)m;
  int i_n = (int)n;
  int i_k = (int)k;
  int i_nnz = (int)nnz;
  int i_ldb = (int)ldb;
  int i_ldc = (int)ldc;

  // 获取当前CUDA稀疏操作句柄
  auto handle = at::cuda::getCurrentCUDASparseHandle();

  // 创建cuSPARSE矩阵描述符
  cusparseMatDescr_t desc;
  cusparseCreateMatDescr(&desc);

  // 调用cuSPARSE库中的稀疏矩阵乘法运算
  TORCH_CUDASPARSE_CHECK(cusparseScsrmm2(handle, opa, opb, i_m, i_n, i_k, i_nnz, alpha, desc, csrvala, csrrowptra, csrcolinda, b, i_ldb, beta, c, i_ldc));
  
  // 销毁cuSPARSE矩阵描述符
  TORCH_CUDASPARSE_CHECK(cusparseDestroyMatDescr(desc));
}

void Dcsrmm2(char transa, char transb, int64_t m, int64_t n, int64_t k, int64_t nnz, const double *alpha, const double *csrvala, int *csrrowptra, int *csrcolinda, const double *b, int64_t ldb, const double *beta, double *c, int64_t ldc)
{
  // 调用辅助函数，调整传输标记和矩阵维度
  adjustLd(transb, m, n, k, &ldb, &ldc);

  // 将传输标记转换为cuSPARSE库所需的操作类型
  cusparseOperation_t opa = convertTransToCusparseOperation(transa);
  cusparseOperation_t opb = convertTransToCusparseOperation(transb);

  // 检查矩阵维度和其他参数是否不超过INT_MAX，否则报错
  TORCH_CHECK((m <= INT_MAX) && (n <= INT_MAX) && (k <= INT_MAX) && (nnz <= INT_MAX)  && (ldb <= INT_MAX) && (ldc <= INT_MAX),
    "cusparseDcsrmm2 only supports m, n, k, nnz, ldb, ldc with the bound [val] <= ", INT_MAX);

  // 将int64_t类型的维度转换为int类型
  int i_m = (int)m;
  int i_n = (int)n;
  int i_k = (int)k;
  int i_nnz = (int)nnz;
  int i_ldb = (int)ldb;
  int i_ldc = (int)ldc;

  // 获取当前CUDA稀疏操作句柄
  auto handle = at::cuda::getCurrentCUDASparseHandle();

  // 创建cuSPARSE矩阵描述符
  cusparseMatDescr_t desc;
  cusparseCreateMatDescr(&desc);

  // 调用cuSPARSE库中的稀疏矩阵乘法运算
  TORCH_CUDASPARSE_CHECK(cusparseDcsrmm2(handle, opa, opb, i_m, i_n, i_k, i_nnz, alpha, desc, csrvala, csrrowptra, csrcolinda, b, i_ldb, beta, c, i_ldc));

  // 销毁cuSPARSE矩阵描述符
  TORCH_CUDASPARSE_CHECK(cusparseDestroyMatDescr(desc));

  // TODO: Proper fix is to create real descriptor classes
}

template<class complex_target_t>
void Ccsrmm2(char transa, char transb, int64_t m, int64_t n, int64_t k, int64_t nnz, const complex_target_t *alpha, const complex_target_t *csrvala, int *csrrowptra, int *csrcolinda, const complex_target_t *b, int64_t ldb, const complex_target_t *beta, complex_target_t *c, int64_t ldc)
{
  // 调用辅助函数，调整传输标记和矩阵维度
  adjustLd(transb, m, n, k, &ldb, &ldc);

  // 将传输标记转换为cuSPARSE库所需的操作类型
  cusparseOperation_t opa = convertTransToCusparseOperation(transa);
  cusparseOperation_t opb = convertTransToCusparseOperation(transb);

  // 检查矩阵维度和其他参数是否不超过INT_MAX，否则报错
  TORCH_CHECK((m <= INT_MAX) && (n <= INT_MAX) && (k <= INT_MAX) && (nnz <= INT_MAX)  && (ldb <= INT_MAX) && (ldc <= INT_MAX),

    // cuSPARSE只支持维度和nnz（非零元素数）小于等于INT_MAX的操作
    "cusparseCcsrmm2 only supports m, n, k, nnz, ldb, ldc with the bound [val] <= ", INT_MAX);
    # 输出一条带有整数最大值的错误信息，说明 cusparseCcsrmm2 只支持特定参数 m, n, k, nnz, ldb, ldc 的取值范围
        "cusparseCcsrmm2 only supports m, n, k, nnz, ldb, ldc with the bound [val] <= ", INT_MAX);
    # 将浮点数 m 转换为整数 i_m
      int i_m = (int)m;
    # 将浮点数 n 转换为整数 i_n
      int i_n = (int)n;
    # 将浮点数 k 转换为整数 i_k
      int i_k = (int)k;
    # 将浮点数 nnz 转换为整数 i_nnz
      int i_nnz = (int)nnz;
    # 将浮点数 ldb 转换为整数 i_ldb
      int i_ldb = (int)ldb;
    # 将浮点数 ldc 转换为整数 i_ldc
      int i_ldc = (int)ldc;
    
    # 获取当前 CUDA 环境下的稀疏运算句柄
      auto handle = at::cuda::getCurrentCUDASparseHandle();
    # 创建一个稀疏矩阵描述符 desc
      cusparseMatDescr_t desc;
      cusparseCreateMatDescr(&desc);
    # 调用 cusparseCcsrmm2 函数进行 CSR 矩阵乘法运算，并检查运行结果
      TORCH_CUDASPARSE_CHECK(cusparseCcsrmm2(handle, opa, opb, i_m, i_n, i_k, i_nnz, alpha, desc, csrvala, csrrowptra, csrcolinda, b, i_ldb, beta, c, i_ldc));
    # 销毁之前创建的稀疏矩阵描述符 desc
      TORCH_CUDASPARSE_CHECK(cusparseDestroyMatDescr(desc));
}

// 定义模板函数 Zcsrmm2，处理复杂目标类型的稀疏矩阵乘法
template<class complex_target_t>
void Zcsrmm2(char transa, char transb, int64_t m, int64_t n, int64_t k, int64_t nnz,
             const complex_target_t *alpha, const complex_target_t *csrvala, int *csrrowptra, int *csrcolinda,
             const complex_target_t *b, int64_t ldb, const complex_target_t *beta, complex_target_t *c, int64_t ldc)
{
  // 调整 ldb 和 ldc 的值，确保在边界内
  adjustLd(transb, m, n, k, &ldb, &ldc);

  // 将 transa 和 transb 转换为 cusparseOperation_t 类型
  cusparseOperation_t opa = convertTransToCusparseOperation(transa);
  cusparseOperation_t opb = convertTransToCusparseOperation(transb);

  // 检查 m、n、k、nnz、ldb、ldc 是否在 INT_MAX 内，否则报错
  TORCH_CHECK((m <= INT_MAX) && (n <= INT_MAX) && (k <= INT_MAX) && (nnz <= INT_MAX) && (ldb <= INT_MAX) && (ldc <= INT_MAX),
              "cusparseZcsrmm2 only supports m, n, k, nnz, ldb, ldc with the bound [val] <= ", INT_MAX);

  // 将 m、n、k、nnz、ldb、ldc 转换为 int 类型
  int i_m = (int)m;
  int i_n = (int)n;
  int i_k = (int)k;
  int i_nnz = (int)nnz;
  int i_ldb = (int)ldb;
  int i_ldc = (int)ldc;

  // 获取当前 CUDA 稀疏操作的句柄
  auto handle = at::cuda::getCurrentCUDASparseHandle();

  // 创建稀疏矩阵描述符
  cusparseMatDescr_t desc;
  cusparseCreateMatDescr(&desc);

  // 调用 cusparseZcsrmm2 执行稀疏矩阵乘法
  TORCH_CUDASPARSE_CHECK(cusparseZcsrmm2(handle, opa, opb, i_m, i_n, i_k, i_nnz, alpha, desc, csrvala, csrrowptra, csrcolinda, b, i_ldb, beta, c, i_ldc));

  // 销毁稀疏矩阵描述符
  TORCH_CUDASPARSE_CHECK(cusparseDestroyMatDescr(desc));
}

// 模板函数 csrmm2 的通用定义，要求 T 类型为 float 或 double
template<typename T>
void csrmm2(
  char transa, char transb,
  int64_t m, int64_t n, int64_t k, int64_t nnz,
  T alpha, T *csrvala, int *csrrowptra, int *csrcolinda,
  T *b, int64_t ldb, T beta, T *c, int64_t ldc)
{
  // 静态断言，确保只支持 float、double、cfloat 和 cdouble 类型的数据
  static_assert(false&&sizeof(T), "cusparse csr MM only supports data type of float, double, cfloat and cdouble.");
}

// csrmm2 函数模板的特化版本，处理 float 类型
template<> void csrmm2<float>(
  char transa, char transb,
  int64_t m, int64_t n, int64_t k, int64_t nnz,
  float alpha, float *csrvala, int *csrrowptra, int *csrcolinda,
  float *b, int64_t ldb, float beta, float *c, int64_t ldc)
{
  // 调用 Scsrmm2 函数处理 float 类型的 csrmm2 操作
  Scsrmm2(transa, transb, m, n, k, nnz, &alpha, csrvala, csrrowptra, csrcolinda, b, ldb, &beta, c, ldc);
}

// csrmm2 函数模板的特化版本，处理 double 类型
template<> void csrmm2<double>(
  char transa, char transb,
  int64_t m, int64_t n, int64_t k, int64_t nnz,
  double alpha, double *csrvala, int *csrrowptra, int *csrcolinda,
  double *b, int64_t ldb, double beta, double *c, int64_t ldc)
{
  // 调用 Dcsrmm2 函数处理 double 类型的 csrmm2 操作
  Dcsrmm2(transa, transb, m, n, k, nnz, &alpha, csrvala, csrrowptra, csrcolinda, b, ldb, &beta, c, ldc);
}

// csrmm2 函数模板的特化版本，处理 c10::complex<float> 类型
template<> void csrmm2<c10::complex<float>>(
  char transa, char transb,
  int64_t m, int64_t n, int64_t k, int64_t nnz,
  c10::complex<float> alpha, c10::complex<float> *csrvala, int *csrrowptra, int *csrcolinda,
  c10::complex<float> *b, int64_t ldb, c10::complex<float> beta, c10::complex<float> *c, int64_t ldc)
{
  // ROCM 平台下的特化处理，调用 Ccsrmm2 函数
  #ifdef USE_ROCM
  Ccsrmm2(transa, transb, m, n, k, nnz,
          reinterpret_cast<const hipComplex*>(&alpha),
          reinterpret_cast<const hipComplex*>(csrvala),
          csrrowptra,
          csrcolinda,
          reinterpret_cast<const hipComplex*>(b),
          ldb,
          reinterpret_cast<const hipComplex*>(&beta),
          c,
          ldc);
  #endif
}
    reinterpret_cast<hipComplex*>(c), ldc);
  #else
  Ccsrmm2(transa, transb, m, n, k, nnz,
      reinterpret_cast<const cuComplex*>(&alpha),
      reinterpret_cast<const cuComplex*>(csrvala),
      csrrowptra,
      csrcolinda,
      reinterpret_cast<const cuComplex*>(b),
      ldb,
      reinterpret_cast<const cuComplex*>(&beta),
      reinterpret_cast<cuComplex*>(c), ldc);
  #endif



// 如果定义了__HIP_PLATFORM_HCC__宏，则使用hipComplex类型对c进行类型重解释，并传入ldc作为参数
  reinterpret_cast<hipComplex*>(c), ldc);
// 否则，调用Ccsrmm2函数进行稀疏矩阵乘法计算，传入相应的参数：
// transa, transb：传输矩阵标志
// m, n, k：矩阵维度
// nnz：非零元素个数
// reinterpret_cast<const cuComplex*>(&alpha)：将alpha转换为cuComplex类型指针
// reinterpret_cast<const cuComplex*>(csrvala)：将csrvala转换为cuComplex类型指针
// csrrowptra：CSR格式的行指针数组
// csrcolinda：CSR格式的列索引数组
// reinterpret_cast<const cuComplex*>(b)：将b转换为cuComplex类型指针
// ldb：矩阵b的列宽
// reinterpret_cast<const cuComplex*>(&beta)：将beta转换为cuComplex类型指针
// reinterpret_cast<cuComplex*>(c)：将c转换为cuComplex类型指针，用于存储计算结果
// ldc：矩阵c的列宽
  reinterpret_cast<cuComplex*>(c), ldc);
// 结束if-else条件编译
#endif
}

// 模板特化：稀疏矩阵稠密矩阵乘法，处理复数双精度类型
template<> void csrmm2<c10::complex<double>>(
  char transa, char transb,
  int64_t m, int64_t n, int64_t k, int64_t nnz,
  c10::complex<double> alpha, c10::complex<double> *csrvala, int *csrrowptra, int *csrcolinda,
  c10::complex<double> *b, int64_t ldb, c10::complex<double> beta, c10::complex<double> *c, int64_t ldc)
{
  // 如果使用 ROCm 平台
  #ifdef USE_ROCM
  // 调用 ROCm 下的稀疏矩阵稠密矩阵乘法函数 Zcsrmm2
  Zcsrmm2(transa, transb, m, n, k, nnz,
    reinterpret_cast<const hipDoubleComplex*>(&alpha),
    reinterpret_cast<const hipDoubleComplex*>(csrvala),
    csrrowptra,
    csrcolinda,
    reinterpret_cast<const hipDoubleComplex*>(b),
    ldb,
    reinterpret_cast<const hipDoubleComplex*>(&beta),
    reinterpret_cast<hipDoubleComplex*>(c), ldc);
  // 否则
  #else
  // 调用 CUDA 下的稀疏矩阵稠密矩阵乘法函数 Zcsrmm2
  Zcsrmm2(transa, transb, m, n, k, nnz,
    reinterpret_cast<const cuDoubleComplex*>(&alpha),
    reinterpret_cast<const cuDoubleComplex*>(csrvala),
    csrrowptra,
    csrcolinda,
    reinterpret_cast<const cuDoubleComplex*>(b),
    ldb,
    reinterpret_cast<const cuDoubleComplex*>(&beta),
    reinterpret_cast<cuDoubleComplex*>(c), ldc);
  #endif
}

// 结束条件
#endif

// 创建单位置换矩阵
/* format conversion */
void CreateIdentityPermutation(int64_t nnz, int *P) {
  // 检查 nnz 是否小于 INT_MAX
  TORCH_CHECK((nnz <= INT_MAX),
    "Xcsrsort_bufferSizeExt only supports m, n, nnz with the bound [val] <= ",
    INT_MAX);
  // 将 nnz 转换为 int 类型
  int i_nnz = (int)nnz;

  // 获取当前 CUDA 稀疏处理句柄
  auto handle = at::cuda::getCurrentCUDASparseHandle();
  // 调用 cuSPARSE 函数创建单位置换矩阵
  cusparseCreateIdentityPermutation(handle, i_nnz, P);
}

// 获取 Xcsrsort_bufferSizeExt 所需缓冲区大小
void Xcsrsort_bufferSizeExt(int64_t m, int64_t n, int64_t nnz, const int *csrRowPtr, const int *csrColInd, size_t *pBufferSizeInBytes)
{
  // 检查 m、n、nnz 是否小于 INT_MAX
  TORCH_CHECK((m <= INT_MAX) && (n <= INT_MAX) && (nnz <= INT_MAX),
    "Xcsrsort_bufferSizeExt only supports m, n, nnz with the bound [val] <=",
    INT_MAX);
  // 将 m、n、nnz 转换为 int 类型
  int i_m = (int)m;
  int i_n = (int)n;
  int i_nnz = (int)nnz;

  // 获取当前 CUDA 稀疏处理句柄
  auto handle = at::cuda::getCurrentCUDASparseHandle();
  // 调用 cuSPARSE 函数获取缓冲区大小
  TORCH_CUDASPARSE_CHECK(cusparseXcsrsort_bufferSizeExt(handle, i_m, i_n, i_nnz, csrRowPtr, csrColInd, pBufferSizeInBytes));
}

// 对 CSR 格式的稀疏矩阵进行排序
void Xcsrsort(int64_t m, int64_t n, int64_t nnz, const int *csrRowPtr, int *csrColInd, int *P, void *pBuffer)
{
  // 检查 m、n、nnz 是否小于 INT_MAX
  TORCH_CHECK((m <= INT_MAX) && (n <= INT_MAX) && (nnz <= INT_MAX),
    "Xcsrsort only supports m, n, nnz with the bound [val] <= ",
    INT_MAX);
  // 将 m、n、nnz 转换为 int 类型
  int i_m = (int)m;
  int i_n = (int)n;
  int i_nnz = (int)nnz;

  // 获取当前 CUDA 稀疏处理句柄
  auto handle = at::cuda::getCurrentCUDASparseHandle();
  cusparseMatDescr_t desc;
  cusparseCreateMatDescr(&desc);
  // 调用 cuSPARSE 函数对 CSR 格式稀疏矩阵进行排序
  TORCH_CUDASPARSE_CHECK(cusparseXcsrsort(handle, i_m, i_n, i_nnz, desc, csrRowPtr, csrColInd, P, pBuffer));
  TORCH_CUDASPARSE_CHECK(cusparseDestroyMatDescr(desc));
}

// 获取 Xcoosort_bufferSizeExt 所需缓冲区大小
void Xcoosort_bufferSizeExt(int64_t m, int64_t n, int64_t nnz, const int *cooRows, const int *cooCols, size_t *pBufferSizeInBytes)
{
  // 检查 m、n、nnz 是否小于 INT_MAX
  TORCH_CHECK((m <= INT_MAX) && (n <= INT_MAX) && (nnz <= INT_MAX),
    "Xcoosort_bufferSizeExt only supports m, n, nnz with the bound [val] <= ",
    INT_MAX);
    INT_MAX);
  int i_m = (int)m;
  int i_n = (int)n;
  int i_nnz = (int)nnz;

  // 获取当前 CUDA 稀疏操作句柄
  auto handle = at::cuda::getCurrentCUDASparseHandle();
  // 调用 cuSPARSE 库函数，计算排序所需缓冲区大小
  TORCH_CUDASPARSE_CHECK(cusparseXcoosort_bufferSizeExt(handle, i_m, i_n, i_nnz, cooRows, cooCols, pBufferSizeInBytes));
} // 关闭前一个命名空间

// 按行对 COO 格式的稀疏矩阵进行排序
void XcoosortByRow(int64_t m, int64_t n, int64_t nnz, int *cooRows, int *cooCols, int *P, void *pBuffer)
{
  // 检查 m、n、nnz 是否在支持的整数范围内
  TORCH_CHECK((m <= INT_MAX) && (n <= INT_MAX) && (nnz <= INT_MAX),
    "XcoosortByRow only supports m, n, nnz with the bound [val] <= ",
    INT_MAX);
  
  // 将输入的 m、n、nnz 转换为 int 类型
  int i_m = (int)m;
  int i_n = (int)n;
  int i_nnz = (int)nnz;

  // 获取当前 CUDA 稀疏处理的句柄
  auto handle = at::cuda::getCurrentCUDASparseHandle();
  
  // 调用 CUDA 函数库中的 XcoosortByRow 函数对 COO 格式稀疏矩阵进行行排序
  TORCH_CUDASPARSE_CHECK(cusparseXcoosortByRow(handle, i_m, i_n, i_nnz, cooRows, cooCols, P, pBuffer));
}

} // 结束命名空间 at::native::sparse::cuda
```