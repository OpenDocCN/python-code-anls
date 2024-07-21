# `.\pytorch\aten\src\ATen\cuda\tunable\GemmRocblas.h`

```py
// 著作权声明和许可证信息
// 包含 CUDAContext.h 中定义的 CUDA 上下文
// 包含 TunableOp.h 中定义的可调节操作
// 包含 GemmCommon.h 中定义的 GEMM 通用功能
// 包含 StringUtil.h 中定义的字符串工具函数

// 定义 ROCBLAS_BETA_FEATURES_API 宏
#define ROCBLAS_BETA_FEATURES_API

// 包含 rocblas.h 头文件，ROCBLAS 库的主头文件

// 定义 TORCH_ROCBLAS_CHECK 宏，用于检查 ROCBLAS 函数调用是否成功
#define TORCH_ROCBLAS_CHECK(EXPR)                 \
  do {                                            \
    rocblas_status __err = EXPR;                  \
    TORCH_CHECK(__err == rocblas_status_success,  \
                "rocblas error: ",                \
                rocblas_status_to_string(__err),  \
                " when calling `" #EXPR "`");     \
  } while (0)

namespace at::cuda::tunable {

// RocBlasDataTypeFor 模板的特化，返回 float 类型的 ROCBLAS 数据类型常量
template <>
constexpr rocblas_datatype RocBlasDataTypeFor<float>() {
  return rocblas_datatype_f32_r;
}

// RocBlasDataTypeFor 模板的特化，返回 double 类型的 ROCBLAS 数据类型常量
template <>
constexpr rocblas_datatype RocBlasDataTypeFor<double>() {
  return rocblas_datatype_f64_r;
}

// RocBlasDataTypeFor 模板的特化，返回 Half 类型的 ROCBLAS 数据类型常量
template <>
constexpr rocblas_datatype RocBlasDataTypeFor<Half>() {
  return rocblas_datatype_f16_r;
}

// RocBlasDataTypeFor 模板的特化，返回 BFloat16 类型的 ROCBLAS 数据类型常量
template <>
constexpr rocblas_datatype RocBlasDataTypeFor<BFloat16>() {
  return rocblas_datatype_bf16_r;
}

// RocBlasDataTypeFor 模板的特化，返回 c10::complex<float> 类型的 ROCBLAS 数据类型常量
template <>
constexpr rocblas_datatype RocBlasDataTypeFor<c10::complex<float>>() {
  return rocblas_datatype_f32_c;
}

// RocBlasDataTypeFor 模板的特化，返回 c10::complex<double> 类型的 ROCBLAS 数据类型常量
template <>
constexpr rocblas_datatype RocBlasDataTypeFor<c10::complex<double>>() {
  return rocblas_datatype_f64_c;
}

// RocBlasComputeTypeFor 模板，返回 T 类型的 ROCBLAS 计算类型常量
template <typename T>
constexpr rocblas_datatype RocBlasComputeTypeFor();

// RocBlasComputeTypeFor 模板的特化，返回 float 类型的 ROCBLAS 计算类型常量
template <>
constexpr rocblas_datatype RocBlasComputeTypeFor<float>() {
  return rocblas_datatype_f32_r;
}

// RocBlasComputeTypeFor 模板的特化，返回 double 类型的 ROCBLAS 计算类型常量
template <>
constexpr rocblas_datatype RocBlasComputeTypeFor<double>() {
  return rocblas_datatype_f64_r;
}

// RocBlasComputeTypeFor 模板的特化，返回 Half 类型的 ROCBLAS 计算类型常量
template <>
constexpr rocblas_datatype RocBlasComputeTypeFor<Half>() {
  // 注意，这里返回的是给定数据类型的计算类型。
  // 截至到 2022 年 12 月，使用 FP16 计算类型来处理 16 位浮点数相比使用 FP32 计算类型速度要慢得多。
  // 因此，即使是 FP16 数据类型，我们仍然使用 FP32 计算类型。这也是 rocblasGemmHelper 函数（参见 fpgeneric.h）中 GEMM 的实现方式。
  return rocblas_datatype_f32_r;
}

// RocBlasComputeTypeFor 模板的特化，返回 BFloat16 类型的 ROCBLAS 计算类型常量
template <>
constexpr rocblas_datatype RocBlasComputeTypeFor<BFloat16>() {
  // 注意，这里返回的是给定数据类型的计算类型。
  // 截至到 2022 年 12 月，使用 FP16 计算类型来处理 16 位浮点数相比使用 FP32 计算类型速度要慢得多。
  // 因此，即使是 BF16 数据类型，我们仍然使用 FP32 计算类型。这也是 rocblasGemmHelper 函数（参见 fpgeneric.h）中 GEMM 的实现方式。
  return rocblas_datatype_f32_r;
}

// RocBlasComputeTypeFor 模板的特化，返回 c10::complex<float> 类型的 ROCBLAS 计算类型常量
template <>
constexpr rocblas_datatype RocBlasComputeTypeFor<c10::complex<float>>() {
  return rocblas_datatype_f32_c;
}

// RocBlasComputeTypeFor 模板的特化，返回 c10::complex<double> 类型的 ROCBLAS 计算类型常量
template <>
constexpr rocblas_datatype RocBlasComputeTypeFor<c10::complex<double>>() {
  return rocblas_datatype_f64_c;
}

// DoCastForHalfOrBfloat16 模板，返回输入值的类型 T
template <typename T>
auto DoCastForHalfOrBfloat16(const T fp) {
  return fp;
}

// DoCastForHalfOrBfloat16 模板的特化，用于 Half 和 BFloat16 类型，直接返回输入值
template <>
// 为半精度（Half）或 bfloat16 类型的数值执行类型转换，并返回转换后的 float 值
inline auto DoCastForHalfOrBfloat16<Half>(const Half fp) {
  // 将 Half 类型转换为 float 类型
  float h = fp;
  return h;
}

// 为 bfloat16 类型的数值执行类型转换，并返回转换后的 float 值
template <>
inline auto DoCastForHalfOrBfloat16<BFloat16>(const BFloat16 fp) {
  // 将 BFloat16 类型转换为 float 类型
  float h = fp;
  return h;
}

// 根据字符参数 op 返回相应的 rocBLAS 操作类型
static rocblas_operation _rocblasOpFromChar(char op) {
  switch (op) {
    case 'n':
    case 'N':
      return rocblas_operation_none;
    case 't':
    case 'T':
      return rocblas_operation_transpose;
    case 'c':
    case 'C':
      return rocblas_operation_conjugate_transpose;
  }
  // 如果输入的字符不符合预期，则抛出错误信息
  AT_ERROR(
      "_rocblasOpFromChar input should be 't', 'n' or 'c' but got `", op, "`");
}

// 表示一个 rocBLAS Gemm 操作的模板类，继承自 Callable<GemmParams<T>>
template <typename T>
class RocblasGemmOp : public Callable<GemmParams<T>> {
  public:
    // 构造函数，初始化使用的解决方案（solution）
    RocblasGemmOp(int solution) : solution_{solution} {}

    // 调用函数，执行 Gemm 操作
    TuningStatus Call(const GemmParams<T>* params) override {
      auto input_output_type = RocBlasDataTypeFor<T>(); // 获取输入输出数据类型
      auto compute_type = RocBlasComputeTypeFor<T>();   // 获取计算数据类型
      auto h_a = DoCastForHalfOrBfloat16(params->alpha);  // 将 alpha 转换为 float
      auto h_b = DoCastForHalfOrBfloat16(params->beta);   // 将 beta 转换为 float
      auto status = rocblas_gemm_ex(
          (rocblas_handle)at::cuda::getCurrentCUDABlasHandle(),  // 获取当前 CUDA BLAS 句柄
          _rocblasOpFromChar(params->transa),  // 转换 params->transa 到 rocBLAS 操作类型
          _rocblasOpFromChar(params->transb),  // 转换 params->transb 到 rocBLAS 操作类型
          params->m, params->n, params->k,  // Gemm 操作的维度参数
          &h_a,  // alpha 值的地址
          params->a, input_output_type, params->lda,  // 输入矩阵 A
          params->b, input_output_type, params->ldb,  // 输入矩阵 B
          &h_b,  // beta 值的地址
          params->c, input_output_type, params->ldc,  // 输出矩阵 C
          params->c, input_output_type, params->ldc,  // 不使用额外参数矩阵 D
          compute_type,  // 计算类型
          rocblas_gemm_algo_solution_index,  // 算法和解决方案的索引
          solution_,  // 使用的解决方案
          rocblas_gemm_flags_none);  // Gemm 操作的标志
      // 如果 rocBLAS 操作不成功，则返回失败状态
      if (status != rocblas_status_success) {
        return FAIL;
      }
      return OK;  // 返回成功状态
    }

  private:
    int solution_;  // Gemm 操作使用的解决方案
};
// 获取当前 CUDA 的 cuBLAS 句柄，并将其转换为 rocBLAS 句柄
auto GetRocBlasGemmTypeStringAndOps() {
  rocblas_handle handle = (rocblas_handle)at::cuda::getCurrentCUDABlasHandle();
  // 定义变量以存储解的数量
  int solution_size;
  // 获取模板类型 T 对应的 rocBLAS 数据类型
  auto input_output_type = RocBlasDataTypeFor<T>();
  // 获取模板类型 T 对应的 rocBLAS 计算类型
  auto compute_type = RocBlasComputeTypeFor<T>();
  // 获取特定类型和计算类型的可用解的数量
  TORCH_ROCBLAS_CHECK(rocblas_gemm_ex_get_solutions_by_type(handle,
                                                            input_output_type,
                                                            input_output_type,
                                                            compute_type,
                                                            rocblas_gemm_flags_none,
                                                            nullptr,
                                                            &solution_size));
  // 创建一个大小为 solution_size 的整数向量，用于存储可用解的索引
  std::vector<int> solutions(solution_size);
  // 获取特定类型和计算类型的可用解的索引列表
  TORCH_ROCBLAS_CHECK(rocblas_gemm_ex_get_solutions_by_type(handle,
                                                            input_output_type,
                                                            input_output_type,
                                                            compute_type,
                                                            rocblas_gemm_flags_none,
                                                            solutions.data(),
                                                            &solution_size));
  // 对解的索引列表进行升序排序，以确保在不同运行中解向量的确定性
  std::sort(solutions.begin(), solutions.end());

  // 创建一个返回结果的向量，每个元素是一个名称和对应的 gemm 操作的可调用对象
  std::vector<std::pair<std::string, std::unique_ptr<Callable<GemmParams<T>>>>> ret;
  // 遍历所有解的索引，为每个解创建一个 RocblasGemmOp<T> 实例，并加入返回结果向量
  for (size_t i = 0; i < solutions.size(); ++i) {
    auto callable = std::make_unique<RocblasGemmOp<T>>(solutions[i]);
    ret.emplace_back(std::make_pair(c10::str("Gemm_Rocblas_", solutions[i]), std::move(callable)));
  }
  // 返回包含所有解的名称和对应 gemm 操作的向量
  return ret;
}

template <typename T>
class RocblasGemmStridedBatchedOp : public Callable<GemmStridedBatchedParams<T>> {
  public:
    // 构造函数，接受一个解的索引作为参数
    RocblasGemmStridedBatchedOp(int solution) : solution_{solution} {}
    // 调用接口函数实现 GEMM 运算
    TuningStatus Call(const GemmStridedBatchedParams<T>* params) override {
      // 确定输入输出数据类型
      auto input_output_type = RocBlasDataTypeFor<T>();
      // 确定计算数据类型
      auto compute_type = RocBlasComputeTypeFor<T>();
      // 转换 alpha 参数为半精度或 BF16 类型
      auto h_a = DoCastForHalfOrBfloat16(params->alpha);
      // 转换 beta 参数为半精度或 BF16 类型
      auto h_b = DoCastForHalfOrBfloat16(params->beta);
      // 调用 RocBLAS 的 GEMM 批处理函数
      auto status = rocblas_gemm_strided_batched_ex(
          // 获取当前 CUDA BLAS 句柄
          (rocblas_handle)at::cuda::getCurrentCUDABlasHandle(),
          // 根据参数 transa 确定 A 矩阵的转置状态
          _rocblasOpFromChar(params->transa),
          // 根据参数 transb 确定 B 矩阵的转置状态
          _rocblasOpFromChar(params->transb),
          // 矩阵尺寸参数
          params->m, params->n, params->k,
          // alpha 参数
          &h_a,
          // A 矩阵及相关参数
          params->a, input_output_type, params->lda, params->stride_a,
          // B 矩阵及相关参数
          params->b, input_output_type, params->ldb, params->stride_b,
          // beta 参数
          &h_b,
          // C 矩阵及相关参数
          params->c, input_output_type, params->ldc, params->stride_c,
          // C 矩阵及相关参数 (输出位置)
          params->c, input_output_type, params->ldc, params->stride_c,
          // 批处理数量
          params->batch,
          // 计算数据类型
          compute_type,
          // 使用的 GEMM 算法及解决方案索引
          rocblas_gemm_algo_solution_index,
          // 选择的解决方案
          solution_,
          // GEMM 计算标志
          rocblas_gemm_flags_none);
      // 检查 RocBLAS 调用状态，若不成功则返回失败状态
      if (status != rocblas_status_success) {
        return FAIL;
      }
      // 若成功则返回成功状态
      return OK;
    }

  private:
    // 用于记录选择的解决方案的整数值
    int solution_;
// 结束 at::cuda::tunable 命名空间的定义

template <typename T>
// 返回 ROCBLAS GEMM Strided Batched 操作的类型字符串和操作对象
auto GetRocBlasGemmStridedBatchedTypeStringAndOps() {
  // 获取当前 CUDA BLAS 句柄，并将其转换为 rocblas_handle 类型
  rocblas_handle handle = (rocblas_handle)at::cuda::getCurrentCUDABlasHandle();
  int solution_size;
  // 获取输入输出类型和计算类型对应的 rocBLAS 数据类型
  auto input_output_type = RocBlasDataTypeFor<T>();
  auto compute_type = RocBlasComputeTypeFor<T>();
  // 获取可用解决方案的数量
  TORCH_ROCBLAS_CHECK(rocblas_gemm_ex_get_solutions_by_type(handle,
                                                            input_output_type,
                                                            input_output_type,
                                                            compute_type,
                                                            rocblas_gemm_flags_none,
                                                            nullptr,
                                                            &solution_size));
  // 创建一个整数向量以存储可用解决方案的索引
  std::vector<int> solutions(solution_size);
  // 获取可用解决方案的列表
  TORCH_ROCBLAS_CHECK(rocblas_gemm_ex_get_solutions_by_type(handle,
                                                            input_output_type,
                                                            input_output_type,
                                                            compute_type,
                                                            rocblas_gemm_flags_none,
                                                            solutions.data(),
                                                            &solution_size));
  // 将解决方案按升序排序，以确保跨多次运行时的一致性
  std::sort(solutions.begin(), solutions.end());

  // 创建一个返回值向量，包含每个解决方案对应的字符串和操作对象
  std::vector<std::pair<std::string, std::unique_ptr<Callable<GemmStridedBatchedParams<T>>>>> ret;
  for (size_t i = 0; i < solutions.size(); ++i) {
    // 使用当前解决方案创建 RocBLAS GEMM Strided Batched 操作对象
    auto callable = std::make_unique<RocblasGemmStridedBatchedOp<T>>(solutions[i]);
    // 将解决方案字符串和操作对象添加到返回值向量中
    ret.emplace_back(std::make_pair(c10::str("Gemm_Rocblas_", solutions[i]), std::move(callable)));
  }
  // 返回包含解决方案字符串和操作对象的向量
  return ret;
}

// 结束 GetRocBlasGemmStridedBatchedTypeStringAndOps 模板函数定义

}  // 结束 at::cuda::tunable 命名空间的作用域
```