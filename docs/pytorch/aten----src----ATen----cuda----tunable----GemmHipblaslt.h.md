# `.\pytorch\aten\src\ATen\cuda\tunable\GemmHipblaslt.h`

```py
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDADataType.h>
#include <ATen/cuda/tunable/TunableOp.h>
#include <ATen/cuda/tunable/GemmCommon.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/util/StringUtil.h>

#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-ext.hpp>

// 定义宏 TORCH_HIPBLASLT_CHECK，用于检查 HIPBLASLT 函数调用是否成功
#define TORCH_HIPBLASLT_CHECK(EXPR)               \
  do {                                            \
    hipblasStatus_t __err = EXPR;                 \
    TORCH_CHECK(__err == HIPBLAS_STATUS_SUCCESS,  \
                "hipblaslt error: ",              \
                hipblasStatusToString(__err),     \
                " when calling `" #EXPR "`");     \
  } while (0)

namespace at::cuda::tunable {

// 模板函数，返回 HIPBLAS 数据类型常量，对于不同类型的 T 返回不同的值
template <typename T>
constexpr hipblasDatatype_t HipBlasDataTypeFor();

template <>
constexpr hipblasDatatype_t HipBlasDataTypeFor<float>() {
  return HIPBLAS_R_32F;
}

template <>
constexpr hipblasDatatype_t HipBlasDataTypeFor<Half>() {
  return HIPBLAS_R_16F;
}

template <>
constexpr hipblasDatatype_t HipBlasDataTypeFor<BFloat16>() {
  return HIPBLAS_R_16B;
}

template <>
constexpr hipblasDatatype_t HipBlasDataTypeFor<double>() {
  return HIPBLAS_R_64F;
}

template <>
constexpr hipblasDatatype_t HipBlasDataTypeFor<c10::Float8_e4m3fnuz>() {
  return HIP_R_8F_E4M3_FNUZ;
}

template <>
constexpr hipblasDatatype_t HipBlasDataTypeFor<c10::Float8_e5m2fnuz>() {
  return HIP_R_8F_E5M2_FNUZ;
}

// 模板函数 GetBatchFromParams，从 GemmParams 获取批处理数
template <typename T>
int GetBatchFromParams(const GemmParams<T>* params) {
  return 1;
}

// 模板函数 GetBatchFromParams，从 GemmStridedBatchedParams 获取批处理数
template <typename T>
int GetBatchFromParams(const GemmStridedBatchedParams<T>* params) {
  return params->batch;
}

// 模板函数 GetBatchFromParams，从 ScaledGemmParams 获取批处理数
template <typename T>
int GetBatchFromParams(const ScaledGemmParams<T>* params) {
  return 1;
}

// 模板函数 GetStrideAFromParams，从 GemmParams 获取矩阵 A 的步幅
template <typename T>
int GetStrideAFromParams(const GemmParams<T>* params) {
  return 1;
}

// 模板函数 GetStrideAFromParams，从 GemmStridedBatchedParams 获取矩阵 A 的步幅
template <typename T>
int GetStrideAFromParams(const GemmStridedBatchedParams<T>* params) {
  return params->stride_a;
}

// 模板函数 GetStrideAFromParams，从 ScaledGemmParams 获取矩阵 A 的步幅
template <typename T>
int GetStrideAFromParams(const ScaledGemmParams<T>* params) {
  return 1;
}

// 模板函数 GetStrideBFromParams，从 GemmParams 获取矩阵 B 的步幅
template <typename T>
int GetStrideBFromParams(const GemmParams<T>* params) {
  return 1;
}

// 模板函数 GetStrideBFromParams，从 GemmStridedBatchedParams 获取矩阵 B 的步幅
template <typename T>
int GetStrideBFromParams(const GemmStridedBatchedParams<T>* params) {
  return params->stride_b;
}

// 模板函数 GetStrideBFromParams，从 ScaledGemmParams 获取矩阵 B 的步幅
template <typename T>
int GetStrideBFromParams(const ScaledGemmParams<T>* params) {
  return 1;
}

// 模板函数 GetStrideCFromParams，从 GemmParams 获取矩阵 C 的步幅
template <typename T>
int GetStrideCFromParams(const GemmParams<T>* params) {
  return 1;
}

// 模板函数 GetStrideCFromParams，从 GemmStridedBatchedParams 获取矩阵 C 的步幅
template <typename T>
int GetStrideCFromParams(const GemmStridedBatchedParams<T>* params) {
  return params->stride_c;
}

// 模板函数 GetStrideCFromParams，从 ScaledGemmParams 获取矩阵 C 的步幅
template <typename T>
int GetStrideCFromParams(const ScaledGemmParams<T>* params) {
  return 1;
}

// 模板函数 GetAlphaFromParams，从 GemmParams 获取 alpha 值
template <typename T>
float GetAlphaFromParams(const GemmParams<T>* params) {
  return params->alpha;
}

// 模板函数 GetAlphaFromParams，从 GemmStridedBatchedParams 获取 alpha 值
template <typename T>
float GetAlphaFromParams(const GemmStridedBatchedParams<T>* params) {
  return params->alpha;
}
// 根据模板参数 T 返回固定值 1.0
template <typename T>
float GetAlphaFromParams(const ScaledGemmParams<T>* params) {
  return 1.0;
}

// 根据模板参数 T 返回参数结构体中的 beta 值
template <typename T>
float GetBetaFromParams(const GemmParams<T>* params) {
  return params->beta;
}

// 根据模板参数 T 返回参数结构体中的 beta 值
template <typename T>
float GetBetaFromParams(const GemmStridedBatchedParams<T>* params) {
  return params->beta;
}

// 根据模板参数 T 返回固定值 0.0
template <typename T>
float GetBetaFromParams(const ScaledGemmParams<T>* params) {
  return 0.0;
}

// 根据模板参数 T 返回 nullptr
template <typename T>
const void* GetAScalePointerFromParams(const GemmParams<T>* params) {
  return nullptr;
}

// 根据模板参数 T 返回 nullptr
template <typename T>
const void* GetAScalePointerFromParams(const GemmStridedBatchedParams<T>* params) {
  return nullptr;
}

// 根据模板参数 T 返回参数结构体中的 a_scale_ptr
template <typename T>
const void* GetAScalePointerFromParams(const ScaledGemmParams<T>* params) {
  return params->a_scale_ptr;
}

// 根据模板参数 T 返回 nullptr
template <typename T>
const void* GetBScalePointerFromParams(const GemmParams<T>* params) {
  return nullptr;
}

// 根据模板参数 T 返回 nullptr
template <typename T>
const void* GetBScalePointerFromParams(const GemmStridedBatchedParams<T>* params) {
  return nullptr;
}

// 根据模板参数 T 返回参数结构体中的 b_scale_ptr
template <typename T>
const void* GetBScalePointerFromParams(const ScaledGemmParams<T>* params) {
  return params->b_scale_ptr;
}

// 根据模板参数 T 返回 nullptr
template <typename T>
const void* GetDScalePointerFromParams(const GemmParams<T>* params) {
  return nullptr;
}

// 根据模板参数 T 返回 nullptr
template <typename T>
const void* GetDScalePointerFromParams(const GemmStridedBatchedParams<T>* params) {
  return nullptr;
}

// 根据模板参数 T 返回参数结构体中的 c_scale_ptr
template <typename T>
const void* GetDScalePointerFromParams(const ScaledGemmParams<T>* params) {
  return params->c_scale_ptr;
}

// 根据模板参数 T 返回 nullptr
template <typename T>
const void* GetBiasPointerFromParams(const GemmParams<T>* params) {
  return nullptr;
}

// 根据模板参数 T 返回 nullptr
template <typename T>
const void* GetBiasPointerFromParams(const GemmStridedBatchedParams<T>* params) {
  return nullptr;
}

// 根据模板参数 T 返回参数结构体中的 bias_ptr
template <typename T>
const void* GetBiasPointerFromParams(const ScaledGemmParams<T>* params) {
  return params->bias_ptr;
}

// 根据模板参数 T 返回 HIP_R_32F 枚举值
template <typename T>
hipDataType GetBiasTypeFromParams(const GemmParams<T>* params) {
  return HIP_R_32F;
}

// 根据模板参数 T 返回 HIP_R_32F 枚举值
template <typename T>
hipDataType GetBiasTypeFromParams(const GemmStridedBatchedParams<T>* params) {
  return HIP_R_32F;
}

// 根据模板参数 T 返回参数结构体中的 bias_dtype 对应的 CUDA 数据类型
template <typename T>
hipDataType GetBiasTypeFromParams(const ScaledGemmParams<T>* params) {
  return at::cuda::ScalarTypeToCudaDataType(params->bias_dtype);
}

// 将字符 op 转换为相应的 hipblasOperation_t 枚举值
static hipblasOperation_t _hipblasOpFromChar(char op) {
  switch (op) {
    case 'n':
    case 'N':
      return HIPBLAS_OP_N;
    case 't':
    case 'T':
      return HIPBLAS_OP_T;
    case 'c':
    case 'C':
      return HIPBLAS_OP_C;
  }
  AT_ERROR(
      "_hipblasOpFromChar input should be 't', 'n' or 'c' but got `", op, "`");
}

// 将 hipblasOperation_t 枚举值 op 转换为对应的字符
static char _charFromhipblasOp(hipblasOperation_t op) {
  switch (op) {
    case HIPBLAS_OP_N:
      return 'N';
    case HIPBLAS_OP_T:
      return 'T';
    case HIPBLAS_OP_C:
      return 'C';
  }
  AT_ERROR(
      "_charFromhipblasOp input should be HIPBLAS_OP_N/T/C but got `", op, "`");
}
# 将布局类型 BlasOp 映射到 HIPBLASLT 中对应的操作类型 hipblasOperation_t
static hipblasOperation_t MapLayoutToHipBlasLt(BlasOp layout) {
    # 如果布局为 BlasOp::N，则返回 HIPBLAS_OP_N，否则返回 HIPBLAS_OP_T
    if (layout == BlasOp::N) {
        return HIPBLAS_OP_N;
    }
    return HIPBLAS_OP_T;
}

# 获取 HIPBLASLT 所需的工作空间大小
static size_t GetHipblasltWorkspaceSize() {
    static const char * env = getenv("HIPBLASLT_WORKSPACE_SIZE");
    # 默认工作空间大小设为 32MB
    size_t workspace_size = 32*1024;  // going with 32MB
    # 如果环境变量 HIPBLASLT_WORKSPACE_SIZE 存在，则尝试解析其值
    if (env) {
        try {
            workspace_size = std::stoi(env);  # 转换环境变量值为整数
        } catch(std::invalid_argument const& e) {
            TORCH_WARN("invalid HIPBLASLT_WORKSPACE_SIZE,", " using default workspace size of ", workspace_size, " KiB.");  # 处理无效参数异常
        } catch(std::out_of_range const& e) {
            TORCH_WARN("HIPBLASLT_WORKSPACE_SIZE out of range,", " using default workspace size of ", workspace_size, " KiB.");  # 处理超出范围异常
    }
    # 返回工作空间大小的字节数
    return workspace_size * 1024;
}

# HipBlasLtDeleter 结构体模板，用于释放 HIPBLASLT 对象
template <typename T, cublasStatus_t (*destructor)(T*)>
struct HipBlasLtDeleter {
    # 重载运算符 ()，用于释放对象
    void operator()(T* x) {
        # 如果对象指针非空，则调用指定的析构函数释放对象
        if (x != nullptr) {
            TORCH_CUDABLAS_CHECK(destructor(x));  # 调用析构函数，检查执行状态
        }
    }
};

# HipBlasLtDescriptor 类模板，用于管理 HIPBLASLT 对象的描述符
template <typename T, hipblasStatus_t (*destructor)(T*)>
class HipBlasLtDescriptor {
 public:
    # 返回描述符对象的指针
    T* descriptor() const {
        return descriptor_.get();
    }
    T* descriptor() {
        return descriptor_.get();
    }

 protected:
    # 使用 std::unique_ptr 管理描述符对象，并指定释放器为 HipBlasLtDeleter
    std::unique_ptr<T, HipBlasLtDeleter<T, destructor>> descriptor_;
};

# HipBlasLtMatmulDescriptor 类，继承自 HipBlasLtDescriptor，用于管理矩阵乘法操作的描述符
class HipBlasLtMatmulDescriptor : public HipBlasLtDescriptor<
                                     hipblasLtMatmulDescOpaque_t,
                                     &hipblasLtMatmulDescDestroy> {
 public:
    # 构造函数，创建矩阵乘法描述符对象
    HipBlasLtMatmulDescriptor(
        hipblasComputeType_t compute_type,
        hipDataType scale_type) {
        hipblasLtMatmulDesc_t raw_descriptor = nullptr;
        # 创建 HIPBLASLT 矩阵乘法描述符对象
        TORCH_HIPBLASLT_CHECK(
            hipblasLtMatmulDescCreate(&raw_descriptor, compute_type, scale_type));
        descriptor_.reset(raw_descriptor);  # 使用 std::unique_ptr 管理描述符对象
    }
    # 设置矩阵乘法描述符的属性值
    template <typename T>
    inline void setAttribute(hipblasLtMatmulDescAttributes_t attr, const T value) {
        # 调用 HIPBLASLT 函数设置描述符的属性
        TORCH_HIPBLASLT_CHECK(::hipblasLtMatmulDescSetAttribute(descriptor(), attr, &value, sizeof(T)));
    }
};

# HipblasltGemmOp 类模板，用于实现 HIPBLASLT 中的矩阵乘法操作
template <typename AT, typename BT, typename CT, BlasOp ALayout, BlasOp BLayout, typename ParamsT>
class HipblasltGemmOp : public Callable<ParamsT> {
  public:
    # 构造函数，初始化使用的算法
    HipblasltGemmOp(hipblasLtMatmulAlgo_t algo) : algo_{algo} {}

  private:
    hipblasLtMatmulAlgo_t algo_;  # 成员变量，存储矩阵乘法算法
};
auto GetHipBlasLtTypeStringAndOps() {
  // 根据布局映射到 HipBLASLT 的操作类型
  hipblasOperation_t transa_outer = MapLayoutToHipBlasLt(ALayout);
  hipblasOperation_t transb_outer = MapLayoutToHipBlasLt(BLayout);
  // 获取 AT、BT、CT 对应的 HipBLAS 数据类型
  auto a_datatype = HipBlasDataTypeFor<AT>();
  auto b_datatype = HipBlasDataTypeFor<BT>();
  auto in_out_datatype = HipBlasDataTypeFor<CT>();
  // 存储启发式算法结果的向量
  std::vector<hipblasLtMatmulHeuristicResult_t> heuristic_result;

  // 创建 HipBLASLT 句柄并检查状态
  hipblasLtHandle_t handle;
  TORCH_HIPBLASLT_CHECK(hipblasLtCreate(&handle));
  // 获取所有的 GEMM 算法并存储在 heuristic_result 中
  TORCH_HIPBLASLT_CHECK(hipblaslt_ext::getAllAlgos(handle,
        hipblaslt_ext::GemmType::HIPBLASLT_GEMM,
        transa_outer,
        transb_outer,
        a_datatype,
        b_datatype,
        in_out_datatype,
        in_out_datatype,
        HIPBLAS_COMPUTE_32F,
        heuristic_result));
  // 销毁 HipBLASLT 句柄
  TORCH_HIPBLASLT_CHECK(hipblasLtDestroy(handle));

  // 根据算法索引排序 heuristic_result，确保返回算法的顺序是确定的
  std::sort(heuristic_result.begin(),
      heuristic_result.end(),
      [](hipblasLtMatmulHeuristicResult_t& a, hipblasLtMatmulHeuristicResult_t& b) {
      return hipblaslt_ext::getIndexFromAlgo(a.algo) < hipblaslt_ext::getIndexFromAlgo(b.algo);
      });

  // 获取返回的算法数量
  int returned_algo_count = heuristic_result.size();
  // 存储返回结果的向量，包含字符串类型和对应的操作
  std::vector<std::pair<std::string, std::unique_ptr<Callable<ParamsT>>>> ret;
  // 遍历每个返回的算法结果
  for (int i = 0; i < returned_algo_count; i++) {
    auto algo = heuristic_result[i].algo;
    int algo_index = hipblaslt_ext::getIndexFromAlgo(algo);
    // 创建 HipBLASLT GEMM 操作对象，并生成类型字符串
    auto callable = std::make_unique<HipblasltGemmOp<AT, BT, CT, ALayout, BLayout, ParamsT>>(algo);
    std::string type_string = c10::str(
        "Gemm_Hipblaslt_", _charFromhipblasOp(transa_outer), _charFromhipblasOp(transb_outer), "_", algo_index);
    // 将类型字符串和操作对象添加到返回结果中
    ret.emplace_back(type_string, std::move(callable));
  }

  // 返回包含类型字符串和操作对象的向量
  return ret;
}

template <typename T, BlasOp ALayout, BlasOp BLayout>
auto GetHipBlasLtGemmTypeStringAndOps() {
  // 返回 HipBLASLT GEMM 类型的字符串和操作对象
  return GetHipBlasLtTypeStringAndOps<T, T, T, ALayout, BLayout, GemmParams<T>>();
}

template <typename T, BlasOp ALayout, BlasOp BLayout>
auto GetHipBlasLtGemmStridedBatchedTypeStringAndOps() {
  // 返回 HipBLASLT GEMM Strided Batched 类型的字符串和操作对象
  return GetHipBlasLtTypeStringAndOps<T, T, T, ALayout, BLayout, GemmStridedBatchedParams<T>>();
}

template <typename AT, typename BT, typename CT, BlasOp ALayout, BlasOp BLayout>
auto GetHipBlasLtScaledGemmTypeStringAndOps() {
  // 返回 HipBLASLT Scaled GEMM 类型的字符串和操作对象
  return GetHipBlasLtTypeStringAndOps<AT, BT, CT, ALayout, BLayout, ScaledGemmParams<CT>>();
}

#undef TORCH_HIPBLASLT_CHECK

}  // namespace at::cuda::tunable


注释结束
```