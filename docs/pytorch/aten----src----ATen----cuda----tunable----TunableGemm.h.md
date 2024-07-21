# `.\pytorch\aten\src\ATen\cuda\tunable\TunableGemm.h`

```
// Original TunableOp is from onnxruntime.
// https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/framework/tunable.h
// https://github.com/microsoft/onnxruntime/tree/main/onnxruntime/core/providers/rocm/tunable
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
// Adapting TunableOp into PyTorch
// Copyright (c) Advanced Micro Devices, Inc.
//
#pragma once

#include <ATen/cuda/tunable/GemmCommon.h> // 包含通用的 GEMM 函数定义
#ifdef USE_ROCM
#include <ATen/cuda/tunable/GemmHipblaslt.h> // 如果使用 ROCm，包含 ROCm 版本的 GEMM 函数定义
#include <ATen/cuda/tunable/GemmRocblas.h> // 如果使用 ROCm，包含 ROCm 版本的 ROCBLAS 函数定义
#endif
#include <ATen/cuda/tunable/StreamTimer.h> // 包含流计时器定义
#include <ATen/cuda/tunable/TunableOp.h> // 包含可调优操作的定义
#include <c10/cuda/CUDACachingAllocator.h> // 包含 CUDA 缓存分配器定义
#include <c10/util/Float8_e4m3fn.h> // 包含浮点数操作相关的工具函数
#include <c10/util/Float8_e4m3fnuz.h> // 包含浮点数操作相关的工具函数
#include <c10/util/Float8_e5m2.h> // 包含浮点数操作相关的工具函数
#include <c10/util/Float8_e5m2fnuz.h> // 包含浮点数操作相关的工具函数
#include <c10/util/StringUtil.h> // 包含字符串处理相关的工具函数

#ifdef USE_ROCM
#include <rocm-core/rocm_version.h> // 如果使用 ROCm，包含 ROCm 版本的定义
#endif

#define STRINGIFY(s) #s // 定义宏，将参数转换为字符串
#define XSTRINGIFY(s) STRINGIFY(s) // 定义宏，将参数转换为字符串

namespace at::cuda::tunable {

template <typename T>
class DefaultGemmOp : public Callable<GemmParams<T>> {
  public:
    TuningStatus Call(const GemmParams<T>* params) override {
      at::cuda::blas::gemm_internal<T>(
          params->transa, params->transb,
          params->m, params->n, params->k,
          params->alpha,
          params->a, params->lda,
          params->b, params->ldb,
          params->beta,
          params->c, params->ldc);
      return OK; // 调用默认的 GEMM 操作并返回调优状态 OK
    }
};

template <typename T>
class DefaultGemmStridedBatchedOp : public Callable<GemmStridedBatchedParams<T>> {
  public:
    TuningStatus Call(const GemmStridedBatchedParams<T>* params) override {
      at::cuda::blas::bgemm_internal<T>(
          params->transa, params->transb,
          params->m, params->n, params->k,
          params->alpha,
          params->a, params->lda, params->stride_a,
          params->b, params->ldb, params->stride_b,
          params->beta,
          params->c, params->ldc, params->stride_c,
          params->batch);
      return OK; // 调用默认的批量 GEMM 操作并返回调优状态 OK
    }
};

template <typename T>
class DefaultScaledGemmOp : public Callable<ScaledGemmParams<T>> {
  public:
    TuningStatus Call(const ScaledGemmParams<T>* params) override {
      at::cuda::blas::scaled_gemm(
          params->transa,
          params->transb,
          params->m,
          params->n,
          params->k,
          params->a,
          params->a_scale_ptr,
          params->lda,
          params->a_dtype,
          params->b,
          params->b_scale_ptr,
          params->ldb,
          params->b_dtype,
          params->bias_ptr,
          params->bias_dtype,
          params->c,
          params->c_scale_ptr,
          params->ldc,
          params->c_dtype,
          params->amax_ptr,
          params->use_fast_accum);
      return OK; // 调用默认的缩放 GEMM 操作并返回调优状态 OK
    }
};

template <typename T>
inline bool IsZero(T v) {
  return v == 0.0f; // 模板函数，检查值是否为零
}

template <>
inline bool IsZero(BFloat16 v) {
  return v.x == 0; // 特化模板函数，针对 BFloat16 类型的值检查是否为零
}

template <>
// 检查半精度浮点数是否为零
inline bool IsZero(Half v) {
  return float(v) == 0.0f;
}

// 检查双精度复数是否为零
template <>
inline bool IsZero(c10::complex<double> v) {
  return v == 0.0;
}

// 检查单精度复数是否为零
template <>
inline bool IsZero(c10::complex<float> v) {
  return v == 0.0f;
}

// 返回模板类型的名称，默认为"unknown"
template <typename T>
inline std::string TypeName(T v) {
  return "unknown";
}

// 返回单精度浮点数类型名称为"float"
template <>
inline std::string TypeName(float v) {
  return "float";
}

// 返回双精度浮点数类型名称为"double"
template <>
inline std::string TypeName(double v) {
  return "double";
}

// 返回BFloat16类型名称为"BFloat16"
template <>
inline std::string TypeName(BFloat16 v) {
  return "BFloat16";
}

// 返回Half类型名称为"Half"
template <>
inline std::string TypeName(Half v) {
  return "Half";
}

// 返回Float8_e4m3fn类型名称为"Float8_e4m3fn"
template <>
inline std::string TypeName(Float8_e4m3fn v) {
  return "Float8_e4m3fn";
}

// 返回Float8_e5m2类型名称为"Float8_e5m2"
template <>
inline std::string TypeName(Float8_e5m2 v) {
  return "Float8_e5m2";
}

// 返回Float8_e4m3fnuz类型名称为"Float8_e4m3fnuz"
template <>
inline std::string TypeName(Float8_e4m3fnuz v) {
  return "Float8_e4m3fnuz";
}

// 返回Float8_e5m2fnuz类型名称为"Float8_e5m2fnuz"
template <>
inline std::string TypeName(Float8_e5m2fnuz v) {
  return "Float8_e5m2fnuz";
}

// 返回双精度复数类型名称为"c10::complex<double>"
template <>
inline std::string TypeName(c10::complex<double> v) {
  return "c10::complex<double>";
}

// 返回单精度复数类型名称为"c10::complex<float>"
template <>
inline std::string TypeName(c10::complex<float> v) {
  return "c10::complex<float>";
}

// 当使用ROCM时，注册ROCBLAS_VERSION的验证器
static void AddRocblasValidator() {
  auto validators = getTuningContext()->GetTuningResultsValidator().GetAllValidators();
  // 如果ROCBLAS_VERSION验证器不存在，则注册
  if (validators.find("ROCBLAS_VERSION") == validators.end()) {
    // 构建ROCBLAS版本字符串
    std::string rocblas_version = c10::str(
        XSTRINGIFY(ROCBLAS_VERSION_MAJOR), ".",
        XSTRINGIFY(ROCBLAS_VERSION_MINOR), ".",
        XSTRINGIFY(ROCBLAS_VERSION_PATCH), "-",
        XSTRINGIFY(ROCBLAS_VERSION_TWEAK));
    // 注册ROCBLAS_VERSION验证器
    getTuningContext()->GetTuningResultsValidator().RegisterValidator(
        "ROCBLAS_VERSION",
        [rocblas_version]() { return rocblas_version; },
        [rocblas_version](auto&& k) { return rocblas_version == k ? OK : FAIL; });
  }
}

// 当使用ROCM时，注册HIPBLASLT_VERSION的验证器
static void AddHipblasltValidator() {
  auto validators = getTuningContext()->GetTuningResultsValidator().GetAllValidators();
  // 如果HIPBLASLT_VERSION验证器不存在，则注册
  if (validators.find("HIPBLASLT_VERSION") == validators.end()) {
    // 构建HIPBLASLT版本字符串
    std::string hipblaslt_version = c10::str(
        XSTRINGIFY(HIPBLASLT_VERSION_MAJOR), ".",
        XSTRINGIFY(HIPBLASLT_VERSION_MINOR), ".",
        XSTRINGIFY(HIPBLASLT_VERSION_PATCH), "-",
        XSTRINGIFY(HIPBLASLT_VERSION_TWEAK));
    // 注册HIPBLASLT_VERSION验证器
    getTuningContext()->GetTuningResultsValidator().RegisterValidator(
        "HIPBLASLT_VERSION",
        [hipblaslt_version]() { return hipblaslt_version; },
        [hipblaslt_version](auto&& k) { return hipblaslt_version == k ? OK : FAIL; });
  }
}

// 当使用ROCM时，注册ROCM_VERSION的验证器
static void AddRocmValidator() {
  auto validators = getTuningContext()->GetTuningResultsValidator().GetAllValidators();
  // 如果ROCM_VERSION验证器不存在，则注册
  if (validators.find("ROCM_VERSION") == validators.end()) {
    // 获取ROCM_BUILD_INFO中的ROCM版本信息
    std::string rocm_version = ROCM_BUILD_INFO;
    // 注册ROCM_VERSION验证器
    getTuningContext()->GetTuningResultsValidator().RegisterValidator(
        "ROCM_VERSION",
        [rocm_version]() { return rocm_version; },
        [rocm_version](auto&& k) { return rocm_version == k ? OK : FAIL; });
  }
}
    # 获取调优上下文中的调优结果验证器，并注册一个名为 "ROCM_VERSION" 的验证器
    getTuningContext()->GetTuningResultsValidator().RegisterValidator(
        "ROCM_VERSION",  # 注册验证器的名称为 "ROCM_VERSION"
        [rocm_version]() { return rocm_version; },  # 匿名函数，返回当前的 rocm_version 值
        [rocm_version](auto&& k) { return rocm_version == k ? OK : FAIL; });  # 匿名函数，接受参数 k，比较 rocm_version 和 k 的值，相同返回 OK，不同返回 FAIL
  }

  # 如果 "GCN_ARCH_NAME" 验证器在 validators 中找不到
  if (validators.find("GCN_ARCH_NAME") == validators.end()) {
    # 获取当前 CUDA 设备的属性中的 GCN 架构名称
    std::string gcn_arch_name = at::cuda::getCurrentDeviceProperties()->gcnArchName;
    # 获取调优上下文中的调优结果验证器，并注册一个名为 "GCN_ARCH_NAME" 的验证器
    getTuningContext()->GetTuningResultsValidator().RegisterValidator(
        "GCN_ARCH_NAME",  # 注册验证器的名称为 "GCN_ARCH_NAME"
        [gcn_arch_name]() { return gcn_arch_name; },  # 匿名函数，返回当前的 gcn_arch_name 值
        [gcn_arch_name](auto&& k) { return gcn_arch_name == k ? OK : FAIL; });  # 匿名函数，接受参数 k，比较 gcn_arch_name 和 k 的值，相同返回 OK，不同返回 FAIL
  }
  // GemmStridedBatchedTunableOp 类的构造函数，初始化对象时执行
  GemmStridedBatchedTunableOp() {
    // 注册默认的操作 'Default'，使用 DefaultGemmStridedBatchedOp<T> 实例
    this->RegisterOp(std::string("Default"), std::make_unique<DefaultGemmStridedBatchedOp<T>>());

#ifdef USE_ROCM
    bool rocm_validators = false;

    // 检查环境变量 PYTORCH_TUNABLEOP_ROCBLAS_ENABLED 是否为 "1"，若是则启用 rocBLAS 支持
    static const char *env_rocblas = std::getenv("PYTORCH_TUNABLEOP_ROCBLAS_ENABLED");
    if (env_rocblas == nullptr || strcmp(env_rocblas, "1") == 0) {
      rocm_validators = true;
      // 遍历获取 ROCm 版本的 GemmStridedBatched 操作类型字符串和操作对象，并注册
      for (auto&& [name, op] : GetRocBlasGemmStridedBatchedTypeStringAndOps<T>()) {
        this->RegisterOp(std::move(name), std::move(op));
      }
      // 添加 rocBLAS 验证器
      AddRocblasValidator();
    }

    // 检查环境变量 PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED 是否为 "1"，若是则启用 hipBLASLT 支持
    static const char *env_hipblaslt = std::getenv("PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED");
    if (env_hipblaslt == nullptr || strcmp(env_hipblaslt, "1") == 0) {
      rocm_validators = true;
      // 禁止对 c10::complex 类型进行 hipBLASLT 的调优
      if constexpr (
          !std::is_same_v<T, c10::complex<float>> &&
          !std::is_same_v<T, c10::complex<double>>) {
        // 遍历获取 HIP 版本的 GemmStridedBatched 操作类型字符串和操作对象，并注册
        for (auto&& [name, op] : GetHipBlasLtGemmStridedBatchedTypeStringAndOps<T, ALayout, BLayout>()) {
          this->RegisterOp(std::move(name), std::move(op));
        }
      }
      // 添加 hipBLASLT 验证器
      AddHipblasltValidator();
    }

    // 若任一 ROCm 相关的验证器被启用，则添加 ROCm 验证器
    if (rocm_validators) {
      AddRocmValidator();
    }
#endif
  }
    // 构造字符串，格式为 "GemmStridedBatchedTunableOp_" 加上模板类型 T 的名称，以及 ALayout 和 BLayout 的字符串表示形式
    return c10::str("GemmStridedBatchedTunableOp_", TypeName<T>(T{}), "_", BlasOpToString(ALayout), BlasOpToString(BLayout));
}
};

template <typename AT, typename BT, typename CT, BlasOp ALayout, BlasOp BLayout>
class ScaledGemmTunableOp : public TunableOp<ScaledGemmParams<CT>, StreamTimer> {
 public:
  ScaledGemmTunableOp() {
    // 注册默认的操作器，使用 DefaultScaledGemmOp<CT>
    this->RegisterOp(std::string("Default"), std::make_unique<DefaultScaledGemmOp<CT>>());

    // 获取调优上下文的验证器，并获取所有验证器
    auto validators = getTuningContext()->GetTuningResultsValidator().GetAllValidators();

#if defined(USE_ROCM)
    // 如果使用 ROCm 平台，则注册 HIP BlasLt 实现的操作器
    for (auto&& [name, op] : GetHipBlasLtScaledGemmTypeStringAndOps<AT, BT, CT, ALayout, BLayout>()) {
      this->RegisterOp(std::move(name), std::move(op));
    }
    // 添加 HIP BlasLt 的验证器
    AddHipblasltValidator();
    // 添加 ROCm 的验证器
    AddRocmValidator();
#endif
  }

  // 返回此操作的签名，格式为 "ScaledGemmTunableOp_<AT type>_<BT type>_<CT type>_<ALayout>_<BLayout>"
  std::string Signature() override {
    return c10::str("ScaledGemmTunableOp",
            "_", TypeName<AT>(AT{}),
            "_", TypeName<BT>(BT{}),
            "_", TypeName<CT>(CT{}),
            "_", BlasOpToString(ALayout), BlasOpToString(BLayout));
  }
};

#undef XSTRINGIFY
#undef STRINGIFY

} // namespace at::cuda::tunable
```