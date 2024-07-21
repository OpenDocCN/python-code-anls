# `.\pytorch\aten\src\ATen\cuda\CUDADataType.h`

```
#pragma once
// 预处理指令，确保头文件只被包含一次

#include <c10/core/ScalarType.h>
// 包含 c10 库中的 ScalarType 头文件

#include <cuda.h>
// 包含 CUDA 运行时 API 头文件

#include <library_types.h>
// 包含 library_types.h 头文件

namespace at::cuda {
// 进入 at::cuda 命名空间

template <typename scalar_t>
cudaDataType getCudaDataType() {
  // 模板函数定义，根据模板类型 scalar_t 返回对应的 cudaDataType
  static_assert(false && sizeof(scalar_t), "Cannot convert type to cudaDataType.");
  // 静态断言，始终失败，显示无法将类型转换为 cudaDataType
  return {};
  // 返回空值
}

template<> inline cudaDataType getCudaDataType<at::Half>() {
  // 特化模板函数，当 scalar_t 是 at::Half 类型时
  return CUDA_R_16F;
  // 返回 CUDA 中的 16-bit 浮点数类型
}
template<> inline cudaDataType getCudaDataType<float>() {
  // 特化模板函数，当 scalar_t 是 float 类型时
  return CUDA_R_32F;
  // 返回 CUDA 中的 32-bit 浮点数类型
}
template<> inline cudaDataType getCudaDataType<double>() {
  // 特化模板函数，当 scalar_t 是 double 类型时
  return CUDA_R_64F;
  // 返回 CUDA 中的 64-bit 浮点数类型
}
template<> inline cudaDataType getCudaDataType<c10::complex<c10::Half>>() {
  // 特化模板函数，当 scalar_t 是 c10::complex<c10::Half> 类型时
  return CUDA_C_16F;
  // 返回 CUDA 中的 16-bit 复数类型
}
template<> inline cudaDataType getCudaDataType<c10::complex<float>>() {
  // 特化模板函数，当 scalar_t 是 c10::complex<float> 类型时
  return CUDA_C_32F;
  // 返回 CUDA 中的 32-bit 复数类型
}
template<> inline cudaDataType getCudaDataType<c10::complex<double>>() {
  // 特化模板函数，当 scalar_t 是 c10::complex<double> 类型时
  return CUDA_C_64F;
  // 返回 CUDA 中的 64-bit 复数类型
}

template<> inline cudaDataType getCudaDataType<uint8_t>() {
  // 特化模板函数，当 scalar_t 是 uint8_t 类型时
  return CUDA_R_8U;
  // 返回 CUDA 中的 8-bit 无符号整数类型
}
template<> inline cudaDataType getCudaDataType<int8_t>() {
  // 特化模板函数，当 scalar_t 是 int8_t 类型时
  return CUDA_R_8I;
  // 返回 CUDA 中的 8-bit 有符号整数类型
}
template<> inline cudaDataType getCudaDataType<int>() {
  // 特化模板函数，当 scalar_t 是 int 类型时
  return CUDA_R_32I;
  // 返回 CUDA 中的 32-bit 整数类型
}

template<> inline cudaDataType getCudaDataType<int16_t>() {
  // 特化模板函数，当 scalar_t 是 int16_t 类型时
  return CUDA_R_16I;
  // 返回 CUDA 中的 16-bit 整数类型
}
template<> inline cudaDataType getCudaDataType<int64_t>() {
  // 特化模板函数，当 scalar_t 是 int64_t 类型时
  return CUDA_R_64I;
  // 返回 CUDA 中的 64-bit 整数类型
}
template<> inline cudaDataType getCudaDataType<at::BFloat16>() {
  // 特化模板函数，当 scalar_t 是 at::BFloat16 类型时
  return CUDA_R_16BF;
  // 返回 CUDA 中的 16-bit BFloat16 类型
}

inline cudaDataType ScalarTypeToCudaDataType(const c10::ScalarType& scalar_type) {
  // 定义一个函数，将 c10::ScalarType 转换为对应的 cudaDataType
  switch (scalar_type) {
    case c10::ScalarType::Byte:
      return CUDA_R_8U;
      // 如果 scalar_type 是 Byte 类型，返回 CUDA 中的 8-bit 无符号整数类型
    case c10::ScalarType::Char:
      return CUDA_R_8I;
      // 如果 scalar_type 是 Char 类型，返回 CUDA 中的 8-bit 有符号整数类型
    case c10::ScalarType::Int:
      return CUDA_R_32I;
      // 如果 scalar_type 是 Int 类型，返回 CUDA 中的 32-bit 整数类型
    case c10::ScalarType::Half:
      return CUDA_R_16F;
      // 如果 scalar_type 是 Half 类型，返回 CUDA 中的 16-bit 浮点数类型
    case c10::ScalarType::Float:
      return CUDA_R_32F;
      // 如果 scalar_type 是 Float 类型，返回 CUDA 中的 32-bit 浮点数类型
    case c10::ScalarType::Double:
      return CUDA_R_64F;
      // 如果 scalar_type 是 Double 类型，返回 CUDA 中的 64-bit 浮点数类型
    case c10::ScalarType::ComplexHalf:
      return CUDA_C_16F;
      // 如果 scalar_type 是 ComplexHalf 类型，返回 CUDA 中的 16-bit 复数类型
    case c10::ScalarType::ComplexFloat:
      return CUDA_C_32F;
      // 如果 scalar_type 是 ComplexFloat 类型，返回 CUDA 中的 32-bit 复数类型
    case c10::ScalarType::ComplexDouble:
      return CUDA_C_64F;
      // 如果 scalar_type 是 ComplexDouble 类型，返回 CUDA 中的 64-bit 复数类型
    case c10::ScalarType::Short:
      return CUDA_R_16I;
      // 如果 scalar_type 是 Short 类型，返回 CUDA 中的 16-bit 整数类型
    case c10::ScalarType::Long:
      return CUDA_R_64I;
      // 如果 scalar_type 是 Long 类型，返回 CUDA 中的 64-bit 整数类型
    case c10::ScalarType::BFloat16:
      return CUDA_R_16BF;
      // 如果 scalar_type 是 BFloat16 类型，返回 CUDA 中的 16-bit BFloat16 类型
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11080
    case c10::ScalarType::Float8_e4m3fn:
      return CUDA_R_8F_E4M3;
      // 如果 scalar_type 是 Float8_e4m3fn 类型，返回 CUDA 中的 8-bit 量子浮点数类型
    case c10::ScalarType::Float8_e5m2:
      return CUDA_R_8F_E5M2;
      // 如果 scalar_type 是 Float8_e5m2 类型，返回 CUDA 中的 8-bit 量子浮点数类型
#endif
#if defined(USE_ROCM)
#if defined(HIP_NEW_TYPE_ENUMS)
    case c10::ScalarType::Float8_e4m3fnuz:
      return HIP_R_8F_E4M3_FNUZ;
      // 如果 scalar_type 是 Float8_e4m3fnuz 类型，返回 HIP 中的 8-bit 量子浮点数类型
    case c10::ScalarType::Float8_e5m2fnuz:
      return HIP_R_8F_E5M2_FNUZ;
      // 如果 scalar_type 是 Float8_e5m2fnuz 类型，返回 HIP 中的 8-bit 量子浮点数类型
#else
    case c10::ScalarType::Float8_e4m3fnuz:
      return static_cast<hipDataType>(1000);
      // 如果 scalar_type 是 Float8_e4m3fnuz 类型，返回 HIP 中的特定数据类型
    case c10::ScalarType::Float8_e5m2fnuz:
      return static_cast<hipDataType>(1001);
      // 如果 scalar_type 是 Float8_e5m2fnuz 类型，返回 HIP 中的特定数据类型
#endif
#endif
    default:
      TORCH_INTERNAL_ASSERT(false, "Cannot convert ScalarType ", scalar_type, " to cudaDataType.")
      // 默认情况下，断言失败，显示无法将 scalar_type 转换为 cudaDataType
  }
}

} // namespace at::cuda
// 结束 at::cuda 命名空间
```