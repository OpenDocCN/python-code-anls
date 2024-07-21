# `.\pytorch\c10\core\QScheme.h`

```py
#pragma once

#include <c10/util/Exception.h> // 包含异常处理的头文件
#include <cstdint> // 包含整数类型定义的头文件
#include <string> // 包含字符串处理的头文件

namespace c10 {

/**
 * QScheme is an enum that specifies the type of quantization. This has a one
 * to one correspondence with Quantizer
 * Please refer to ATen/quantized/Quantizer.h to see the Quantizers classes.
 * Keep this file in sync with torch/nn/_qscheme.py
 */
enum class QScheme : uint8_t {
  PER_TENSOR_AFFINE = 0, // 每张量仿射量化的枚举值，对应值为0
  PER_CHANNEL_AFFINE = 1, // 每通道仿射量化的枚举值，对应值为1
  PER_TENSOR_SYMMETRIC = 2, // 每张量对称量化的枚举值，对应值为2
  PER_CHANNEL_SYMMETRIC = 3, // 每通道对称量化的枚举值，对应值为3
  PER_CHANNEL_AFFINE_FLOAT_QPARAMS = 4, // 每通道仿射浮点参数量化的枚举值，对应值为4
  COMPILE_TIME_NUM_QSCHEMES = 5, // 编译时的量化方案数量，对应值为5
};

constexpr auto kPerTensorAffine = QScheme::PER_TENSOR_AFFINE; // 常量kPerTensorAffine对应于每张量仿射量化的枚举值
constexpr auto kPerChannelAffine = QScheme::PER_CHANNEL_AFFINE; // 常量kPerChannelAffine对应于每通道仿射量化的枚举值
constexpr auto kPerTensorSymmetric = QScheme::PER_TENSOR_SYMMETRIC; // 常量kPerTensorSymmetric对应于每张量对称量化的枚举值
constexpr auto kPerChannelSymmetric = QScheme::PER_CHANNEL_SYMMETRIC; // 常量kPerChannelSymmetric对应于每通道对称量化的枚举值
constexpr auto kPerChannelAffineFloatQParams =
    QScheme::PER_CHANNEL_AFFINE_FLOAT_QPARAMS; // 常量kPerChannelAffineFloatQParams对应于每通道仿射浮点参数量化的枚举值
constexpr int COMPILE_TIME_NUM_QSCHEMES =
    static_cast<int>(QScheme::COMPILE_TIME_NUM_QSCHEMES); // 编译时的量化方案数量，转换为整数类型的常量

/**
 * Convert QScheme enum value to string representation.
 * Converts each QScheme enum value to its corresponding string representation.
 * Throws an exception for unrecognized QScheme values.
 */
inline std::string toString(QScheme qscheme) {
  switch (qscheme) {
    case kPerTensorAffine:
      return "per_tensor_affine"; // 返回每张量仿射量化的字符串表示
    case kPerChannelAffine:
      return "per_channel_affine"; // 返回每通道仿射量化的字符串表示
    case kPerTensorSymmetric:
      return "per_tensor_symmetric"; // 返回每张量对称量化的字符串表示
    case kPerChannelSymmetric:
      return "per_channel_symmetric"; // 返回每通道对称量化的字符串表示
    case kPerChannelAffineFloatQParams:
      return "per_channel_affine_float_qparams"; // 返回每通道仿射浮点参数量化的字符串表示
    default:
      TORCH_CHECK(false, "Unrecognized qscheme: ", static_cast<int>(qscheme)); // 对于未识别的量化方案枚举值，抛出异常
  }
}

} // namespace c10
```