# `.\pytorch\aten\src\ATen\native\quantized\cpu\QuantUtils.h`

```
#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/core/List.h>
#include <ATen/TensorOperators.h>
#include <c10/util/irange.h>
#include <algorithm>
#include <cmath>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/quantize_per_tensor_native.h>
#include <ATen/ops/quantize_per_channel_native.h>
#include <ATen/ops/zeros.h>
#endif

namespace quant_utils {
namespace {
  // Function to convert raw 16-bit half precision floating point number
  // to single precision floating point number.
  float RawUint16ToFp16(unsigned short value) {
    const unsigned short sign_bits = value >> 15;
    const unsigned short exponent_bits = value >> 10 & 0x1f;
    const unsigned short significand_bits = value & 0x3ff;

    const float sign = sign_bits ? -1 : 1;
    // Calculate the significand and exponent for conversion
    const float significand =
        1 + significand_bits * 0.0009765625f; // 0.0009765625f = 0x1p-10 = 2^-10;
    const float exponent = exponent_bits - 0xf;

    // Return the converted float value
    return sign * std::ldexp(significand, exponent);
}

// Template function to check if an element exceeds max_val and saturate it if true
template <typename T>
bool CheckAndSaturate(T max_val, T* element) {
  if (*element > max_val) {
    *element = max_val;
    return true;
  }
  if (*element < -max_val) {
    *element = -max_val;
    return true;
  }
  return false;
}
}
using namespace std;

// A structure to hold quantization parameters 'scale' and 'zero_point'.
// The meaning of these values is as the constants in the quantization equation
//
//   real_value = scale * (quantized_value - zero_point)
//
// In other words, 'zero_point' is the quantized value that corresponds
// to the real value 0, and 'scale' is the difference of real values
// corresponding to consecutive quantized values.
struct TensorQuantizationParams {
  double scale;         // Scale factor in the quantization equation
  std::int32_t zero_point;  // Quantized value corresponding to real value 0
  int precision;        // Precision of the quantization
};

// Use fp16_min as the small scale cutoff because we don't want to use scales in
// fp16 subnormal range. This is to be consistent with Glow and FakeLowP
// implementation for NNPI.
constexpr float SMALL_SCALE_THRESHOLD = 6.1e-5f;

// Function to choose quantization parameters based on input range and constraints
// Following implementation should be identical to fbgemm::ChooseQuantizationParams
inline TensorQuantizationParams ChooseQuantizationParams(
    float min,
    float max,
    int32_t qmin,
    int32_t qmax,
    bool preserve_sparsity = false,
    bool force_scale_power_of_two = false,
    bool reduce_range = false) {
  // Check if min is less than or equal to max
  TORCH_CHECK(
      min <= max,
      "In ChooseQuantizationParams, min should be less than or equal to max");

  // Reduce the quantization range if requested
  if (reduce_range) {
    qmin = qmin/2;
    qmax = qmax/2;
  }

  // Adjust min and max if preserving sparsity between positive and negative values
  if (min < 0 && max > 0 && preserve_sparsity) {
    int symmetric_qmin = -((qmax - qmin) / 2 + 1);
    int symmetric_qmax = (qmax - qmin) / 2;
    double max_scale =
        std::max(fabs(min / symmetric_qmin), fabs(max / symmetric_qmax));
    min = max_scale * symmetric_qmin;
  // 计算 max 的值，这里使用了 max_scale 乘以 symmetric_qmax
  max = max_scale * symmetric_qmax;
}

// 扩展 [min, max] 区间以确保其包含 0
// 否则，我们无法满足 0 是可精确表示的要求
min = std::min(min, 0.f);
max = std::max(max, 0.f);

TORCH_CHECK(
    qmin < qmax,
    "In ChooseQuantizationParams, qmin should be less than qmax");

// 使用双精度进行中间计算，但最终使用单精度来反映量化过程中实际使用的数值
double scale = (static_cast<double>(max) - min) / (qmax - qmin);
// 如果 scale 是 0 或太小，其倒数为无穷大，我们将 scale 调整为 0.1
// 避免 scale 的倒数为无穷大，因为 fbgemm 代码中的一些部分预先计算了 scale 的倒数
if (float(scale) == 0.0f || std::isinf(1.0f / float(scale))) {
  scale = 0.1;
}
TORCH_CHECK(scale > 0, "quantization scale should be > 0");

if (force_scale_power_of_two) {
  // 如果要求 scale 是 2 的幂次方，则进行调整
  if (scale < 1) {
    scale = 1.0 / (1 << static_cast<int>(floor(log(1.0 / scale) / log(2))));
  } else {
    scale = 1 << static_cast<int>(ceil(log(scale) / log(2)));
  }
}

// 截断小的 scale 值
if (scale < SMALL_SCALE_THRESHOLD) {
  float org_scale = scale;
  scale = SMALL_SCALE_THRESHOLD;
  // 根据新的 scale 调整 min 和 max 的值
  if (min == 0.0f) {
    max = SMALL_SCALE_THRESHOLD * (qmax - qmin);
  } else if (max == 0.0f) {
    min = -SMALL_SCALE_THRESHOLD * (qmax - qmin);
  } else {
    float amplifier = SMALL_SCALE_THRESHOLD / org_scale;
    min *= amplifier;
    max *= amplifier;
  }
}

// 计算零点值
// 首先进行初始的浮点计算，零点可以从解决仿射方程中确定
// 我们知道两对这样的值：(rmin, qmin) 和 (rmax, qmax)
// 零点的算术误差大约为机器 epsilon * (所有项的绝对值之和)
// 所以我们选择添加较小项的变体
double zero_point_from_min = qmin - min / static_cast<double>(scale);
double zero_point_from_max = qmax - max / static_cast<double>(scale);
double zero_point_from_min_error =
    std::abs(qmin) - std::abs(min / static_cast<double>(scale));
double zero_point_from_max_error =
    std::abs(qmax) - std::abs(max / static_cast<double>(scale));
double initial_zero_point =
    zero_point_from_min_error < zero_point_from_max_error
    ? zero_point_from_min
    : zero_point_from_max;

// 对于对称量化 (preserve_sparsity == true)，我们强制零点为 qmin 和 qmax 的中间值
// 如果 min 或 max 为 0，则直接使用 0 作为零点
if (min < 0 && max > 0 && preserve_sparsity) {
    // 计算初始的零点，将量化的最小值和最大值的中间值转换为双精度浮点数
    initial_zero_point = static_cast<double>(qmin + qmax) / 2;
  }

  // 现在我们需要将零点微调为整数
  // （我们的零点是整数，这是因为需要能够精确表示实际值“0”作为一个量化值，
  // 这在多个地方都是必需的，例如在使用零填充的Im2col操作中）。
  int32_t nudged_zero_point = 0;
  // 如果初始零点小于量化的最小值，则将微调后的零点设置为量化的最小值
  if (initial_zero_point < qmin) {
    nudged_zero_point = qmin;
  } 
  // 如果初始零点大于量化的最大值，则将微调后的零点设置为量化的最大值
  else if (initial_zero_point > qmax) {
    nudged_zero_point = qmax;
  } 
  // 否则，将微调后的零点设置为最接近初始零点的整数值
  else {
    nudged_zero_point = nearbyint(initial_zero_point);
  }

  // 创建一个 TensorQuantizationParams 结构体实例
  TensorQuantizationParams result;
  // 将预先计算好的 scale 存入 result 结构体中
  result.scale = scale;
  // 将微调后的零点存入 result 结构体中
  result.zero_point = nudged_zero_point;
  // 返回存有量化参数的结构体实例
  return result;
// 结束 quant_utils 命名空间

// 此常量用于将 Conv1D 的维度转换为 Conv2D 操作可用的维度
constexpr int64_t kConv1dSqueezeDim = 0;

// 为 Conv1D 创建参数的辅助函数
static C10_UNUSED torch::List<int64_t> MakeArgForConv1d(const torch::List<int64_t>& arg,
                                             int64_t base_value) {
  // 检查参数是否为空
  TORCH_CHECK(!arg.empty(), "Argument must have elements.");
  
  // 创建结果列表，初始值为 arg 的第一个元素和 base_value
  torch::List<int64_t> result({arg.get(0), base_value});

  // 如果参数列表只有一个元素，则将结果列表的第二个元素设为参数列表的第一个元素
  if (arg.size() == 1) {
    result[1] = arg.get(0);
  } else {
    // 否则，将结果列表的第二个元素设为参数列表的第二个元素
    result[1] = arg.get(1);
  }

  // 将结果列表中的 kConv1dSqueezeDim 元素设为 base_value
  result[kConv1dSqueezeDim] = base_value;

  // 返回最终结果列表
  return result;
}

// 处理权重饱和的辅助函数，用于 FP16 量化
inline void HandleWeightsSaturation(int64_t N, float* weight) {
  // FP16 的最大值
  const float kFp16Max = RawUint16ToFp16(0x7BFF);
  // 是否找到超出范围的权重
  bool found_out_of_range = false;

  // 遍历权重数组
  for (const auto i : c10::irange(N)) {
    // 检查并饱和超出范围的权重值
    bool saturate = CheckAndSaturate<float>(kFp16Max, weight + i);
    if (saturate) {
      found_out_of_range = true;
    }
  }

  // 如果找到超出范围的权重，发出警告
  if (found_out_of_range) {
    TORCH_WARN("FOUND weight out of range ");
  }
}

// 量化偏置的实用函数
inline at::Tensor QuantizeBias(
    bool is_per_channel,
    const at::Tensor& bias,
    const at::Tensor& weight_contig,
    double input_scale) {
  at::Tensor qbias;

  // 如果是按通道量化
  if (is_per_channel) {
    // 计算偏置的量化比例尺度
    auto bias_quant_scales =
        weight_contig.q_per_channel_scales() * input_scale;
    // 创建偏置的零点张量
    auto bias_zp = at::zeros(bias_quant_scales.sizes(), c10::kInt);
    // 对偏置进行按通道量化
    qbias = at::native::quantize_per_channel(
        bias, bias_quant_scales, bias_zp, 0, c10::kQInt32);
  } else {
    // 否则，对偏置进行整张量量化
    qbias = at::native::quantize_per_tensor(
        bias, weight_contig.q_scale() * input_scale, 0, c10::kQInt32);
  }

  // 返回量化后的偏置张量
  return qbias;
}
```