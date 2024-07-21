# `.\pytorch\aten\src\ATen\native\quantized\AffineQuantizerBase.h`

```py
#pragma once
// 预处理指令，确保头文件只被编译一次

#include <c10/macros/Export.h>
// 导入c10库中的导出宏

#include <c10/core/ScalarType.h>
// 导入c10库中的标量类型定义

namespace at {
namespace native {

// 在给定scale和zero_point的情况下，将float值量化为uint值的模板函数声明
template <typename T>
TORCH_API T quantize_val(double scale, int64_t zero_point, float value);

// TODO: 一旦ARM的数值计算与quantize_val对齐，将此函数与quantize_val合并
template <typename T>
T quantize_val_arm(
    const float scale,
    const int32_t zero_point,
    const float value);

// 使用给定的scale和zero_point将float数组src中的值量化为类型T的数组dst
template <typename T, int precision = 8>
void quantize_vec(
    double scale,
    int64_t zero_point,
    const float* src,
    T* dst,
    size_t count = 8);

// 在给定scale和zero_point的情况下，将类型T的值反量化为float值的函数声明
template <typename T>
TORCH_API float dequantize_val(double scale, int64_t zero_point, T value);

// 使用给定的scale和zero_point将类型T的数组src中的值反量化为float数组dst
template <typename T>
TORCH_API float dequantize_vec(
    double scale,
    int64_t zero_point,
    const T* src,
    float* dst,
    size_t count = 8);

// 将类型SRC_T的值重新量化为类型DST_T的值，给定两组scale和zero_point
template <typename SRC_T, typename DST_T>
TORCH_API DST_T requantize_val(double, int64_t, double, int64_t, SRC_T src);

// 给定一个乘数和一个zero_point，将int32_t类型的计算值重新量化为量化值
// 详见上述make_per_tensor_affine_quantizer函数对int64_t的用法
template <typename DST_T>
TORCH_API DST_T
requantize_from_int(double multiplier, int64_t zero_point, int64_t src);

// 使用float类型的量化参数，将float值量化到int范围内的函数声明
int quantize_val_float_qparams(float scale, float zero_point, float value, int qmin, int qmax);

} // namespace native
} // namespace at
```