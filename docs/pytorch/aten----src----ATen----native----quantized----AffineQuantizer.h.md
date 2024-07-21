# `.\pytorch\aten\src\ATen\native\quantized\AffineQuantizer.h`

```py
#pragma once
// 包含头文件 <ATen/core/Tensor.h>，提供张量（Tensor）相关功能
#include <ATen/core/Tensor.h>
// 包含头文件 <ATen/Dispatch.h>，提供调度相关功能
#include <ATen/Dispatch.h>
// 包含头文件 <ATen/native/DispatchStub.h>，提供分派存根（DispatchStub）功能
#include <ATen/native/DispatchStub.h>
// 包含头文件 <ATen/native/quantized/AffineQuantizerBase.h>，提供量化器基类功能
#include <ATen/native/quantized/AffineQuantizerBase.h>

// 进入 at 命名空间
namespace at {
// 进入 native 命名空间
namespace native {

// 函数声明：在每个张量上进行按张量仿射量化
Tensor& quantize_tensor_per_tensor_affine(
    const Tensor& rtensor,
    Tensor& qtensor,
    double scale,
    int64_t zero_point);

// 函数声明：在每个通道上进行按通道仿射量化
Tensor& quantize_tensor_per_channel_affine(
    const Tensor& rtensor,
    Tensor& qtensor,
    Tensor scales,
    Tensor zero_points,
    int64_t axis);

// 函数声明：使用浮点参数进行每通道仿射量化
Tensor& quantize_tensor_per_channel_float_qparams(
    const Tensor& rtensor,
    Tensor& qtensor,
    Tensor scales,
    Tensor zero_points,
    int64_t axis);

// 函数声明：在每个张量上进行按张量仿射反量化
Tensor& dequantize_tensor_per_tensor_affine(
    const Tensor& qtensor,
    Tensor& rtensor,
    double scale,
    int64_t zero_point);

// 函数声明：在每个通道上进行按通道仿射反量化
Tensor& dequantize_tensor_per_channel_affine(
    const Tensor& qtensor,
    Tensor& rtensor,
    Tensor scales,
    Tensor zero_points,
    int64_t axis);

// 函数声明：使用浮点参数进行每通道仿射反量化
Tensor& dequantize_tensor_per_channel_float_qparams(
    const Tensor& qtensor,
    Tensor& rtensor,
    Tensor scales,
    Tensor zero_points,
    int64_t axis);

// 定义函数指针类型：在每个张量上进行按张量仿射量化函数指针
using quantize_tensor_per_tensor_affine_fn =
    void (*)(const Tensor& rtensor, Tensor& qtensor, double scale, int64_t zero_point);

// 定义函数指针类型：在每个通道上进行按通道仿射量化函数指针
using quantize_tensor_per_channel_affine_fn = void (*)(
    const Tensor& rtensor,
    Tensor& qtensor,
    const Tensor& scales,
    const Tensor& zero_points,
    int64_t axis);

// 定义函数指针类型：使用浮点参数进行每通道仿射量化函数指针
using quantize_tensor_per_channel_float_qparams_fn = void (*)(
    const Tensor& rtensor,
    Tensor& qtensor,
    const Tensor& scales,
    const Tensor& zero_points,
    int64_t axis);

// 定义函数指针类型：在每个张量上进行按张量仿射反量化函数指针
using dequantize_tensor_per_tensor_affine_fn =
    void (*)(const Tensor& qtensor, Tensor& rtensor, double scale, int64_t zero_point);

// 定义函数指针类型：在每个通道上进行按通道仿射反量化函数指针
using dequantize_tensor_per_channel_affine_fn = void (*)(
    const Tensor& qtensor,
    Tensor& rtensor,
    const Tensor& scales,
    const Tensor& zero_points,
    int64_t axis);

// 定义函数指针类型：使用浮点参数进行每通道仿射反量化函数指针
using dequantize_tensor_per_channel_float_qparams_fn = void (*)(
    const Tensor& qtensor,
    Tensor& rtensor,
    const Tensor& scales,
    const Tensor& zero_points,
    int64_t axis);

// 定义函数指针类型：在每个张量上进行按张量仿射量化（子字节级别）函数指针
using quantize_tensor_per_tensor_affine_sub_byte_fn =
    void (*)(const Tensor& rtensor, Tensor& qtensor, float scale, float zero_point);

// 定义函数指针类型：在每个张量上进行按张量仿射反量化（子字节级别）函数指针
using dequantize_tensor_per_tensor_affine_sub_byte_fn =
    void (*)(const Tensor& qtensor, Tensor& rtensor, float scale, float zero_point);

// 声明分派函数：在每个张量上进行按张量仿射量化分派函数
DECLARE_DISPATCH(
    quantize_tensor_per_tensor_affine_fn,
    quantize_tensor_per_tensor_affine_stub);
// 声明分派函数：在每个通道上进行按通道仿射量化分派函数
DECLARE_DISPATCH(
    quantize_tensor_per_channel_affine_fn,
    quantize_tensor_per_channel_affine_stub);
// 声明分派函数：使用浮点参数进行每通道仿射量化分派函数
DECLARE_DISPATCH(
    quantize_tensor_per_channel_float_qparams_fn,
    quantize_tensor_per_channel_float_qparams_stub);

// 声明分派函数：在每个张量上进行按张量仿射反量化分派函数
DECLARE_DISPATCH(
    dequantize_tensor_per_tensor_affine_fn,
    dequantize_tensor_per_tensor_affine_stub);
// 声明分派函数：在每个通道上进行按通道仿射反量化分派函数
DECLARE_DISPATCH(
    dequantize_tensor_per_channel_affine_fn,
    dequantize_tensor_per_channel_affine_stub);
// 声明分派函数：使用浮点参数进行每通道仿射反量化分派函数
DECLARE_DISPATCH(
    dequantize_tensor_per_channel_float_qparams_fn,
    dequantize_tensor_per_channel_float_qparams_stub);
    dequantize_tensor_per_channel_float_qparams_fn,
    # 调用 dequantize_tensor_per_channel_float_qparams_fn 函数
    dequantize_tensor_per_channel_float_qparams_stub);
    # 调用 dequantize_tensor_per_channel_float_qparams_stub 函数
// 声明量化函数的调度器，用于按张量级别进行仿射子字节减法
DECLARE_DISPATCH(
    quantize_tensor_per_tensor_affine_sub_byte_fn,
    quantize_tensor_per_tensor_affine_sub_byte_stub);

// 声明反量化函数的调度器，用于按张量级别进行仿射子字节减法
DECLARE_DISPATCH(
    dequantize_tensor_per_tensor_affine_sub_byte_fn,
    dequantize_tensor_per_tensor_affine_sub_byte_stub);

// 定义模板函数，用于将张量按给定的缩放因子和零点进行量化
template <typename T>
TORCH_API Tensor quantize_tensor(
    Tensor rtensor,     // 实数张量，待量化的输入张量
    Tensor qtensor,     // 量化后的张量，存储量化后的结果
    double scale,       // 缩放因子，用于量化操作
    int64_t zero_point  // 零点，用于量化操作
);

// 定义模板函数，用于将张量按给定的缩放因子和零点进行反量化
template <typename T>
TORCH_API Tensor dequantize_tensor(
    Tensor qtensor,     // 量化张量，待反量化的输入张量
    Tensor rtensor,     // 实数张量，存储反量化后的结果
    double scale,       // 缩放因子，用于反量化操作
    int64_t zero_point  // 零点，用于反量化操作
);

} // namespace native
} // namespace at
```