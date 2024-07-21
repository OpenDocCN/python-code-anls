# `.\pytorch\aten\src\ATen\quantized\Quantizer.h`

```py
#pragma once
// 预处理指令，确保头文件只被包含一次

#include <c10/core/QScheme.h>
#include <c10/core/MemoryFormat.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <c10/util/intrusive_ptr.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorOptions.h>

#include <ATen/Tensor.h>
#include <ATen/TensorUtils.h>

#include <ATen/core/QuantizerBase.h>

#include <cmath>
#include <memory>
#include <utility>

namespace at {
// 命名空间声明，包含 ATen 库的所有内容

/**
 * UnknownQuantizer 是一个占位量化器，用于实现量化的两步过程。
 * 首先分配一个张量，但使用未知的量化器，然后量化内核决定最终的量化器。
 */
struct TORCH_API UnknownQuantizer : public Quantizer {
  explicit UnknownQuantizer(ScalarType scalar_type)
    : Quantizer(scalar_type) {}

  // 量化操作，覆盖基类的虚函数
  Tensor quantize(const Tensor& tensor) override;
  // 反量化操作，覆盖基类的虚函数
  Tensor dequantize(const Tensor& qtensor) override;
  // 带输出参数的反量化操作，覆盖基类的虚函数
  Tensor& dequantize_out(Tensor& rtensor, const Tensor& qtensor) override;
  // 返回量化方案，覆盖基类的虚函数
  QScheme qscheme() const override;
  // 比较函数，覆盖基类的虚函数
  bool equalTo(QuantizerPtr other) const override;
};

/**
 * UniformQuantizer 是所有均匀量化器的父类。
 * 这些量化方案将浮点值均匀映射到量化值。
 * 例如，仿射量化器是这一类别中最常用的方案。
 */
struct TORCH_API UniformQuantizer : public Quantizer {
  explicit UniformQuantizer(ScalarType scalar_type) : Quantizer(scalar_type) {}
};

/**
 * NonUniformQuantizer 是所有非均匀量化器的父类。
 * 这些量化方案可能会将浮点值非均匀地映射到量化值。
 * K-means 量化是这一类别中的一个典型例子。
 */
struct TORCH_API NonUniformQuantizer : public Quantizer {
  explicit NonUniformQuantizer(ScalarType scalar_type) : Quantizer(scalar_type) {}
};

// 还有一种 StochasticQuantizer 是均匀但不是仿射的量化器

/**
 * AffineQuantizer 使用仿射变换进行量化。
 *
 * 对于量化操作：
 * Y = clamp(round(X / scale + zero_point), min, max)
 * 对于反量化操作：
 * X = (Y - zero_point) * scale
 */
struct TORCH_API AffineQuantizer : public UniformQuantizer {
  explicit AffineQuantizer(ScalarType scalar_type) : UniformQuantizer(scalar_type) {}
};

// 请注意，我们不会在后端中使用对称量化器，以减少量化内核实现中的复杂性。

/**
 * PerTensorAffineQuantizer 存储一个用于张量中所有值的尺度和零点。
 */
struct TORCH_API PerTensorAffineQuantizer : public AffineQuantizer {
  explicit PerTensorAffineQuantizer(ScalarType scalar_type, double scale, int64_t zero_point)
    : AffineQuantizer(scalar_type),
        scale_(scale),
        zero_point_(zero_point) {}

  // 量化操作，覆盖基类的虚函数
  Tensor quantize(const Tensor& tensor) override;
  // 反量化操作，覆盖基类的虚函数
  Tensor dequantize(const Tensor& qtensor) override;
  // 带输出参数的反量化操作，覆盖基类的虚函数
  Tensor& dequantize_out(Tensor& rtensor, const Tensor& qtensor) override;

  // 返回量化方案，覆盖基类的虚函数
  QScheme qscheme() const override {
    返回量化器对象的量化方案
  }
  
  double scale() const {
    返回私有成员变量 scale_ 的值
  }

  int64_t zero_point() const {
    返回私有成员变量 zero_point_ 的值
  }

  bool equalTo(QuantizerPtr other) const override {
    检查传入的量化器指针是否存在且其量化方案是否为 kPerTensorAffine
    if (!other.get() || other->qscheme() != kPerTensorAffine) {
      如果不满足条件则返回 false
      return false;
    }
    将 other 转换为 PerTensorAffineQuantizer 类型的指针
    auto* other_per_tensor_affine =
        static_cast<PerTensorAffineQuantizer*>(other.get());
    比较当前对象和 other_per_tensor_affine 的标量类型、scale 和 zero_point 是否相等
    return scalar_type() == other_per_tensor_affine->scalar_type() &&
        scale() == other_per_tensor_affine->scale() &&
        zero_point() == other_per_tensor_affine->zero_point();
  }

 private:
  const double scale_;
  // We use int64_t for consistency with Python
  const int64_t zero_point_;
/**
 * PerChannelAffineQuantizer is the same as PerTensorAffineQuantizer
 * except that we have an independent scale and zero_point parameter
 * for each channel.
 *
 * Also note that per channel quantization is mostly applied to output channels
 * of weights since per-input channel of weight quantization or per-channel
 * quantization for activations can't be efficiently supported in most of
 * processors since it requires each multiplication result within a single
 * dot-product to have a different scale.
 */
struct TORCH_API PerChannelAffineQuantizer : public AffineQuantizer {
  explicit PerChannelAffineQuantizer(
      ScalarType scalar_type,           // 构造函数，接受标量类型
      Tensor scales,                    // 接受张量作为尺度参数
      Tensor zero_points,               // 接受张量作为零点参数
      int64_t axis)                     // 接受整数作为轴参数
      : AffineQuantizer(scalar_type),   // 调用基类构造函数初始化
        scales_(std::move(scales)),     // 使用 std::move 移动尺度参数到成员变量
        zero_points_(std::move(zero_points)), // 使用 std::move 移动零点参数到成员变量
        axis_(axis) {}                  // 初始化轴参数

  QScheme qscheme() const override {    // 返回量化方案，覆盖基类虚函数
    return kPerChannelAffine;           // 返回 Per Channel Affine 的量化方案
  }

  Tensor scales() const {               // 返回尺度张量
    return scales_;
  }

  Tensor zero_points() const {          // 返回零点张量
    return zero_points_;
  }

  int64_t axis() const {                // 返回轴参数
    return axis_;
  }

  Tensor quantize(const Tensor& tensor) override;   // 量化函数声明
  Tensor dequantize(const Tensor& qtensor) override; // 反量化函数声明
  Tensor& dequantize_out(Tensor& rtensor, const Tensor& qtensor) override; // 输出反量化函数声明

  bool equalTo(QuantizerPtr other) const override { // 判断两个量化器是否相等，覆盖基类虚函数
    if (!other.get() || other->qscheme() != kPerChannelAffine) { // 如果参数为空或者量化方案不同
      return false;                     // 返回不相等
    }
    auto* other_per_channel_affine =
        static_cast<PerChannelAffineQuantizer*>(other.get()); // 转换为 PerChannelAffineQuantizer 指针
    return scalar_type() == other_per_channel_affine->scalar_type() && // 比较标量类型
        scales().equal(other_per_channel_affine->scales()) && // 比较尺度张量
        zero_points().equal(other_per_channel_affine->zero_points()) && // 比较零点张量
        axis() == other_per_channel_affine->axis(); // 比较轴参数
  }

 protected:
  Tensor scales_;                       // 尺度张量成员变量
  Tensor zero_points_;                  // 零点张量成员变量
  const int64_t axis_;                  // 轴参数常量成员
};

/**
 * PerChannelAffineFloatQParamsQuantizer is the same as PerChannelAffineQuantizer
 * except that it expects both scale and zero point to be floating point values.
 *
 * This quantizer uses the kPerChannelAffineFloatQParams qscheme which is a variant of
 * kPerChannelAffine.
 *
 * The quantize equation in this case looks like -
 * Xq = (Xf - zero_point) * inv_scale, where inv_scale = 1.0/scale
 *
 * Note: Usage of floating point zero point is useful in cases where 0 doesn't need to
 * be exactly represented in the quantized space. We can get additional precision by
 * using floating point values for zero point.
 */
struct TORCH_API PerChannelAffineFloatQParamsQuantizer : public PerChannelAffineQuantizer {
  explicit PerChannelAffineFloatQParamsQuantizer(
      ScalarType scalar_type,           // 构造函数，接受标量类型
      Tensor scales,                    // 接受张量作为尺度参数
      Tensor zero_points,               // 接受张量作为零点参数
      int64_t axis)                     // 接受整数作为轴参数
      : PerChannelAffineQuantizer(scalar_type, // 调用基类构造函数初始化
        scales,
        zero_points,
        axis) {}

  QScheme qscheme() const override {    // 返回量化方案，覆盖基类虚函数
    return kPerChannelAffineFloatQParams; // 返回带有浮点参数的 Per Channel Affine 量化方案
  }
  # 返回当前量化器的量化参数，这里是一个常量，表示每通道的仿射浮点量化参数
  return kPerChannelAffineFloatQParams;
}

# 对输入张量进行量化操作，返回量化后的张量
Tensor quantize(const Tensor& tensor) override;

# 对量化后的张量进行反量化操作，返回反量化后的张量
Tensor dequantize(const Tensor& qtensor) override;

# 对输入的量化张量进行反量化操作，并将结果存储到输出张量中，返回输出张量的引用
Tensor& dequantize_out(Tensor& rtensor, const Tensor& qtensor) override;

# 检查当前量化器是否与另一个量化器相等
bool equalTo(QuantizerPtr other) const override {
  # 如果输入的量化器为空或者其量化模式不是每通道仿射浮点量化，则返回false
  if (!other.get() || other->qscheme() != kPerChannelAffineFloatQParams) {
    return false;
  }
  # 将other转换为PerChannelAffineFloatQParamsQuantizer类型的指针，并进行详细比较
  auto* other_per_channel_float_qparams =
      static_cast<PerChannelAffineFloatQParamsQuantizer*>(other.get());
  # 比较量化器的标量类型、缩放因子、零点和轴是否相同，若全部相同则返回true，否则返回false
  return scalar_type() == other_per_channel_float_qparams->scalar_type() &&
      scales().equal(other_per_channel_float_qparams->scales()) &&
      zero_points().equal(other_per_channel_float_qparams->zero_points()) &&
      axis() == other_per_channel_float_qparams->axis();
}
// 这是一个用于获取 QTensorImpl 的内部实用函数，
// 只应用于编写 QTensorImpl 字段的低级设置器/获取器；
// 否则，应使用使用了此函数实现的低级设置器/获取器。
// 可能会重复调用，因此确保其开销较低。

TORCH_API QTensorImpl* get_qtensorimpl(const TensorBase& self);

// double 和 int64_t 是因为本机函数 API 的限制，目前我们只有这些参数类型
TORCH_API QuantizerPtr
make_per_tensor_affine_quantizer(
    double scale, int64_t zero_point, ScalarType scalar_type);

// 创建一个基于每通道的仿射量化器
TORCH_API QuantizerPtr make_per_channel_affine_quantizer(
    const Tensor& scales,
    const Tensor& zero_points,
    int64_t axis,
    ScalarType scalar_type);

// 创建一个未知量化器
TORCH_API QuantizerPtr make_unknown_quantizer(ScalarType scalar_type);

// 创建一个量化张量，给定普通张量和量化器的参数
TORCH_API Tensor new_qtensor(
    IntArrayRef sizes,
    const TensorOptions& options,
    QuantizerPtr quantizer);

// 设置张量的量化器
TORCH_API void set_quantizer_(const Tensor& self, ConstQuantizerPtr quantizer);

// 从量化数据块创建基于每张量的仿射量化张量
TORCH_API Tensor from_blob_quantized_per_tensor_affine(
    void* data,
    IntArrayRef sizes,
    IntArrayRef strides,
    std::function<void(void*)> deleter,
    const float scale,
    const int64_t zeroPoint,
    const TensorOptions& options);

// 从量化数据块创建基于每张量的仿射量化张量，不带步长信息
TORCH_API Tensor from_blob_quantized_per_tensor_affine(
    void* data,
    IntArrayRef sizes,
    std::function<void(void*)> deleter,
    const float scale,
    const int64_t zeroPoint,
    const TensorOptions& options);

// 从量化数据块创建基于每通道的仿射量化张量
TORCH_API Tensor from_blob_quantized_per_channel_affine(
    void* data,
    IntArrayRef sizes,
    std::function<void(void*)> deleter,
    const Tensor& scales,
    const Tensor& zero_points,
    const int64_t axis,
    const TensorOptions& options);

} // namespace at
```