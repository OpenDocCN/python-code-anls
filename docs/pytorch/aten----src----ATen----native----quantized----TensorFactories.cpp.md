# `.\pytorch\aten\src\ATen\native\quantized\TensorFactories.cpp`

```py
// 引入 ATen 库中所需的头文件
#include <ATen/ATen.h>
#include <ATen/quantized/Quantizer.h>
#include <c10/core/QScheme.h>
#include <c10/core/TensorOptions.h>

// 声明位于 at 命名空间下的 native 命名空间
namespace at {
namespace native {

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ empty ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// 由于当前 Python 前端不支持量化器，故显式传递 scale 和 zero_point；
// 一旦支持，将转为使用量化器
Tensor empty_affine_quantized(
    IntArrayRef size,                                               // 数组尺寸
    std::optional<ScalarType> dtype,                                // 可选的数据类型
    std::optional<Layout> layout,                                   // 可选的布局
    std::optional<Device> device,                                   // 可选的设备
    std::optional<bool> pin_memory,                                 // 可选的内存固定
    double scale,                                                   // 量化的比例因子
    int64_t zero_point,                                             // 量化的零点
    std::optional<c10::MemoryFormat> optional_memory_format) {       // 可选的内存格式
  // 查看 [Note: hacky wrapper removal for TensorOptions]
  TensorOptions options_ = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

  // 检查是否在 TensorOptions 和显式参数中都设置了 memory_format；如是，报错
  TORCH_CHECK(
    !(options_.has_memory_format() && optional_memory_format.has_value()),
    "Cannot set memory_format both in TensorOptions and explicit argument; please delete "
    "the redundant setter.");
  auto options = options_.merge_memory_format(optional_memory_format);

  // 检查是否提供了数据类型，否则报错
  TORCH_CHECK(
      options.has_dtype(),
      "Must provide data type for Tensor creation functions.");

  // 使用给定的 scale 和 zero_point 创建 per-tensor affine 量化器，返回量化后的 Tensor
  return new_qtensor(
      size,
      options,
      make_per_tensor_affine_quantizer(
          scale, zero_point, typeMetaToScalarType(options.dtype())));
}

// 创建 per-channel affine 量化 Tensor
Tensor empty_per_channel_affine_quantized(
    IntArrayRef size,                                               // 数组尺寸
    const Tensor& scales,                                           // 缩放因子 Tensor
    const Tensor& zero_points,                                      // 零点 Tensor
    int64_t axis,                                                   // 量化轴
    std::optional<ScalarType> dtype,                                // 可选的数据类型
    std::optional<Layout> layout,                                   // 可选的布局
    std::optional<Device> device,                                   // 可选的设备
    std::optional<bool> pin_memory,                                 // 可选的内存固定
    std::optional<c10::MemoryFormat> optional_memory_format) {       // 可选的内存格式
  // 查看 [Note: hacky wrapper removal for TensorOptions]
  TensorOptions options_ = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

  // 检查是否在 TensorOptions 和显式参数中都设置了 memory_format；如是，报错
  TORCH_CHECK(
    !(options_.has_memory_format() && optional_memory_format.has_value()),
    "Cannot set memory_format both in TensorOptions and explicit argument; please delete "
    "the redundant setter.");
  auto options = options_.merge_memory_format(optional_memory_format);

  // 检查是否提供了数据类型，否则报错
  TORCH_CHECK(
      options.has_dtype(),
      "Must provide data type for Tensor creation functions.");

  // 创建 per-channel affine 量化器，将其用于创建量化后的 Tensor
  QuantizerPtr quantizer = make_per_channel_affine_quantizer(
          scales.to(options.device()), zero_points.to(options.device()), axis, typeMetaToScalarType(options.dtype()));
  return new_qtensor(
      size,
      options,
      std::move(quantizer));
}

// 创建未知量化方式的量化 Tensor
Tensor empty_unknown_quantized(
    IntArrayRef size,                                               // 数组尺寸
    std::optional<ScalarType> dtype,                                // 可选的数据类型
    std::optional<Layout> layout,                                   // 可选的布局
    std::optional<Device> device,                                   // 可选的设备
    std::optional<bool> pin_memory,                                 // 可选的内存固定
    double scale,                                                   // 量化的比例因子
    int64_t zero_point,                                             // 量化的零点
    std::optional<c10::MemoryFormat> optional_memory_format) {       // 可选的内存格式
    // 创建一个 TensorOptions 对象 options_，设定其数据类型 dtype、布局 layout、设备 device、以及是否使用固定内存 pin_memory
    TensorOptions options_ = TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);
    
    // 检查是否同时在 TensorOptions 和显式参数中设置了 memory_format，如果是则抛出错误提示信息
    TORCH_CHECK(
      !(options_.has_memory_format() && optional_memory_format.has_value()),
      "Cannot set memory_format both in TensorOptions and explicit argument; please delete "
      "the redundant setter.");
    
    // 将 optional_memory_format 合并到 options_ 中，形成最终的 TensorOptions 对象 options
    auto options = options_.merge_memory_format(optional_memory_format);
    
    // 检查 options 是否设置了数据类型，如果没有设置则抛出错误提示信息
    TORCH_CHECK(
        options.has_dtype(),
        "Must provide data type for Tensor creation functions.");
    
    // 根据 options 的数据类型创建一个未知量化器 QuantizerPtr
    QuantizerPtr quantizer = make_unknown_quantizer(typeMetaToScalarType(options.dtype()));
    
    // 返回使用指定大小、选项和量化器创建的新量化 Tensor 对象
    return new_qtensor(size, options, std::move(quantizer));
// 提供错误消息，指出当前不支持在量化张量上使用 empty_strided，请参阅指定的 GitHub 问题页面
Tensor empty_strided_unknown_quantized(
    IntArrayRef size,
    IntArrayRef strided,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {

  TORCH_CHECK(false, "empty_strided not supported on quantized tensors yet see https://github.com/pytorch/pytorch/issues/74540")
}

// 如果 dtype 类型错误，提供更好的错误消息
Tensor empty_affine_quantized_other_backends_stub(
    IntArrayRef,
    std::optional<ScalarType>,
    std::optional<Layout>,
    std::optional<Device>,
    std::optional<bool>,
    double,
    int64_t,
    std::optional<c10::MemoryFormat>) {
  TORCH_CHECK(false, "Creation of quantized tensor requires quantized dtype like torch.quint8");
}

// 如果 dtype 类型错误，提供更好的错误消息
Tensor empty_per_channel_affine_quantized_other_backends_stub(
    IntArrayRef,
    const Tensor&,
    const Tensor&,
    int64_t,
    std::optional<ScalarType>,
    std::optional<Layout>,
    std::optional<Device>,
    std::optional<bool>,
    std::optional<c10::MemoryFormat>) {
  TORCH_CHECK(false, "Creation of quantized tensor requires quantized dtype like torch.quint8");
}

// 根据指定的选项和输入量化张量的量化参数，创建一个空的量化张量
Tensor empty_quantized(
    IntArrayRef size,
    const Tensor& qtensor,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory,
    std::optional<c10::MemoryFormat> memory_format) {
  // 创建包含指定选项的 TensorOptions 对象
  TensorOptions specified_options =
      TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

  // 检查不能同时在 TensorOptions 和显式参数中设置 memory_format；请删除多余的设置器
  TORCH_CHECK(
      !(specified_options.has_memory_format() && memory_format.has_value()),
      "Cannot set memory_format both in TensorOptions and explicit argument; please delete "
      "the redundant setter.");

  // 合并量化张量的选项和指定的选项，包括内存格式
  TensorOptions options = qtensor.options()
                              .merge_in(specified_options)
                              .merge_memory_format(memory_format);

  Tensor output;
  // 根据量化张量的量化策略不同，选择不同的创建函数
  if (qtensor.qscheme() == kPerTensorAffine) {
    output = at::_empty_affine_quantized(
        size, options, qtensor.q_scale(), qtensor.q_zero_point());
  } else if (
      qtensor.qscheme() == kPerChannelAffine ||
      qtensor.qscheme() == kPerChannelAffineFloatQParams) {
    output = at::_empty_per_channel_affine_quantized(
        size,
        qtensor.q_per_channel_scales(),
        qtensor.q_per_channel_zero_points(),
        qtensor.q_per_channel_axis(),
        options);
  } else {
    // 如果量化策略不被支持，抛出错误
    TORCH_CHECK(
        false,
        "QScheme not supported by empty_quantized:",
        toString(qtensor.qscheme()));
  }
  return output;
}

} // namespace native
} // namespace at
```