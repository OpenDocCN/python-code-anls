# `.\pytorch\aten\src\ATen\native\quantized\AffineQuantizer.cpp`

```
namespace at {
namespace native {

// 定义分发函数调度器，用于量化和反量化操作的分发
DEFINE_DISPATCH(quantize_tensor_per_tensor_affine_stub);
DEFINE_DISPATCH(quantize_tensor_per_channel_affine_stub);
DEFINE_DISPATCH(quantize_tensor_per_channel_float_qparams_stub);
DEFINE_DISPATCH(dequantize_tensor_per_tensor_affine_stub);
DEFINE_DISPATCH(dequantize_tensor_per_channel_affine_stub);
DEFINE_DISPATCH(dequantize_tensor_per_channel_float_qparams_stub);
DEFINE_DISPATCH(quantize_tensor_per_tensor_affine_sub_byte_stub);
DEFINE_DISPATCH(dequantize_tensor_per_tensor_affine_sub_byte_stub);

namespace {

// 检查舍入模式是否正确设置
void checkRoundingMode(const std::string& fn_name) {
  // 现在暂时禁用此警告消息，因为它的打印方式不正确，需要修复
  return;
}

// 检查输入张量是否为浮点型
void checkFloatTensor(const std::string& fn_name, const Tensor& t) {
  TORCH_CHECK(
      t.scalar_type() == kFloat, fn_name, " expects a Float Tensor, got ",
      t.scalar_type());
}

// 检查两个张量是否在同一设备上
void checkSameDevice(
    const std::string& fn_name,
    const Tensor& t1,
    const Tensor& t2) {
  TORCH_CHECK(
      t1.device() == t2.device(),
      fn_name,
      " expects a quantized and float tensors to be on the same device.");
}

// 检查量化张量的类型是否正确
template <typename T>
void checkQuantizedTensor(const std::string& fn_name, const Tensor& t) {
  TORCH_CHECK(t.is_quantized(), fn_name, " expects a quantized Tensor.");
  TORCH_CHECK(
      t.scalar_type() == caffe2::TypeMeta::Make<T>(),
      fn_name,
      " expects a ",
      caffe2::TypeMeta::Make<T>(),
      " Tensor, got ",
      t.scalar_type());
}

// 检查零点值是否在有效范围内
template <typename T>
void checkZeroPoint(const std::string& fn_name, int64_t zero_point) {
  TORCH_CHECK(
      zero_point <= std::numeric_limits<T>::max(),
      fn_name,
      " zero_point ",
      zero_point,
      " is above upper bound.");
  TORCH_CHECK(
      zero_point >= std::numeric_limits<T>::min(),
      fn_name,
      " zero_point ",
      zero_point,
      " is below lower bound.");
}

// 检查零点张量中每个零点值是否在有效范围内
template <typename T>
void checkZeroPoints(const std::string& fn_name, const Tensor& zero_points) {
  auto zero_points_data = zero_points.data_ptr<int64_t>();
  for (const auto i : c10::irange(zero_points.numel())) {
    checkZeroPoint<T>(fn_name, zero_points_data[i]);
  }
}

// 检查两个张量是否具有相同的大小
void checkSameSize(
    const std::string& fn_name,
    const Tensor& qt,
    const Tensor& rt) {
  TORCH_CHECK(
      qt.sizes().equals(rt.sizes()),
      fn_name,
      " only works with Tensors with the same shape");
}

// 检查通道参数的大小是否匹配
void checkPerChannelParamsSize(
    const Tensor& rtensor,
    int64_t axis,
    const Tensor& scales,
    const Tensor& zero_points) {
  // 省略的部分需要补充完整
}
// 匿名命名空间结束

Tensor& quantize_tensor_per_tensor_affine(
    const Tensor& rtensor,
    Tensor& qtensor,
    double scale,
    int64_t zero_point) {
  static constexpr auto fn_name = "quantize_tensor_per_tensor_affine";

  // 检查量化模式的舍入方式
  checkRoundingMode(fn_name);
  // 检查输入张量是否为浮点数张量
  checkFloatTensor(fn_name, rtensor);
  // 检查输入张量和输出张量是否在相同的设备上
  checkSameDevice(fn_name, rtensor, qtensor);
  // 检查输入张量和输出张量是否具有相同的大小
  checkSameSize(fn_name, qtensor, rtensor);

  // 根据量化张量的数据类型执行特定操作
  // 如果数据类型为 torch.quint4x2 或 torch.quint2x4，则调用对应的量化函数
  // 否则，调用普通的量化函数
  if (qtensor.scalar_type() == at::ScalarType::QUInt4x2 || qtensor.scalar_type() == at::ScalarType::QUInt2x4) {
    quantize_tensor_per_tensor_affine_sub_byte_stub(
        rtensor.device().type(), rtensor, qtensor, scale, zero_point);
  } else {
    quantize_tensor_per_tensor_affine_stub(
        rtensor.device().type(), rtensor, qtensor, scale, zero_point);
  }
  return qtensor;
}

Tensor& quantize_tensor_per_channel_affine(
    const Tensor& rtensor,
    Tensor& qtensor,
    Tensor scales,
    Tensor zero_points,
    int64_t axis) {
  static constexpr auto fn_name = "quantize_tensor_per_channel_affine";

  // 检查量化模式的舍入方式
  checkRoundingMode(fn_name);
  // 检查输入张量是否为浮点数张量
  checkFloatTensor(fn_name, rtensor);
  // 检查输入张量和输出张量是否在相同的设备上
  checkSameDevice(fn_name, rtensor, qtensor);
  // 检查输入张量和输出张量是否具有相同的大小
  checkSameSize(fn_name, qtensor, rtensor);

  // 根据量化张量的数据类型执行特定操作
  AT_DISPATCH_QINT_TYPES(qtensor.scalar_type(), fn_name, [&]() {
    // 检查量化张量是否满足特定要求
    checkQuantizedTensor<scalar_t>(fn_name, qtensor);
    // 如果量化张量的设备类型不是 CUDA 或 PrivateUse1，则检查量化零点的数值
    if (qtensor.device().type() != c10::DeviceType::CUDA &&
        qtensor.device().type() != c10::DeviceType::PrivateUse1) {
      checkZeroPoints<underlying_t>(fn_name, zero_points);
    }  // 对于 CUDA 和 PrivateUse1 设备，此检查将在实际设备函数中执行
  });

  // 检查通道轴是否在合理范围内
  TORCH_CHECK(
      0 <= axis && axis < rtensor.dim(),
      "Channel axis out of range in per channel affine quantization. Got: ",
      axis,
      "Expected: [0, ",
      rtensor.dim(),
      ")");
  // 检查每通道量化参数的尺寸
  checkPerChannelParamsSize(rtensor, axis, scales, zero_points);

  // 调用特定的每通道量化函数
  quantize_tensor_per_channel_affine_stub(
      rtensor.device().type(), rtensor, qtensor, scales, zero_points, axis);
  return qtensor;
}

// 下面是未完整的代码块，无法提供完整的注释
    // 定义静态常量字符串，表示当前函数名
    static constexpr auto fn_name =
        "quantize_tensor_per_channel_float_qparams";

    // 检查舍入模式是否正确
    checkRoundingMode(fn_name);
    // 检查实数张量是否有效
    checkFloatTensor(fn_name, rtensor);
    // 检查两个张量是否在同一设备上
    checkSameDevice(fn_name, rtensor, qtensor);
    // 检查两个张量是否具有相同的大小
    checkSameSize(fn_name, qtensor, rtensor);

    // 使用 AT_DISPATCH_QINT_AND_SUB_BYTE_TYPES 宏分发量化整数和子字节类型的操作
    // NOLINTNEXTLINE(clang-diagnostic-unused-variable)
    AT_DISPATCH_QINT_AND_SUB_BYTE_TYPES(qtensor.scalar_type(), fn_name, [&]() {
      // 检查量化张量是否正确
      checkQuantizedTensor<scalar_t>(fn_name, qtensor);
    });

    // 检查通道轴是否在范围内
    TORCH_CHECK(
        0 <= axis && axis < rtensor.dim(),
        "Channel axis out of range in per channel float qparams quantization. Got: ",
        axis,
        "Expected: [0, ",
        rtensor.dim(),
        ")");

    // 检查按通道参数的大小是否正确
    checkPerChannelParamsSize(rtensor, axis, scales, zero_points);

    // 调用量化函数的底层实现
    quantize_tensor_per_channel_float_qparams_stub(
        rtensor.device().type(), rtensor, qtensor, scales, zero_points, axis);
    // 返回量化后的张量
    return qtensor;
}

// 对张量进行每通道仿射量化逆过程，基于张量级别的仿射参数
Tensor& dequantize_tensor_per_tensor_affine(
    const Tensor& qtensor,
    Tensor& rtensor,
    double scale,
    int64_t zero_point) {
  static constexpr auto fn_name = "dequantize_tensor_per_tensor_affine";
  // 检查目标张量是否为浮点类型
  checkFloatTensor(fn_name, rtensor);
  // 检查目标张量和输入张量是否在同一设备上
  checkSameDevice(fn_name, rtensor, qtensor);
  // 检查目标张量和输入张量是否具有相同的尺寸
  checkSameSize(fn_name, qtensor, rtensor);

  // 根据量化张量的数据类型分发操作，检查量化张量的有效性和零点
  AT_DISPATCH_QINT_AND_SUB_BYTE_TYPES(qtensor.scalar_type(), fn_name, [&]() {
    checkQuantizedTensor<scalar_t>(fn_name, qtensor);
    checkZeroPoint<underlying_t>(fn_name, zero_point);
  });

  // 根据量化张量的数据类型选择相应的仿射量化逆过程的实现
  if (qtensor.scalar_type() == at::ScalarType::QUInt4x2 || qtensor.scalar_type() == at::ScalarType::QUInt2x4) {
    dequantize_tensor_per_tensor_affine_sub_byte_stub(
        qtensor.device().type(), qtensor, rtensor, scale, zero_point);
  } else {
    dequantize_tensor_per_tensor_affine_stub(
        qtensor.device().type(), qtensor, rtensor, scale, zero_point);
  }
  // 返回反量化后的张量
  return rtensor;
}

// 对张量进行每通道仿射量化逆过程，基于通道级别的仿射参数
Tensor& dequantize_tensor_per_channel_affine(
    const Tensor& qtensor,
    Tensor& rtensor,
    Tensor scales,
    Tensor zero_points,
    int64_t axis) {
  static constexpr auto fn_name = "dequantize_tensor_per_channel_affine";

  // 检查目标张量是否为浮点类型
  checkFloatTensor(fn_name, rtensor);
  // 检查目标张量和输入张量是否在同一设备上
  checkSameDevice(fn_name, rtensor, qtensor);
  // 检查目标张量和输入张量是否具有相同的尺寸
  checkSameSize(fn_name, qtensor, rtensor);

  // 根据量化张量的数据类型分发操作，检查量化张量的有效性和零点
  AT_DISPATCH_QINT_TYPES(qtensor.scalar_type(), fn_name, [&]() {
    checkQuantizedTensor<scalar_t>(fn_name, qtensor);
    // 对于非 CUDA 设备和私有设备，检查仿射量化参数的零点
    if(qtensor.device().type() != c10::DeviceType::CUDA &&
       qtensor.device().type() != c10::DeviceType::PrivateUse1){
      checkZeroPoints<underlying_t>(fn_name, zero_points);
    }  // 对于 CUDA 和私有设备，这个检查将在实际设备函数中进行
  });

  // 检查通道轴是否在合理范围内
  TORCH_CHECK(
      0 <= axis && axis < qtensor.dim(),
      "Channel axis out of range in per channel affine dequantization. Got:",
      axis,
      " Expected: [0, ",
      qtensor.dim(),
      ")");
  // 检查每通道仿射参数的尺寸是否匹配
  checkPerChannelParamsSize(rtensor, axis, scales, zero_points);

  // 根据量化张量的数据类型选择相应的仿射量化逆过程的实现
  dequantize_tensor_per_channel_affine_stub(
      qtensor.device().type(), qtensor, rtensor, scales, zero_points, axis);
  // 返回反量化后的张量
  return rtensor;
}

// 对张量进行每通道浮点量化参数的仿射量化逆过程
Tensor& dequantize_tensor_per_channel_float_qparams(
    const Tensor& qtensor,
    Tensor& rtensor,
    Tensor scales,
    Tensor zero_points,
    int64_t axis) {
  static constexpr auto fn_name = "dequantize_tensor_per_channel_affine";

  // 检查目标张量是否为浮点类型
  checkFloatTensor(fn_name, rtensor);
  // 检查目标张量和输入张量是否在同一设备上
  checkSameDevice(fn_name, rtensor, qtensor);
  // 检查目标张量和输入张量是否具有相同的尺寸
  checkSameSize(fn_name, qtensor, rtensor);

  // 根据量化张量的数据类型分发操作，检查量化张量的有效性和零点
  AT_DISPATCH_QINT_AND_SUB_BYTE_TYPES(qtensor.scalar_type(), fn_name, [&]() {
  // 对每个量化张量执行检查函数，检查是否匹配给定的标量类型
  checkQuantizedTensor<scalar_t>(fn_name, qtensor);
});

// 检查通道轴是否在张量维度范围内，用于通道间浮点量化参数反量化
TORCH_CHECK(
    0 <= axis && axis < qtensor.dim(),
    "Channel axis out of range in per channel float qparams dequantization. Got:",
    axis,
    " Expected: [0, ",
    qtensor.dim(),
    ")");

// 检查在按通道浮点量化参数反量化时，参数的尺寸是否匹配
checkPerChannelParamsSize(rtensor, axis, scales, zero_points);

// 调用针对按通道浮点量化参数反量化的底层函数
dequantize_tensor_per_channel_float_qparams_stub(
    qtensor.device().type(), qtensor, rtensor, scales, zero_points, axis);
// 返回反量化后的张量
return rtensor;
}

} // namespace native
} // namespace at
```