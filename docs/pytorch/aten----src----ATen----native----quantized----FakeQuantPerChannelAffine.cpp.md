# `.\pytorch\aten\src\ATen\native\quantized\FakeQuantPerChannelAffine.cpp`

```py
// 包含 ATen 库中的相关头文件
#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>
#include <ATen/native/quantized/FakeQuantAffine.h>

#include <c10/util/irange.h>

// 用于定义 PerChannelAffine 量化方案的 FakeQuantize 操作符
namespace at {
namespace native {

// 使用 REGISTER_DISPATCH 宏来运行 CPU 和 CUDA 后端
DEFINE_DISPATCH(fake_quant_per_channel_cachemask_stub); // 定义了一个分发函数的存根，用于执行缓存掩码版本的 per-channel fake quantize
DEFINE_DISPATCH(fake_quant_grad_learnable_channel_stub); // 定义了一个分发函数的存根，用于执行可学习通道版本的 fake quantize 梯度计算

/* Per channel fake-quantizes the 'inputs' tensor.
Args:
  X: Forward input tensor. 前向输入张量
  dY: Backward input tensor (_backward op only). 反向输入张量（仅限 _backward 操作）
  scale: scale of per channel affine quantization 每通道仿射量化的缩放因子
  zero_point: zero_point of per channel affine quantization 每通道仿射量化的零点
  axis: int specifying the axis to be quantized 指定要量化的轴
  quant_min: minimum quantized value 最小量化值
  quant_max: maximum quantized value 最大量化值
Returns:
  Fake quantized tensor (double dtype). 返回伪量化后的张量（双精度类型）。
*/

Tensor fake_quantize_per_channel_affine(
    const Tensor& self,
    const Tensor& scale,
    const Tensor& zero_point,
    int64_t axis,
    int64_t quant_min,
    int64_t quant_max) {
  // 调用 fake_quantize_per_channel_affine_cachemask 函数执行 per-channel fake quantize 操作
  const auto res = at::fake_quantize_per_channel_affine_cachemask(
      self, scale, zero_point, axis, quant_min, quant_max);
  // 返回结果中的第一个张量，即伪量化后的结果
  return std::get<0>(res);
}

// 实现了带缓存掩码的 per-channel 仿量化函数
std::tuple<Tensor, Tensor> fake_quantize_per_channel_affine_cachemask(
    const Tensor& self,
    const Tensor& scale,
    const Tensor& zero_point,
    int64_t axis,
    int64_t quant_min,
    int64_t quant_max) {
    // 检查 scale 张量的数据类型是否为 Float
    // 如果不是，抛出错误信息并指出实际类型
    TORCH_CHECK(scale.scalar_type() == ScalarType::Float,
                "Scale must be Float, found ", scale.scalar_type());
    
    // 检查 zero_point 张量的数据类型是否为 Int、Float 或 Half
    // 如果不是，抛出错误信息并指出实际类型
    TORCH_CHECK(zero_point.scalar_type() == ScalarType::Int || zero_point.scalar_type() == ScalarType::Float || zero_point.scalar_type() == ScalarType::Half,
                "Zero-point must be Int32, Float or Half, found ", zero_point.scalar_type());
    
    // 检查 scale 张量是否为 1 维
    // 如果不是，抛出错误信息
    TORCH_CHECK(scale.dim() == 1, "scale should be a 1-D tensor");
    
    // 检查 zero_point 张量是否为 1 维
    // 如果不是，抛出错误信息
    TORCH_CHECK(zero_point.dim() == 1, "zero point should be a 1-D tensor");
    
    // 检查 scale 和 zero_point 张量的元素数量是否相等
    // 如果不相等，抛出错误信息
    TORCH_CHECK(
        scale.numel() == zero_point.numel(),
        "scale and zero-point need to have the same dimensions");
    
    // 检查 scale 和 zero_point 张量的元素数量是否与输入张量在指定轴上的大小一致
    // 如果不一致，抛出错误信息
    TORCH_CHECK(
        scale.numel() == self.size(axis),
        "dimensions of scale and zero-point are not consistent with input tensor")
    
    // 检查 quant_min 是否小于等于 quant_max
    // 如果不是，抛出错误信息
    TORCH_CHECK(
        quant_min <= quant_max,
        "`quant_min` should be less than or \
          equal to `quant_max`.");
    
    // 如果 zero_point 的数据类型不是浮点类型，
    // 检查 zero_point 张量的最小值和最大值是否在 quant_min 和 quant_max 范围内
    // 如果不在范围内，抛出错误信息
    if(!at::isFloatingType(zero_point.scalar_type())){
        TORCH_CHECK(
            at::min(zero_point).item().toInt() >= quant_min &&
                at::max(zero_point).item().toInt() <= quant_max,
            "`zero_point` must be between `quant_min` and `quant_max`.");
    }
    
    // 检查 axis 是否在输入张量的维度范围内
    // 如果不在范围内，抛出错误信息
    TORCH_CHECK(
        axis >= 0 && axis <= self.dim(),
        "`axis` must be between 0 and number of dimensions of input");
    
    // 根据输入张量创建一个和 self 具有相同配置的空张量 Y
    auto Y = at::empty_like(self, self.options(), MemoryFormat::Preserve);
    
    // 根据输入张量创建一个和 self 具有相同配置的空布尔张量 mask
    auto mask = at::empty_like(self, at::kBool, MemoryFormat::Preserve);
    
    // 创建一个期望的形状向量，初始化为每个维度为 1
    // 将期望的形状向量中的指定轴设置为输入张量在该轴上的大小
    std::vector<int64_t> expected_shape(self.dim(), 1);
    expected_shape[axis] = self.size(axis);
    
    // 使用 TensorIteratorConfig 配置对象创建一个迭代器 iter
    // 迭代器将 Y 作为输出，self、scale 和 zero_point 作为输入
    // native::_unsafe_view 用于将 scale 和 zero_point 转换为期望的形状
    TensorIterator iter = TensorIteratorConfig()
      .check_all_same_dtype(false)
      .add_output(Y)
      .add_input(self)
      .add_owned_input(native::_unsafe_view(scale, expected_shape))
      .add_owned_input(native::_unsafe_view(zero_point, expected_shape))
      .build();
    
    // 使用 TensorIteratorConfig 配置对象创建另一个迭代器 iter_mask
    // 迭代器将 mask 作为输出，self、scale 和 zero_point 作为输入
    // native::_unsafe_view 用于将 scale 和 zero_point 转换为期望的形状
    TensorIterator iter_mask = TensorIteratorConfig()
      .check_all_same_dtype(false)
      .add_output(mask)
      .add_input(self)
      .add_owned_input(native::_unsafe_view(scale, expected_shape))
      .add_owned_input(native::_unsafe_view(zero_point, expected_shape))
      .build();
    
    // 调用 fake_quant_per_channel_cachemask_stub 函数，
    // 该函数用于执行量化相关的操作，传入迭代器和相关参数
    fake_quant_per_channel_cachemask_stub(iter.device_type(), iter, iter_mask, quant_min, quant_max);
    
    // 返回一个元组，包含计算得到的 Y 和 mask 张量
    return std::make_tuple(Y, mask);
/* 返回 'dY' 关于 'X' 的梯度，以及 'scale' 和 'zero_point' 的梯度 */

std::tuple<Tensor, Tensor, Tensor> _fake_quantize_learnable_per_channel_affine_backward(
    const Tensor& dY,
    const Tensor& X,
    const Tensor& scale,
    const Tensor& zero_point,
    int64_t axis,
    int64_t quant_min,
    int64_t quant_max,
    double grad_factor) {
  // 检查 'mask' 张量的类型是否为布尔类型
  TORCH_CHECK(mask.scalar_type() == ScalarType::Bool);
  // 检查 'mask' 和 'dY' 的元素数量是否相同
  TORCH_CHECK(mask.numel() == dY.numel(),
      "`mask` and `dY` are not the same size: ",
      "`mask` is size ", mask.numel(), " and `dY` is size ", dY.numel());
  // 如果 'dY' 的元素数量小于等于 0，则直接返回 'dY'
  if (dY.numel() <= 0) {
    return dY;
  }
  // 注意：由于 'mask' 已预先计算，并且我们可以使用现有的张量乘法内核
  // 返回 'dY' 乘以 'mask'
  return dY * mask;
}

/* 获取四舍五入并在范围 [quant_min, quant_max] 内夹住的 'zero_point' 张量 */

static Tensor _get_rounded_zero_point(
    const Tensor& zero_point,
    int64_t quant_min,
    int64_t quant_max) {
  // 假设每通道的零点向量是单维度的
  return zero_point.round().clamp_(quant_min, quant_max);
}

/* 对每个通道可学习的仿射量化 */

Tensor _fake_quantize_learnable_per_channel_affine(
    const Tensor& self,
    const Tensor& scale,
    const Tensor& zero_point,
    int64_t axis,
    int64_t quant_min,
    int64_t quant_max,
    double grad_factor) {
  // 获取四舍五入并在范围 [quant_min, quant_max] 内夹住的 'zero_point' 张量
  Tensor zero_point_rounded = _get_rounded_zero_point(zero_point, quant_min, quant_max).to(at::kInt);
  // 调用 'native::fake_quantize_per_channel_affine' 函数
  return native::fake_quantize_per_channel_affine(
    self, scale, zero_point_rounded, axis, quant_min, quant_max);
}
    double grad_factor) {
  /* 计算尺度和零点的梯度如下：
     设 Xfq 为 X 的伪量化版本。
     设 Xq 为 X 的量化版本（在 qmin 和 qmax 上截断）。
     设 Delta 和 z 分别为尺度和零点。
     公式如下：
      \frac{d\Delta }{dx} =
        \begin{cases}
          q_{\min} - z& \text{ if } X_q= q_{\min} \\
          q_{\max} - z& \text{ if } X_q= q_{\max} \\
          (X_{fq} - X) / \Delta & \text{ else }
        \end{cases}

      \frac{dz }{dx} =
        \begin{cases}
          -\Delta& \text{ if } X_q= q_{\min} \text{ or } X_q = q_{\max} \\
          0 & \text{ else }
        \end{cases}
  */
  auto zero_point_rounded = _get_rounded_zero_point(zero_point, quant_min, quant_max);

  TORCH_CHECK(dY.scalar_type() == ScalarType::Float);
  TORCH_CHECK(X.scalar_type() == ScalarType::Float);
  TORCH_CHECK(scale.scalar_type() == ScalarType::Float);
  TORCH_CHECK(zero_point.scalar_type() == ScalarType::Float);

  TORCH_CHECK(X.sizes() == dY.sizes(), "`X` and `dY` are not the same size");
  TORCH_CHECK(
      quant_min <= 0 && quant_max >= 0,
      "Expecting `quant_min` <= 0 and `quant_max` >= 0");
  TORCH_CHECK(scale.dim() == 1, "scale should be a 1-D tensor");
  TORCH_CHECK(zero_point.dim() == 1, "zero point should be a 1-D tensor");
  TORCH_CHECK(
      scale.numel() == zero_point.numel(),
      "scale and zero-point need to have the same dimensions");
  TORCH_CHECK(
      scale.numel() == X.size(axis),
      "dimensions of scale and zero-point are not consistent with input tensor")

  TORCH_CHECK(
      at::min(zero_point_rounded).item().toLong() >= quant_min &&
          at::max(zero_point_rounded).item().toLong() <= quant_max,
      "`zero_point` must be between `quant_min` and `quant_max`.");

  TORCH_CHECK(
      axis >= 0 && axis < X.dim(),
      "`axis` must be between 0 and number of dimensions of input");

  if (X.numel() <= 0) {
    return std::make_tuple(X, scale, zero_point);
  }

  auto dX = at::empty_like(X, X.options(), MemoryFormat::Preserve);
  auto dScale_vec = at::empty_like(X, X.options(), MemoryFormat::Preserve);
  auto dZeroPoint_vec = at::empty_like(X, X.options(), MemoryFormat::Preserve);
  int numDimensions = X.ndimension();

  // 创建一个轴掩码，用于将尺度和零点张量向量化并重新形状化为沿着通道轴与 X 相同的形状。
  // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
  int64_t* axis_mask = (int64_t *) calloc(numDimensions, sizeof(int64_t));
  for (const auto i : c10::irange(numDimensions)) {
    axis_mask[i] = (i == axis) ? X.size(axis) : 1;
  }
  auto X_shape = X.sizes();
  auto scale_vectorized = scale.reshape(at::IntArrayRef(axis_mask, numDimensions)).expand(X_shape);
  auto zero_point_vectorized = zero_point_rounded.reshape(at::IntArrayRef(axis_mask, numDimensions)).expand(X_shape);

  auto iter = TensorIteratorConfig()
    .add_output(dX)
    .add_output(dScale_vec)
    .add_output(dZeroPoint_vec)
  .add_input(X)
  .add_input(dY)
  .add_input(scale_vectorized)
  .add_input(zero_point_vectorized)
  .build();

// 调用链式方法，向构建器添加输入张量 X
// 调用链式方法，向构建器添加输入梯度张量 dY
// 调用链式方法，向构建器添加向量化的尺度 scale_vectorized
// 调用链式方法，向构建器添加向量化的零点 zero_point_vectorized
// 完成构建器的配置，并生成可执行的计算图

fake_quant_grad_learnable_channel_stub(
  X.device().type(), iter, quant_min, quant_max, grad_factor);

// 调用虚构的量化梯度学习通道存根函数，处理输入张量 X 的设备类型，迭代次数 iter，
// 量化最小值 quant_min，量化最大值 quant_max，以及梯度因子 grad_factor

auto numElements = X.ndimension() - 1;

// 计算张量 X 的维度数减去 1，并存储在变量 numElements 中

// 创建一个包含所有轴的集合，用于在对 dScale 和 dZeroPoint 张量求和时进行缩减。
// NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
int64_t* axis_for_reduction = (int64_t*) calloc(numElements, sizeof(int64_t));
for (const auto i : c10::irange(axis)) {
  axis_for_reduction[i] = i;
}
for (const auto i : c10::irange(axis, numElements)) {
  axis_for_reduction[i] = i + 1;
}

// 动态分配内存创建一个整数数组 axis_for_reduction，其大小为 numElements，用于指定缩减轴
// 第一个循环初始化包含通道轴之外的所有轴，第二个循环初始化包含通道轴的所有轴

auto dScale = dScale_vec.sum(at::IntArrayRef(axis_for_reduction, numElements));
auto dZeroPoint = dZeroPoint_vec.sum(at::IntArrayRef(axis_for_reduction, numElements));

// 使用 axis_for_reduction 数组对 dScale_vec 和 dZeroPoint_vec 张量进行求和，得到 dScale 和 dZeroPoint

// NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
free(axis_mask);
// 释放动态分配的内存，以释放 axis_mask 数组占用的内存空间
// NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
free(axis_for_reduction);
// 释放动态分配的内存，以释放 axis_for_reduction 数组占用的内存空间

return std::make_tuple(dX, dScale, dZeroPoint);
// 返回一个包含 dX、dScale 和 dZeroPoint 的元组
} // 结束 at 命名空间
} // 结束 native 命名空间
} // 结束 at 命名空间
```