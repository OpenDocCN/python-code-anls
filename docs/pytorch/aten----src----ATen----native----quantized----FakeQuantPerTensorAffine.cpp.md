# `.\pytorch\aten\src\ATen\native\quantized\FakeQuantPerTensorAffine.cpp`

```py
// 包含 ATen 库的头文件
#include <ATen/ATen.h>
// 包含 ATen 的调度功能
#include <ATen/Dispatch.h>
// 包含 ATen 的原生函数
#include <ATen/NativeFunctions.h>
// 包含 ATen 的张量迭代器
#include <ATen/native/TensorIterator.h>
// 包含 ATen 的 CPU 循环实现
#include <ATen/native/cpu/Loops.h>
// 包含 ATen 的量化仿真函数
#include <ATen/native/quantized/FakeQuantAffine.h>

// ATen 命名空间
namespace at {
// ATen 内部命名空间 native
namespace native {

// 使用 REGISTER_DISPATCH 进行 CPU 和 CUDA 后端的调度
DEFINE_DISPATCH(fake_quant_tensor_cachemask_stub);
DEFINE_DISPATCH(fake_quant_grad_learnable_tensor_stub);
DEFINE_DISPATCH(fake_quant_tensor_cachemask_tensor_qparams_stub);

/* 对 'inputs' 张量进行仿真量化。

Args:
  self: 前向输入张量。
  dY: 反向输入张量（仅限于 _backward 操作）。
  scale: 每张量仿真量化的缩放因子。
  zero_point: 每张量仿真量化的零点。
  quant_min: 最小量化值。
  quant_max: 最大量化值。

Returns:
  量化后的张量（double 类型）。
*/
Tensor fake_quantize_per_tensor_affine(
    const Tensor& self,
    double scale,
    int64_t zero_point,
    int64_t quant_min,
    int64_t quant_max) {
  // 调用带缓存掩码的仿真量化函数
  const auto res = at::fake_quantize_per_tensor_affine_cachemask(
      self, scale, zero_point, quant_min, quant_max);
  return std::get<0>(res);
}

/* 对 'inputs' 张量进行仿真量化，并保存反向传播时使用的掩码。

这与 `fake_quantize_per_tensor_affine` 在数值上是等效的，
但在反向传播时具有更低的内存开销。

Args:
  self: 前向输入张量。
  scale: 每张量仿真量化的缩放因子。
  zero_point: 每张量仿真量化的零点。
  quant_min: 最小量化值。
  quant_max: 最大量化值。

Returns:
  量化后的张量（double 类型）。
  掩码（bool 类型）。
*/
std::tuple<Tensor, Tensor> fake_quantize_per_tensor_affine_cachemask(
    const Tensor& self,
    double scale,
    int64_t zero_point,
    int64_t quant_min,
    int64_t quant_max) {
  // 检查量化范围的有效性
  TORCH_CHECK(
      quant_min <= quant_max,
      "`quant_min` should be less than or equal to `quant_max`.");
  TORCH_CHECK(
      zero_point >= quant_min && zero_point <= quant_max,
      "`zero_point` must be between `quant_min` and `quant_max`.");

  // 创建和输入张量相同尺寸和数据类型的空张量 Y 和掩码 mask
  auto Y = at::empty_like(self, self.options(), MemoryFormat::Preserve);
  auto mask = at::empty_like(self, at::kBool, MemoryFormat::Preserve);
  
  // 调用带缓存掩码的张量仿真量化的底层函数
  fake_quant_tensor_cachemask_stub(
      self.device().type(), Y, mask, self, scale, zero_point, quant_min, quant_max);
  
  // 返回量化后的张量和掩码的元组
  return std::make_tuple(Y, mask);
}

} // namespace native
} // namespace at
std::tuple<Tensor, Tensor> _fake_quantize_per_tensor_affine_cachemask_tensor_qparams(
    const Tensor& self,
    const Tensor& scale,
    const Tensor& zero_point,
    const Tensor& fake_quant_enabled,
    int64_t quant_min,
    int64_t quant_max) {
  // 检查量化范围是否合法
  TORCH_CHECK(
      quant_min <= quant_max,
      "`quant_min` should be less than or \
        equal to `quant_max`.");
  
  // 创建一个与输入张量 `self` 相同大小和类型的空张量 `Y`
  auto Y = at::empty_like(self, self.options(), MemoryFormat::Preserve);
  // 创建一个与输入张量 `self` 相同大小和布尔类型的空掩码张量 `mask`
  auto mask = at::empty_like(self, at::kBool, MemoryFormat::Preserve);
  
  // 调用 C++ 扩展库中的函数来执行张量量化和掩码操作
  fake_quant_tensor_cachemask_tensor_qparams_stub(
      self.device().type(), Y, mask, self, scale, zero_point, fake_quant_enabled, quant_min, quant_max);
  
  // 返回量化后的张量 `Y` 和掩码张量 `mask`
  return std::make_tuple(Y, mask);
}

/* Backward path to fake-quantize the 'inputs' tensor, with mask.

Args:
  dY: output grad.
  mask: mask tensor from the forward pass.

Returns:
  dX (input grad).
*/
Tensor fake_quantize_per_tensor_affine_cachemask_backward(
    const Tensor& dY,
    const Tensor& mask) {
  // 检查掩码张量 `mask` 是否为布尔类型
  TORCH_CHECK(mask.scalar_type() == ScalarType::Bool);
  // 检查掩码张量 `mask` 和梯度张量 `dY` 的大小是否相同
  TORCH_CHECK(mask.sym_numel() == dY.sym_numel(),
      "`mask` and `dY` are not the same size: ",
      "`mask` is size ", mask.sym_numel(), " and `dY` is size ", dY.sym_numel());
  
  // 如果梯度张量 `dY` 的元素数小于等于 0，则直接返回 `dY`
  if (dY.sym_numel() <= 0) {
    return dY;
  }
  
  // 注意：由于掩码已经预先计算好，因此不需要额外的计算核心，可以直接使用现有的张量乘法核心
  // 返回掩码张量 `mask` 与梯度张量 `dY` 的元素乘积结果
  return dY * mask;
}

static int64_t _get_zero_point_from_tensor(
    const Tensor& zero_point,
    int64_t quant_min,
    int64_t quant_max,
    bool is_forward) {
  // 获取张量 `zero_point` 中的浮点值
  float zero_point_fp = zero_point[0].item<float>();
  
  // 根据是否为前向传播过程，调整零点值的计算方法
  zero_point_fp = is_forward ? std::nearbyint(zero_point_fp) : zero_point_fp + 0.5f;
  
  // 将计算出的浮点零点值限制在指定的量化范围内，并转换为整数返回
  float zero_point_clamped = std::min(std::max(zero_point_fp, static_cast<float>(quant_min)),
                                       static_cast<float>(quant_max));
  return static_cast<int64_t>(zero_point_clamped);
}

Tensor _fake_quantize_learnable_per_tensor_affine(
    const Tensor& self,
    const Tensor& scale,
    const Tensor& zero_point,
    int64_t quant_min,
    int64_t quant_max,
    double grad_factor) {
  // 获取标量 `scale` 中的浮点值
  float scale_val = scale[0].item<float>();
  
  // 从张量 `zero_point` 中获取整数类型的零点值，并进行向前传播的零点调整
  int64_t zero_point_val = native::_get_zero_point_from_tensor(zero_point, quant_min, quant_max, true);
  
  // 调用 C++ 扩展库中的函数来执行张量的可学习量化操作
  return native::fake_quantize_per_tensor_affine(
    self, scale_val, zero_point_val, quant_min, quant_max);
}

std::tuple<Tensor, Tensor, Tensor> _fake_quantize_learnable_per_tensor_affine_backward(
    const Tensor& dY,
    const Tensor& X,
    const Tensor& scale,
    const Tensor& zero_point,
    int64_t quant_min,
    int64_t quant_max,
  /* 计算尺度和零点的梯度如下：
     设 Xfq 为 X 的伪量化版本。
     设 Xq 为 X 的量化版本（在 qmin 和 qmax 处截断）。
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
  // 从 scale 中获取尺度值
  float scale_val = scale[0].item<float>();
  // 计算尺度的倒数
  float inv_scale_val = 1.0f / scale_val;
  // 从 zero_point 中获取零点值
  int64_t zero_point_val = native::_get_zero_point_from_tensor(zero_point, quant_min, quant_max, false);

  // 检查输入张量的数据类型是否为 Float
  TORCH_CHECK(dY.scalar_type() == ScalarType::Float);
  TORCH_CHECK(X.scalar_type() == ScalarType::Float);
  TORCH_CHECK(scale.scalar_type() == ScalarType::Float);
  TORCH_CHECK(zero_point.scalar_type() == ScalarType::Float);
  // 检查 X 和 dY 的元素数量是否相等
  TORCH_CHECK(X.numel() == dY.numel(), "`X` and `dY` are not the same size");
  // 检查量化范围的有效性
  TORCH_CHECK(
      quant_min <= 0 && quant_max >= 0,
      "`quant_min` should be less than or \
        equal to `quant_max`, and the quantization range should include 0.");
  // 检查零点值是否在量化范围内
  TORCH_CHECK(
      zero_point_val >= quant_min && zero_point_val <= quant_max,
      "`zero_point` must be between `quant_min` and `quant_max`.");
  
  // 如果 X 的元素数量为 0，则直接返回 X、scale 和 zero_point
  if (X.numel() <= 0) {
    return std::make_tuple(X, scale, zero_point);
  }

  // 创建与 X 相同大小和类型的张量来存储梯度
  auto dX = at::empty_like(X, X.options(), MemoryFormat::Preserve);
  auto dScale_vec = at::empty_like(X, X.options(), MemoryFormat::Preserve);
  auto dZeroPoint_vec = at::empty_like(X, X.options(), MemoryFormat::Preserve);

  // 配置张量迭代器，用于遍历 X 和 dY 并输出 dX、dScale_vec 和 dZeroPoint_vec
  auto iter = TensorIteratorConfig()
    .add_output(dX)
    .add_output(dScale_vec)
    .add_output(dZeroPoint_vec)
    .add_input(X)
    .add_input(dY)
    .build();

  // 调用伪量化梯度学习函数来计算梯度
  fake_quant_grad_learnable_tensor_stub(
    X.device().type(), iter, scale_val, inv_scale_val, zero_point_val, quant_min, quant_max, grad_factor);

  // 对尺度梯度向量求和并扩展维度，以便与 scale 的设备兼容
  auto dScale = dScale_vec.sum().unsqueeze(0).to(scale.device());
  // 对零点梯度向量求和并扩展维度，以便与 zero_point 的设备兼容
  auto dZeroPoint = dZeroPoint_vec.sum().unsqueeze(0).to(zero_point.device());

  // 返回计算得到的梯度张量组成的元组
  return std::make_tuple(dX, dScale, dZeroPoint);
} // 结束 at 命名空间

} // 结束 native 命名空间
} // 结束 at 命名空间
```