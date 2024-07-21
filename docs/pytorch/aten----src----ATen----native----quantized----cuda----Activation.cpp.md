# `.\pytorch\aten\src\ATen\native\quantized\cuda\Activation.cpp`

```py
// 引入头文件 Exception.h，包含了异常处理的相关定义
// 引入 ATen.h 和 Functions.h，提供了 PyTorch Tensor 操作和函数的定义

// 进入 ATen 命名空间
namespace at {
// 进入 native 命名空间
namespace native {

// 此核心函数当前实现为 dequantize -> fp32 gelu -> quantize，与 int8 gelu 不等价
// 可能可以编写一个等价于 dequantize -> fp32 cuda gelu 核心函数 -> quantize 的 int8 gelu 变体
// 这可以作为未来工作的主题
Tensor gelu_quantized_cuda(const Tensor& qx, c10::string_view approximate) {
  (void)approximate; // 抑制未使用变量的 lint 警告
  // 如果 qx 张量的元素个数为 0，返回一个空 Tensor
  if (qx.numel() == 0) {
    return Tensor{};
  }
  // 将 qx 张量去量化为 float32 类型的张量 x_fp32
  auto x_fp32 = at::dequantize(qx);
  // 对 x_fp32 应用 GELU 激活函数得到 result_fp32
  auto result_fp32 = at::gelu(x_fp32);
  // 将 result_fp32 张量量化为与 qx 张量相同的量化参数，并返回
  return at::quantize_per_tensor(result_fp32, qx.q_scale(), qx.q_zero_point(), qx.scalar_type());
}

// 在 CUDA 上实现的量化 ReLU 函数
Tensor relu_quantized_cuda(const Tensor& self) {
  // 获取 self 张量的零点值
  auto zero_point = self.q_zero_point();
  // 获取 self 张量的整数表示 int_repr
  auto int_repr = self.int_repr();
  // 创建一个掩码，标记 int_repr 中大于零点值的位置
  auto mask = (int_repr > zero_point);
  // 使用掩码选择 int_repr 或零点值，形成量化后的 ReLU 结果 relu_int_repr
  const auto relu_int_repr = at::where(mask, int_repr, zero_point);
  // 根据 relu_int_repr、self 张量的量化参数创建新的量化 Tensor 并返回
  return at::_make_per_tensor_quantized_tensor(relu_int_repr, self.q_scale(), zero_point);
}

}  // namespace at::native
}  // namespace at
```