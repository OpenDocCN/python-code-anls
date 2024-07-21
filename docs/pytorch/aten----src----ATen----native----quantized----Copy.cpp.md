# `.\pytorch\aten\src\ATen\native\quantized\Copy.cpp`

```py
// 包含 ATen 库中的必要头文件
#include <ATen/ATen.h>
// 包含 ATen 库中量化相关的头文件
#include <ATen/native/quantized/AffineQuantizer.h>
// 包含 ATen 库中的拷贝操作头文件
#include <ATen/native/quantized/Copy.h>
// 包含 c10 核心库中的内存格式定义
#include <c10/core/MemoryFormat.h>
// 包含 c10 工具库中的迭代范围工具
#include <c10/util/irange.h>

// 定义 ATen 命名空间
namespace at {
// 定义 ATen 库中的 native 命名空间
namespace native {

// 从浮点数 Tensor 复制到 QInt Tensor，用于将浮点值赋给量化 Tensor
// 第二个异常条件 `self.is_contiguous() && src.is_contiguous()` 要求 self 和 src 张量都是连续的
// 这意味着目前不支持对非连续量化子张量的赋值操作
// 例如，考虑一个 2x2 的量化 Tensor qt1 和一个非量化 Tensor t2。操作 `qt1[:, 0] = t2[:, 0]` 将触发异常，因为左右张量都不是连续的
Tensor& quantized_copy_from_float_(Tensor& self, const Tensor& src) {
  // 检查源 Tensor 是否为 kFloat 类型，只能使用 kFloat 作为源 Tensor 进行量化拷贝
  TORCH_CHECK(
      src.scalar_type() == at::kFloat,
      "Quantized copy only works with kFloat as source Tensor");
  // 检查是否 self 和 src 张量都是连续的或者都是 NHWC 内存格式的连续张量
  TORCH_CHECK(
      (self.is_contiguous() && src.is_contiguous()) ||
      (self.is_contiguous(at::MemoryFormat::ChannelsLast) && src.is_contiguous(at::MemoryFormat::ChannelsLast)),
      "Quantized copy only works with contiguous and NHWC Tensors");
  // 检查 self 和 src 张量是否具有相同的形状
  TORCH_CHECK(
      self.sizes().equals(src.sizes()),
      "Quantized copy only works with Tensors with the same shape");
  
  // 根据 self 的数据类型分发到不同的量化函数
  AT_DISPATCH_QINT_TYPES(self.scalar_type(), "Copy", [&]() {
    if (self.qscheme() == kPerChannelAffine || self.qscheme() == kPerChannelAffineFloatQParams
        || self.qscheme() == kPerChannelSymmetric) {
      // 如果量化方案为按通道的仿射量化，则调用相应的函数进行量化
      quantize_tensor_per_channel_affine(src, self, self.q_per_channel_scales(),
                                         self.q_per_channel_zero_points(),
                                         self.q_per_channel_axis());
    } else {
      // 否则使用按张量的仿射量化进行量化
      quantize_tensor_per_tensor_affine(src, self, self.q_scale(), self.q_zero_point());
    }
  });
  
  // 返回量化后的 self 张量
  return self;
}

} // namespace native
} // namespace at


这些注释解释了代码中每个重要部分的功能和条件，确保了代码的逻辑和使用方式清晰可读。
```