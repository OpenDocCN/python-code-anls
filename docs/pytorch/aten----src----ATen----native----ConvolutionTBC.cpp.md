# `.\pytorch\aten\src\ATen\native\ConvolutionTBC.cpp`

```
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
// 引入张量库的头文件
#include <ATen/core/Tensor.h>
// 引入范围迭代工具
#include <c10/util/irange.h>
// 引入元组处理
#include <tuple>

#ifndef AT_PER_OPERATOR_HEADERS
// 引入张量函数库
#include <ATen/Functions.h>
// 引入张量本地函数库
#include <ATen/NativeFunctions.h>
#else
// 引入特定操作的头文件
#include <ATen/ops/conv_tbc_backward_native.h>
#include <ATen/ops/conv_tbc_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/zeros_like.h>
#endif

// 定义在 at 命名空间下的 native 子命名空间
namespace at::native {

// TBC 卷积操作，输入张量 self、权重 weight、偏置 bias，填充 pad
Tensor conv_tbc(const Tensor& self, const Tensor& weight, const Tensor& bias, int64_t pad) {
  // 检查输入张量的维度是否符合要求
  TORCH_CHECK(self.dim() == 3, "Input must have 3 dims: time, batch, "
      "in_channel");
  // 检查权重张量的维度是否符合要求
  TORCH_CHECK(weight.dim() == 3, "Weight tensor must have 3 dims: kernel_width,"
      " in_channels, out_channels.");
  // 检查偏置张量的维度是否符合要求
  TORCH_CHECK(bias.dim() == 1, "Bias must be 1-D");

  // 获取输入张量和权重张量的大小信息
  auto input_size = self.sizes();
  auto weight_size = weight.sizes();

  // 提取输入张量的尺寸信息
  auto ilen = input_size[0];
  auto batchSize = input_size[1];
  auto inputPlanes = input_size[2];
  // 提取权重张量的输出通道数信息
  auto outputPlanes = weight_size[2];
  auto kw = weight_size[0];
  // 计算输出张量的长度
  auto olen = input_size[0] - kw + 1 + pad * 2;
  // 计算实际填充量
  auto real_pad = (olen - ilen + kw - 1) / 2;

  // 确保张量的形状符合预期
  // 输入张量形状：(time, batch, in_channels)
  // 权重张量形状：(kernel_width, in_channels, out_channels)
  // 偏置张量形状：(out_channels)
  TORCH_CHECK(inputPlanes == weight_size[1], "Input dim 2 (input channels) "
      "is not == dim 1 in the weight tensor");
  TORCH_CHECK(weight_size[2] == bias.sizes()[0], "Bias size must equal dim 2 in "
      "the weight tensor (output channels).");

  // 创建输出张量，形状为 olen x batchSize x outputPlanes，并使用 self 的选项
  Tensor output = at::empty({
    olen,
    input_size[1],
    weight_size[2],
  }, self.options());
  // 将偏置广播到输出张量的所有元素
  output.copy_(bias.expand(output.sizes()));

  // 对于权重张量的每个 kernel_width，进行 TBC 卷积操作
  for (const auto k : c10::irange(kw)) {
    // 计算输入和输出的偏移量
    int iShift = std::max(0, static_cast<int>(k - real_pad));
    int oShift = std::max(0, static_cast<int>(real_pad - k));
    // 计算当前处理的时间步数 t
    int t = std::min(ilen + real_pad - k, olen) - oShift;

    // 注意：gemm 操作假设列主序列矩阵
    // input    是 l*m (行主序列)
    // weight   是 m*r (行主序列)
    // output   是 l*r (行主序列)
    if (t > 0) {
      auto W = weight[k];
      auto I = self.narrow(0, iShift, t).view({t * batchSize, inputPlanes});
      auto O = output.narrow(0, oShift, t).view({t * batchSize, outputPlanes});
      // 使用 gemm 进行矩阵乘法累加
      O.addmm_(I, W);
    }
  }

  // 返回 TBC 卷积的输出张量
  return output;
}

} // namespace at::native
// 定义一个函数，用于计算 Convolution Transposed Backward 的梯度，返回输入、权重和偏置的梯度
std::tuple<Tensor, Tensor, Tensor> conv_tbc_backward(const Tensor& dOutput, const Tensor& input, const Tensor& weight, const Tensor& bias, int64_t pad) {
  // 获取输入和权重的尺寸信息
  auto input_size = input.sizes();
  auto weight_size = weight.sizes();

  // 提取输入的长度、批次大小和输入平面数量
  auto ilen = input_size[0];
  auto batchSize = input_size[1];
  auto inputPlanes = input_size[2];
  // 提取输出平面数量
  auto outputPlanes = weight_size[2];
  // 提取卷积核大小
  auto kw = weight.sizes()[0];
  // 计算输出长度，考虑填充
  auto olen = input_size[0] - kw + 1 + pad * 2;
  // 计算实际填充量
  int real_pad = (olen - ilen + kw - 1) / 2;

  // 创建一个与输入张量形状相同的零张量，用于存储输入的梯度
  Tensor dInput = at::zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  // 遍历卷积核的每个位置
  for (int k = 0; k < kw; k++) {
    // 计算输入和输出的偏移量，确保不越界
    int iShift = std::max(0, k - real_pad);
    int oShift = std::max(0, real_pad - k);
    // 计算有效长度
    int t = std::min(ilen + real_pad - k, olen) - oShift;
    // 如果有效长度大于零，执行反向传播操作：dOutput * T(weight) -> dInput
    if (t > 0) {
      auto dO = dOutput.narrow(0, oShift, t).view({t * batchSize, outputPlanes});
      auto dI = dInput.narrow(0, iShift, t).view({t * batchSize, inputPlanes});
      dI.addmm_(dO, weight[k].t());
    }
  }

  // 创建一个与权重张量形状相同的零张量，用于存储权重的梯度
  Tensor dWeight = at::zeros_like(weight, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  // 再次遍历卷积核的每个位置
  for (int k = 0; k < kw; k++) {
    // 计算输入和输出的偏移量，确保不越界
    int iShift = std::max(0, k - real_pad);
    int oShift = std::max(0, real_pad - k);
    // 计算有效长度
    int t = std::min(ilen + real_pad - k, olen) - oShift;
    // 如果有效长度大于零，执行反向传播操作：T(input) * dOutput -> dWeight
    if (t > 0) {
      auto dW = dWeight[k];
      auto dO = dOutput.narrow(0, oShift, t).view({t * batchSize, outputPlanes});
      auto I = input.narrow(0, iShift, t).view({t * batchSize, inputPlanes}).t();
      dW.addmm_(I, dO);
    }
  }

  // 创建一个与偏置张量形状相同的零张量，用于存储偏置的梯度
  Tensor dBias = at::zeros_like(bias, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  // 计算输出梯度的和，并复制到偏置的梯度张量中
  auto tmp = dOutput.sum(0, false);
  dBias.copy_(tmp.sum(0));

  // 返回输入、权重和偏置的梯度作为元组
  return std::make_tuple(dInput, dWeight, dBias);
}

} // namespace at::native
```